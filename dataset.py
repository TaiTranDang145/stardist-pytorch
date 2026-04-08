import os
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tifffile
import threading
from scipy.ndimage import maximum_filter
from utils import fill_label_holes, edt_prob, star_dist
from fourier_utils import rays_to_fourier


class StarDistDataset2D(Dataset):
    """
    PyTorch Dataset cho StarDist 2D trên DSB2018.
    - Load ảnh .tif và mask instance .tif
    - Crop patch ngẫu nhiên với foreground sampling (prob=0.9)
    - Tính prob (EDT normalized) và dist (radial 32 rays)
    - Áp dụng augmentation (fliprot + intensity + noise)
    - Trả về (image, prob, dist_and_mask)
    """
    def __init__(
        self,
        image_paths,          # list[str]: đường dẫn ảnh .tif
        mask_paths,           # list[str]: đường dẫn mask .tif
        patch_size=(256, 256),
        n_rays=32,
        foreground_prob=0.9,
        augmenter=None,       # hàm augmenter(x, y) -> (x_aug, y_aug)
        normalize=True,       # normalize image về [0,1]
        cache_valid_inds=True,
        maxfilter_patch_size=None,
        grid=(1, 1),
        n_harmonics=16,       # Số lượng hài Fourier
    ):
        assert len(image_paths) == len(mask_paths), "Số lượng ảnh và mask phải bằng nhau"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = tuple(patch_size)
        self.n_rays = n_rays
        self.foreground_prob = foreground_prob
        self.augmenter = augmenter if augmenter is not None else lambda x, y: (x, y)
        self.normalize = normalize
        self.cache_valid_inds = cache_valid_inds
        self.n_harmonics = n_harmonics
        self.lock = threading.Lock()
        
        # RAM Cache
        self.img_cache = {}
        self.mask_cache = {}

        # maxfilter_patch_size để kiểm tra foreground
        self.maxfilter_patch_size = maxfilter_patch_size or self.patch_size

        # Cache valid indices
        self._ind_cache_fg = {}
        self._ind_cache_all = {}
        # Grid chuyển xuống __getitem__
        self.grid = tuple(grid)
        self.ss_grid = tuple(slice(0, None, g) for g in self.grid)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Không pickle lock vì nó không picklable
        if 'lock' in state:
            del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Tạo lại lock ở process con
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.image_paths)

    # Trong class StarDistDataset2D

    def get_valid_inds(self, idx, foreground_only=False):
        """Lấy vị trí crop hợp lệ, đảm bảo patch không vượt quá ảnh"""
        key = (idx, foreground_only)
        cache = self._ind_cache_fg if foreground_only else self._ind_cache_all

        if key in cache:
            return cache[key]

        mask = self.mask_cache[idx]
        mask = fill_label_holes(mask)

        h, w = mask.shape
        ph, pw = self.patch_size

        # Nếu ảnh nhỏ hơn patch_size → fallback dùng toàn bộ ảnh (sẽ pad sau)
        if h < ph or w < pw:
            valid_y = np.array([0])
            valid_x = np.array([0])
        else:
            if foreground_only:
                fg_mask = (mask > 0).astype(np.uint8)
                fg_max = maximum_filter(fg_mask, size=self.maxfilter_patch_size)
                valid_y, valid_x = np.where(fg_max > 0)
            else:
                valid_y = np.arange(h - ph + 1)
                valid_x = np.arange(w - pw + 1)

            if len(valid_y) == 0:
                valid_y = np.array([0])
                valid_x = np.array([0])

        inds = (valid_y, valid_x)

        if self.cache_valid_inds:
            with self.lock:
                cache[key] = inds

        return inds


    def __getitem__(self, idx):
        if idx not in self.img_cache:
            img = tifffile.imread(self.image_paths[idx]).astype(np.float32)
            if img.ndim == 2:
                img = img[..., None]
            self.img_cache[idx] = img
            self.mask_cache[idx] = tifffile.imread(self.mask_paths[idx]).astype(np.int32)
            
        img = self.img_cache[idx]
        mask = self.mask_cache[idx]

        h, w = mask.shape
        ph, pw = self.patch_size

        use_fg = np.random.rand() < self.foreground_prob
        valid_y, valid_x = self.get_valid_inds(idx, foreground_only=use_fg)

        # Chọn vị trí crop (luôn có ít nhất 1)
        rand_idx = np.random.randint(0, len(valid_y))
        y_start = valid_y[rand_idx]
        x_start = valid_x[rand_idx]

        # Crop (nếu vượt quá ảnh → crop đến biên)
        y_end = min(y_start + ph, h)
        x_end = min(x_start + pw, w)
        img_patch = img[y_start:y_end, x_start:x_end]
        mask_patch = mask[y_start:y_end, x_start:x_end]

        
        pad_h = ph - img_patch.shape[0]
        pad_w = pw - img_patch.shape[1]
        if pad_h > 0 or pad_w > 0:
            pad_img = ((0, pad_h), (0, pad_w), (0, 0)) if img_patch.ndim == 3 else ((0, pad_h), (0, pad_w))
            pad_mask = ((0, pad_h), (0, pad_w))
            img_patch = np.pad(img_patch, pad_img, mode='constant', constant_values=0)
            mask_patch = np.pad(mask_patch, pad_mask, mode='constant', constant_values=0)

        # Augmentation
        img_patch, mask_patch = self.augmenter(img_patch, mask_patch)

        # Normalize
        if self.normalize:
            # Paper suggests 1st and 99th percentiles for robustness
            pmin = np.percentile(img_patch, 1)
            pmax = np.percentile(img_patch, 99)
            img_patch = (img_patch - pmin) / (pmax - pmin + 1e-8)
            img_patch = np.clip(img_patch, 0, 1)

        # Tính prob & dist (trên patch đã pad đúng size)
        prob = edt_prob(mask_patch)
        dist = star_dist(mask_patch, n_rays=self.n_rays)
        dist_mask = prob.copy()

        # Tính Fourier coefficients từ dist
        # dist: (H, W, n_rays)
        coeffs = rays_to_fourier(dist, n_harmonics=self.n_harmonics)
        # coeffs: (H, W, n_harmonics + 1) phức
        # Chuyển thành thực bằng cách tách Real và Imag
        coeffs_real = np.real(coeffs)
        coeffs_imag = np.imag(coeffs)
        fourier_gt = np.concatenate([coeffs_real, coeffs_imag], axis=-1)

        # Tính Complexity GT: Tổng năng lượng các hài bậc cao (n > 2)
        # Bậc cao là từ index 3 trở đi trong coeffs
        if self.n_harmonics > 2:
            high_freq_energy = np.sum(np.abs(coeffs[..., 3:])**2, axis=-1)
            total_energy = np.sum(np.abs(coeffs)**2, axis=-1) + 1e-10
            complexity_gt = high_freq_energy / total_energy
            # Chuẩn hóa về [0, 1] (có thể dùng log hoặc sigmoid-like scaling)
            complexity_gt = np.clip(complexity_gt * 10, 0, 1) # Giả sử 10% năng lượng cao là cực kỳ phức tạp
        else:
            complexity_gt = np.zeros_like(prob)
        
        # Thêm complexity vào fourier_gt tensor hoặc trả về riêng
        fourier_gt = np.concatenate([fourier_gt, complexity_gt[..., None]], axis=-1)
        dist_and_mask = np.concatenate([dist, dist_mask[..., None]], axis=-1)
        
        # To torch
        img_t = torch.from_numpy(img_patch).permute(2, 0, 1).float()
        
        # Downsample labels if grid > 1
        if any(g > 1 for g in self.grid):
            prob = prob[self.ss_grid]
            dist = dist[self.ss_grid]
            dist_mask = dist_mask[self.ss_grid]
            fourier_gt = fourier_gt[self.ss_grid]
            dist_and_mask = np.concatenate([dist, dist_mask[..., None]], axis=-1)

        prob_t = torch.from_numpy(prob[..., None]).permute(2, 0, 1).float()
        dist_and_mask_t = torch.from_numpy(dist_and_mask).permute(2, 0, 1).float()
        fourier_t = torch.from_numpy(fourier_gt).permute(2, 0, 1).float()

        return img_t, prob_t, dist_and_mask_t, fourier_t


def custom_collate(batch):
    images, probs, dist_masks, fourier_coeffs = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(probs),
        torch.stack(dist_masks),
        torch.stack(fourier_coeffs)
    )


# Hàm tạo DataLoader
def create_dataloaders(
    root_dir="data/dsb2018/train/",
    patch_size=(256, 256),
    batch_size=8,
    foreground_prob=0.9,
    num_workers=0,
    pin_memory=True,
    val_split_ratio=0.2,
):
    image_paths = sorted(glob.glob(os.path.join(root_dir, "images/*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(root_dir, "masks/*.tif")))

    assert len(image_paths) == len(mask_paths), "Số lượng ảnh và mask không khớp"

    # Tạo dataset
    dataset = StarDistDataset2D(
        image_paths=image_paths,
        mask_paths=mask_paths,
        patch_size=patch_size,
        n_rays=32,
        foreground_prob=foreground_prob,
        augmenter=augmenter, 
        normalize=True,
        grid=(1, 1),
        n_harmonics=16,
    )

    train_size = int((1 - val_split_ratio) * len(dataset))
    
    indices = list(range(len(dataset)))
    train_ds = torch.utils.data.Subset(dataset, indices[:train_size])
    val_ds = torch.utils.data.Subset(dataset, indices[train_size:])

    val_image_paths = [image_paths[i] for i in indices[train_size:]]
    val_mask_paths = [mask_paths[i] for i in indices[train_size:]]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
 
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, val_image_paths, val_mask_paths


# Hàm augmenter
def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img

def augmenter(x, y):
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


if __name__ == "__main__":
    # Ví dụ chạy
    train_loader, val_loader = create_dataloaders(
        root_dir="data/dsb2018/train/",
        patch_size=(256, 256),
        batch_size=8,
        foreground_prob=0.9,
    )

    # Lấy 1 batch
    for images, probs, dist_masks, fourier_coeffs in train_loader:
        print(images.shape)         # torch.Size([8, 1, 256, 256])
        print(probs.shape)          # torch.Size([8, 1, 256, 256])
        print(dist_masks.shape)     # torch.Size([8, 33, 256, 256])
        print(fourier_coeffs.shape) # torch.Size([8, 35, 256, 256])  # 2 * (16 + 1) + 1 (complexity)
        break
    