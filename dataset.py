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

# Giả sử bạn đã có 3 hàm này trong file utils.py
# from utils import fill_label_holes, edt_prob, star_dist

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

        # maxfilter_patch_size để kiểm tra foreground
        self.maxfilter_patch_size = maxfilter_patch_size or self.patch_size

        # Cache valid indices
        self._ind_cache_fg = {}
        self._ind_cache_all = {}
        self.lock = threading.Lock()

        # Grid mặc định (1,1) như bạn yêu cầu
        self.grid = (1, 1)
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in self.grid)

    def __len__(self):
        return len(self.image_paths)

    # Trong class StarDistDataset2D

    def get_valid_inds(self, idx, foreground_only=False):
        """Lấy vị trí crop hợp lệ, đảm bảo patch không vượt quá ảnh"""
        key = (idx, foreground_only)
        cache = self._ind_cache_fg if foreground_only else self._ind_cache_all

        if key in cache:
            return cache[key]

        mask = tifffile.imread(self.mask_paths[idx])
        mask = fill_label_holes(mask)

        h, w = mask.shape
        ph, pw = self.patch_size

        # Nếu ảnh nhỏ hơn patch_size → fallback dùng toàn bộ ảnh (sẽ pad sau)
        if h < ph or w < pw:
            print(f"Ảnh {self.image_paths[idx]} nhỏ hơn patch_size ({h}x{w} < {ph}x{pw}), dùng toàn bộ ảnh")
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
        img = tifffile.imread(self.image_paths[idx]).astype(np.float32)
        mask = tifffile.imread(self.mask_paths[idx]).astype(np.int32)

        if img.ndim == 2:
            img = img[..., None]

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

        # Pad nếu patch nhỏ hơn patch_size (rất quan trọng!)
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
            img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min() + 1e-8)

        # Tính prob & dist (trên patch đã pad đúng size)
        prob = edt_prob(mask_patch)
        dist = star_dist(mask_patch, n_rays=self.n_rays)
        dist_mask = prob.copy()

        dist_and_mask = np.concatenate([dist, dist_mask[..., None]], axis=-1)

        # To torch
        img_t = torch.from_numpy(img_patch).permute(2, 0, 1).float()
        prob_t = torch.from_numpy(prob[..., None]).permute(2, 0, 1).float()
        dist_and_mask_t = torch.from_numpy(dist_and_mask).permute(2, 0, 1).float()

        return img_t, prob_t, dist_and_mask_t

def custom_collate(batch):
    images, probs, dist_masks = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(probs),
        torch.stack(dist_masks)
    )


# ====================== Hàm tạo DataLoader ======================
def create_dataloaders(
    root_dir="data/dsb2018/train/",
    patch_size=(256, 256),
    batch_size=8,
    foreground_prob=0.9,
    num_workers=0,
    pin_memory=True,
    val_split_ratio=0.2,  # nếu bạn muốn tự split, nhưng bạn nói đã chia sẵn nên có thể bỏ
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
        augmenter=augmenter,  # hàm bạn đã cung cấp
        normalize=True,
    )

    # Nếu bạn đã có split sẵn (train/test folder riêng), thì tạo 2 dataset
    # Còn nếu muốn random split trong dataset này:
    train_size = int((1 - val_split_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=pin_memory,
    )
 
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


# ====================== Hàm augmenter của bạn (đã copy) ======================
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
    for images, probs, dist_masks in train_loader:
        print(images.shape)         # torch.Size([8, 1 or 3, 256, 256])
        print(probs.shape)          # torch.Size([8, 1, 256, 256])   # grid=1,1 nên bằng patch
        print(dist_masks.shape)     # torch.Size([8, 33, 256, 256])  # 32 rays + 1 mask
        break
    