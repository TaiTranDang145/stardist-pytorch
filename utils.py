import torch
import numpy as np
from scipy.ndimage import find_objects, binary_fill_holes, distance_transform_edt
import warnings


def fill_label_holes(lbl_img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Điền lỗ hổng bên trong từng instance riêng biệt.
    Giữ nguyên các pixel negative label (-1) để ignore trong loss.
    
    Logic:
    - Tìm bounding box của từng object bằng find_objects
    - Với mỗi object: mở rộng vùng 1 pixel → fill holes → thu lại → gán label
    - Xử lý riêng negative regions (ignore) bằng cách đệ quy
    """
    def grow(sl, interior):
        # Mở rộng slice thêm 1 pixel ở các cạnh trong
        return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) 
                     for s, w in zip(sl, interior))

    def shrink(interior):
        # Thu lại slice về kích thước gốc
        return tuple(slice(int(w[0]), (-1 if int(w[1]) else None)) 
                     for w in interior)

    obj = find_objects(lbl_img)
    lbl_filled = np.zeros_like(lbl_img)

    for i, sl in enumerate(obj, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink(interior)]
        lbl_filled[sl][mask_filled] = i

    # Xử lý vùng negative label (ignore regions, thường = -1)
    if lbl_img.min() < 0:
        neg_part = -np.minimum(lbl_img, 0)           # lấy phần negative
        neg_filled = fill_label_holes(neg_part)      # đệ quy fill holes
        lbl_neg_filled = -neg_filled
        mask_neg = lbl_neg_filled < 0
        lbl_filled[mask_neg] = lbl_neg_filled[mask_neg]

    return lbl_filled


def edt_prob(lbl_img: np.ndarray, anisotropy=None) -> np.ndarray:
    """
    Tính probability map bằng cách normalize Euclidean Distance Transform (EDT)
    cho từng object riêng biệt.
    
    Ý nghĩa: 
    - Pixel càng xa biên giới object → prob càng cao (gần trung tâm)
    - Đây là ground-truth cho đầu ra 'prob' của mô hình (object probability)
    
    Logic:
    - Với mỗi object: mở rộng bounding box → tính EDT → normalize → gán lại
    """
    def grow(sl, interior):
        # Mở rộng slice thêm 1 pixel ở các cạnh trong
        return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) 
                     for s, w in zip(sl, interior))

    def shrink(interior):
        # Thu lại slice về kích thước gốc
        return tuple(slice(int(w[0]), (-1 if int(w[1]) else None)) 
                     for w in interior)
    
    lbl_img = np.asarray(lbl_img)

    # Trường hợp đặc biệt: toàn bộ ảnh là 1 object → padding để EDT hợp lệ
    constant_img = (lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0)
    if constant_img:
        warnings.warn("EDT of constant label image is ill-defined. Padding assumed.")
        lbl_img = np.pad(lbl_img, ((1,1),) * lbl_img.ndim, mode='constant')

    objects = find_objects(lbl_img)
    prob = np.zeros(lbl_img.shape, dtype=np.float32)

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        grown_mask = (lbl_img[grow(sl, interior)] == i)
        shrink_slice = shrink(interior)

        edt = distance_transform_edt(grown_mask, sampling=anisotropy)
        edt_shrunk = edt[shrink_slice]
        mask = grown_mask[shrink_slice]

        # Normalize: chia cho max EDT trong object → prob từ 0 đến 1
        prob[sl][mask] = edt_shrunk[mask] / (np.max(edt_shrunk) + 1e-10)

    # Bỏ padding nếu có
    if constant_img:
        prob = prob[(slice(1, -1),) * lbl_img.ndim].copy()

    return prob


def star_dist(a: np.ndarray, n_rays: int = 32, grid=(1,1)) -> np.ndarray:
    """
    Tính khoảng cách radial từ mỗi pixel đến biên giới object theo n_rays hướng.
    Đây là ground-truth cho đầu ra 'dist' của mô hình (n_rays kênh).
    
    Logic gốc: ray marching từ tâm pixel theo từng hướng cho đến khi ra khỏi object.
    """
    if grid != (1,1):
        raise NotImplementedError("grid != (1,1) chưa được hỗ trợ trong phiên bản này")

    n_rays = int(n_rays)
    a = a.astype(np.uint16, copy=False)
    dst = np.empty(a.shape + (n_rays,), dtype=np.float32)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i, j]
            if value == 0:
                dst[i, j] = 0
                continue

            angle_step = np.float32(2 * np.pi / n_rays)
            for k in range(n_rays):
                phi = k * angle_step
                dy = np.cos(phi)
                dx = np.sin(phi)
                x, y = np.float32(0), np.float32(0)

                while True:
                    x += dx
                    y += dy
                    ii = int(round(i + x))
                    jj = int(round(j + y))

                    if (ii < 0 or ii >= a.shape[0] or
                        jj < 0 or jj >= a.shape[1] or
                        value != a[ii, jj]):
                        # Điều chỉnh nhỏ vì thường overshoot 1 bước
                        t_corr = 1 - 0.5 / max(abs(dx), abs(dy))
                        x -= t_corr * dx
                        y -= t_corr * dy
                        dist = np.sqrt(x**2 + y**2)
                        dst[i, j, k] = dist
                        break

    return dst


def star_dist_torch(labels: torch.Tensor, n_rays: int = 32, max_steps: int = 512) -> torch.Tensor:
    """
    Phiên bản PyTorch vectorized của star_dist (ray marching).
    Hiện tại vẫn loop theo ray nhưng vectorized theo pixel → nhanh hơn nhiều so với pure Python.
    
    Input:
        labels: torch.Tensor shape (B, H, W) hoặc (H, W) - instance labels
    Output:
        dist:   torch.Tensor shape (B, H, W, n_rays)
    """
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)  # thêm batch dim
    B, H, W = labels.shape
    device = labels.device
    dtype = torch.float32

    # Tạo các hướng radial
    angles = torch.linspace(0, 2 * np.pi, n_rays, device=device, dtype=dtype)
    dirs_y = torch.cos(angles)  # (n_rays,)
    dirs_x = torch.sin(angles)

    # Grid tọa độ pixel
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    yy = yy[None].expand(B, -1, -1)  # (B, H, W)
    xx = xx[None].expand(B, -1, -1)

    dist = torch.zeros(B, H, W, n_rays, device=device, dtype=dtype)

    # Marching theo từng ray
    for k in range(n_rays):
        dy = dirs_y[k]
        dx = dirs_x[k]

        pos_y = yy.clone()
        pos_x = xx.clone()
        current_label = labels.clone()

        for step in range(max_steps):
            pos_y += dy
            pos_x += dx

            ii = torch.clamp(pos_y.round().long(), 0, H - 1)
            jj = torch.clamp(pos_x.round().long(), 0, W - 1)

            # Lấy label tại vị trí mới
            next_label = torch.gather(labels.flatten(1), 1, 
                                     (ii * W + jj).flatten(1)).view(B, H, W)

            # Nơi nào ra khỏi object hoặc chạm biên → ghi nhận khoảng cách
            exited = (next_label != current_label) | (next_label == 0)

            dist[..., k] = torch.where(exited,
                                       torch.hypot(pos_y - yy, pos_x - xx),
                                       dist[..., k])

            if exited.all():
                break

            # Cập nhật vị trí và label cho bước tiếp theo
            pos_y = torch.where(exited, pos_y, pos_y + dy)
            pos_x = torch.where(exited, pos_x, pos_x + dx)
            current_label = torch.where(exited, current_label, next_label)

    return dist.squeeze(0) if B == 1 else dist