import torch
from stardist.geometry import star_dist
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




def _py_star_dist_demo(a, n_rays=32, grid=(1, 1)):
    """Phiên bản Python của star_dist (chỉ dùng để demo/tham khảo)"""
    if not (np.isscalar(n_rays) and 0 < int(n_rays)):
        raise ValueError("n_rays must be a positive integer")
    if grid != (1, 1):
        raise NotImplementedError(f"grid {grid} not supported in Python version")
    
    n_rays = int(n_rays)
    a = a.astype(np.uint16, copy=False)
    dst = np.empty(a.shape + (n_rays,), np.float32)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i, j]
            if value == 0:
                dst[i, j] = 0
            else:
                st_rays = np.float32((2 * np.pi) / n_rays)
                for k in range(n_rays):
                    phi = np.float32(k * st_rays)
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
                            # small correction as we overshoot the boundary
                            t_corr = 1 - .5 / max(np.abs(dx), np.abs(dy))
                            x -= t_corr * dx
                            y -= t_corr * dy
                            dist = np.sqrt(x**2 + y**2)
                            dst[i, j, k] = dist
                            break
    return dst