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


def star_dist(a: np.ndarray, n_rays: int = 32, grid=(1, 1)) -> np.ndarray:
    """
    Tính khoảng cách radial từ mỗi pixel đến biên giới object theo n_rays hướng.
    Tối ưu hóa bằng cách xử lý riêng từng object và dùng vectorization.
    """
    if grid != (1, 1):
        raise NotImplementedError("grid != (1, 1) chưa được hỗ trợ")

    n_rays = int(n_rays)
    a = a.astype(np.uint16, copy=False)
    dst = np.zeros(a.shape + (n_rays,), dtype=np.float32)

    # Precompute angles
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dys = np.cos(angles)
    dxs = np.sin(angles)

    # Tìm các object
    objects = find_objects(a)
    
    for label_id, sl in enumerate(objects, 1):
        if sl is None:
            continue
            
        # Lấy mask của object hiện tại
        mask = (a[sl] == label_id)
        if not np.any(mask):
            continue

        # Tọa độ các pixel foreground trong bounding box
        coords = np.argwhere(mask)
        
        # Với mỗi pixel foreground, tính radial distance
        for r, c in coords:
            # Tọa độ thực trong ảnh gốc
            y0, x0 = r + sl[0].start, c + sl[1].start
            
            for k in range(n_rays):
                dy, dx = dys[k], dxs[k]
                
                # Ước lượng bước nhảy dựa trên distance transform (nếu đã tính)
                # Ở đây ta dùng ray marching đơn giản nhưng giới hạn trong bounding box của object
                t = 0.0
                while True:
                    t += 1.0
                    y = int(round(y0 + t * dy))
                    x = int(round(x0 + t * dx))
                    
                    if (y < 0 or y >= a.shape[0] or 
                        x < 0 or x >= a.shape[1] or 
                        a[y, x] != label_id):
                        
                        # Điều chỉnh chính xác hơn biên giới
                        t_fine = t - 1.0
                        for _ in range(3): # Bisection
                            t_mid = t_fine + 0.5
                            ym = int(round(y0 + t_mid * dy))
                            xm = int(round(x0 + t_mid * dx))
                            if (0 <= ym < a.shape[0] and 0 <= xm < a.shape[1] and 
                                a[ym, xm] == label_id):
                                t_fine = t_mid
                            else:
                                t = t_mid
                            t_mid = (t_fine + t) / 2
                        
                        dst[y0, x0, k] = t_fine
                        break
    return dst



