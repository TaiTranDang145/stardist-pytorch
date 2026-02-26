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
    Tối ưu hóa bằng cách vector hóa quá trình ray marching cho toàn bộ pixel của object.
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

    objects = find_objects(a)
    
    for label_id, sl in enumerate(objects, 1):
        if sl is None:
            continue
            
        mask = (a[sl] == label_id)
        if not np.any(mask):
            continue

        # Tọa độ gốc của bounding box
        y_off, x_off = sl[0].start, sl[1].start
        h, w = mask.shape
        
        # Với mỗi hướng ray
        for k in range(n_rays):
            dy, dx = dys[k], dxs[k]
            
            # Khởi tạo distance cho object hiện tại trong hướng k
            # Chỉ tính cho các pixel thuộc mask
            res = np.zeros((h, w), dtype=np.float32)
            
            # "Marching" đồng thời cho tất cả pixel trong mask
            # Chúng ta sẽ kiểm tra các bước t = 1, 2, 4, 8... hoặc đơn giản là t++
            # Để tối ưu hơn, ta có thể dùng search hoặc đơn giản là bước 1.0
            
            # Mảng active: pixel nào vẫn đang ở trong object
            active = mask.copy()
            t = 0.0
            step = 1.0
            
            while np.any(active):
                t += step
                # Tọa độ mới của tất cả pixel sau khi nhảy bước t
                # Dùng vectorization của NumPy
                # y_new = y_orig + t * dy
                
                # Tạo bản đồ dịch chuyển
                y_look = np.round(np.arange(h)[:, None] + t * dy).astype(int) + y_off
                x_look = np.round(np.arange(w)[None, :] + t * dx).astype(int) + x_off
                
                # Mask kiểm tra biên giới ảnh
                within_bounds = (y_look >= 0) & (y_look < a.shape[0]) & \
                                (x_look >= 0) & (x_look < a.shape[1])
                                
                # Cập nhật active: pixel vẫn trong object nếu y_look, x_look vẫn mang label_id
                current_active = np.zeros((h, w), dtype=bool)
                # Chỉ check những pixel đang active
                y_idx, x_idx = np.where(active)
                if len(y_idx) > 0:
                    yl = y_look[y_idx, 0] # y_look is (h,1), x_look is (1,w)
                    xl = x_look[0, x_idx]
                    
                    # Lấy các pixel tại vị trí nhìn tới
                    # Cẩn thận: y_look[y_idx] lấy hàng y_idx, x_look[x_idx] lấy cột x_idx
                    # Ta cần mapping (y_idx, x_idx) -> (y_look[y_idx], x_look[x_idx])
                    
                    # Re-calculate mapped coordinates for active pixels only
                    ya = np.round(y_idx + t * dy).astype(int) + y_off
                    xa = np.round(x_idx + t * dx).astype(int) + x_off
                    
                    valid = (ya >= 0) & (ya < a.shape[0]) & (xa >= 0) & (xa < a.shape[1])
                    # Nếu out of bounds hoặc khác label -> stop
                    
                    still_in = np.zeros(len(y_idx), dtype=bool)
                    if np.any(valid):
                        still_in[valid] = (a[ya[valid], xa[valid]] == label_id)
                    
                    # Lưu lại t cho những pixel vừa "thoát" ra
                    newly_stopped = active.copy()
                    newly_stopped[y_idx, x_idx] = ~still_in
                    
                    # Cập nhật kết quả cho những thằng vừa stop
                    # Dùng t - 0.5 as approximation hoặc bisection nếu muốn chính xác cực cao
                    # Ở đây dùng t_corr đơn giản như bản gốc
                    if np.any(newly_stopped):
                        res[newly_stopped & mask] = t - 0.5
                    
                    active[y_idx, x_idx] = still_in
            
            dst[sl][mask] = res[mask]

    return dst



