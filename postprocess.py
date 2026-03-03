import numpy as np
import torch
from skimage.measure import regionprops
from scipy.ndimage import zoom
from skimage.segmentation import clear_border
from models import StarDist2D
from skimage.draw import polygon

# Giả sử bạn đã có các hàm phụ trợ này (từ utils hoặc tự implement)
# Nếu chưa có, mình sẽ bổ sung sau
def ray_angles(n_rays=32):
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def dist_to_coord(dist, points, scale_dist=(1, 1)):
    """
    Chuyển radial distances + center points → tọa độ đỉnh polygon.
    Đầu ra sát với official: (n_polys, 2, n_rays)
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    n_rays = dist.shape[-1]
    phis = ray_angles(n_rays)
    
    # coord[..., 0, :] là Y (row), coord[..., 1, :] là X (col)
    # Sát với paper: Y = center_Y + rho * sin(phi), X = center_X + rho * cos(phi)
    coord = (dist[:, np.newaxis] * np.array([np.sin(phis), np.cos(phis)])).astype(np.float32)
    coord *= np.asarray(scale_dist).reshape(1, 2, 1)
    coord += points[..., np.newaxis]
    return coord


def polygons_to_label(dist, points, shape, prob=None, prob_thresh=0.5, scale_dist=(1, 1)):
    """
    Render polygons thành label mask.
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    prob = np.ones(len(points)) if prob is None else np.asarray(prob)

    # Lọc & Sắp xếp theo score tăng dần (để polygon cao hơn đè lên sau)
    # Hoặc sort giảm dần và render id cố định (official dùng sort stable)
    ind = prob > prob_thresh
    points = points[ind]
    dist = dist[ind]
    prob = prob[ind]

    ind_sort = np.argsort(prob, kind='stable')
    points = points[ind_sort]
    dist = dist[ind_sort]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)
    
    label = np.zeros(shape, dtype=np.int32)
    for i, poly in enumerate(coord):
        # poly: (2, n_rays)
        rr, cc = polygon(poly[0], poly[1], shape)
        label[rr, cc] = i + 1

    return label


def non_maximum_suppression(dist, prob, grid=(1, 1), prob_thresh=0.5, nms_thresh=0.5, verbose=False):
    """
    Python-only NMS cho StarDist polygons.
    Logic: P1 loại P2 nếu IoU(P1, P2) > nms_thresh.
    IoU được tính là: Area_Intersection(P1, P2) / min(Area(P1), Area(Area2))
    """
    dist = np.asarray(dist)
    prob = np.asarray(prob)
    assert prob.ndim == 2 and dist.ndim == 3 and prob.shape == dist.shape[:2]

    # Threshold & Get candidates
    mask = prob > prob_thresh
    points = np.stack(np.where(mask), axis=1)
    scores = prob[mask]
    dists = dist[mask]

    # Sort descending (score cao nhất trước)
    ind_sort = np.argsort(scores)[::-1]
    points = points[ind_sort]
    scores = scores[ind_sort]
    dists = dists[ind_sort]

    # Rescale points theo grid
    points = points * np.array(grid).reshape(1, 2)
    
    n_poly = len(points)
    if n_poly == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, dist.shape[-1]))

    # Pre-calculate coords và bboxes
    coords = dist_to_coord(dists, points) # (N, 2, n_rays)
    bboxes = []
    for i in range(n_poly):
        c = coords[i]
        bboxes.append([c[0].min(), c[1].min(), c[0].max(), c[1].max()])
    
    # Pre-calculate areas (dùng công thức cho star-convex polygon)
    n_rays = dists.shape[1]
    dphi = 2 * np.pi / n_rays
    areas = 0.5 * np.sin(dphi) * np.sum(dists * np.roll(dists, -1, axis=1), axis=1)
    
    survivors = np.ones(n_poly, dtype=bool)
    
    for i in range(n_poly):
        if not survivors[i]:
            continue
        
        for j in range(i + 1, n_poly):
            if not survivors[j]:
                continue
            
            # Fast Bbox check
            b1 = bboxes[i]
            b2 = bboxes[j]
            if b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]:
                continue
            
            # Tính IoU (render cục bộ để tối ưu tốc độ trong Python)
            ymin = int(min(b1[0], b2[0]))
            xmin = int(min(b1[1], b2[1]))
            ymax = int(max(b1[2], b2[2])) + 1
            xmax = int(max(b1[3], b2[3])) + 1
            
            h, w = ymax - ymin, xmax - xmin
            if h <= 0 or w <= 0: continue
            
            mask1 = np.zeros((h, w), bool)
            mask2 = np.zeros((h, w), bool)
            
            rr1, cc1 = polygon(coords[i, 0] - ymin, coords[i, 1] - xmin, (h, w))
            rr2, cc2 = polygon(coords[j, 0] - ymin, coords[j, 1] - xmin, (h, w))
            mask1[rr1, cc1] = True
            mask2[rr2, cc2] = True
            
            inter = np.logical_and(mask1, mask2).sum()
            # Official logic: inter / min(a1, a2)
            iou = inter / (min(areas[i], areas[j]) + 1e-10)
            
            if iou > nms_thresh:
                survivors[j] = False

    return points[survivors], scores[survivors], dists[survivors]


def inference(model, image, prob_thresh=0.5, nms_thresh=0.5, device='cuda'):
    """
    Inference full image → instance segmentation.
    """
    model.eval()
    with torch.no_grad():
        if image.ndim == 2:
            image = image[..., None]
        
        # Tiền xử lý đơn giản (nên trùng với dataset.py)
        # Ở đây giả sử ảnh đã được normalized hoặc dùng 1/99 percentile như dataset
        pmin = np.percentile(image, 1)
        pmax = np.percentile(image, 99)
        img_norm = (image - pmin) / (pmax - pmin + 1e-8)
        img_norm = np.clip(img_norm, 0, 1)

        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        prob, dist = model(img_tensor)
        
        prob = prob.cpu().numpy()[0, 0]
        dist = dist.cpu().numpy()[0]

    # NMS
    points, scores, dist_filtered = non_maximum_suppression(
        dist.transpose(1, 2, 0),
        prob,
        grid=model.grid,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh
    )

    # Chuyển thành label mask
    labels = polygons_to_label(dist_filtered, points, shape=image.shape[:2], prob=scores)

    return labels, {'points': points, 'prob': scores, 'dist': dist_filtered}


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StarDist2D(n_channels_in=1).to(device)
    dummy_img = np.zeros((256, 256), dtype=np.float32)
    labels, info = inference(model, dummy_img, device=device)
    print("Instance mask shape:", labels.shape)
    print("Số object detect:", len(np.unique(labels)) - 1)