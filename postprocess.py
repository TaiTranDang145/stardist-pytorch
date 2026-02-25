import numpy as np
import torch
from skimage.measure import regionprops
from scipy.ndimage import zoom
from skimage.segmentation import clear_border
from models import StarDist2D
from skimage.draw import polygon

# Giả sử bạn đã có các hàm phụ trợ này (từ utils hoặc tự implement)
# Nếu chưa có, mình sẽ bổ sung sau
def dist_to_coord(dist, points, scale_dist=(1,1)):
    """
    Chuyển radial distances + center points → tọa độ đỉnh polygon
    dist: (N, n_rays)
    points: (N, 2)
    scale_dist: tuple (sy, sx) để rescale nếu cần
    """
    n_polys, n_rays = dist.shape
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dy = np.cos(angles) * scale_dist[0]
    dx = np.sin(angles) * scale_dist[1]

    coord = np.zeros((n_polys, n_rays, 2), dtype=np.float32)
    for i in range(n_polys):
        for k in range(n_rays):
            coord[i, k, 0] = points[i, 0] + dist[i, k] * dy[k]
            coord[i, k, 1] = points[i, 1] + dist[i, k] * dx[k]
    return coord


def polygons_to_label(coord, points, prob=None, shape=None, scale_dist=(1,1)):
    """
    Render polygons thành label mask (mỗi polygon một id)
    coord: (N, n_rays, 2)
    points: (N, 2) - centers
    prob: (N,) - để sort nếu cần
    shape: (H, W) - kích thước ảnh
    """
    if shape is None:
        raise ValueError("Cần cung cấp shape=(H, W) của ảnh")

    if prob is not None:
        ind = np.argsort(prob)[::-1]
        coord = coord[ind]
        points = points[ind]
    else:
        ind = np.arange(len(coord))

    label = np.zeros(shape, dtype=np.int32)
    for i, poly in enumerate(coord):
        # poly: (n_rays, 2) - các đỉnh polygon
        rr, cc = polygon(poly[:, 0], poly[:, 1], shape)
        label[rr, cc] = i + 1  # id bắt đầu từ 1

    return label


def non_maximum_suppression(dist, prob, grid=(1,1), prob_thresh=0.5, nms_thresh=0.5,
                            use_bbox=True, use_kdtree=True, verbose=False):
    """
    Non-Maximum Suppression cho dense prediction (grid-based)
    dist: (H, W, n_rays)
    prob: (H, W)
    """
    from skimage.measure import label as sk_label  # dùng để hỗ trợ nếu cần

    dist = np.asarray(dist)
    prob = np.asarray(prob)
    assert prob.ndim == 2 and dist.ndim == 3 and prob.shape == dist.shape[:2]

    grid = np.array(grid)
    n_rays = dist.shape[-1]

    # Threshold prob
    mask = prob > prob_thresh
    points = np.stack(np.where(mask), axis=1)  # (N, 2)
    scores = prob[mask]
    dists = dist[mask]

    # Sort descending theo prob
    ind_sort = np.argsort(scores)[::-1]
    points = points[ind_sort]
    scores = scores[ind_sort]
    dists = dists[ind_sort]

    # Rescale points theo grid
    points = points * grid.reshape(1, 2)

    # NMS (dùng implementation NumPy đơn giản, sát với c_non_max_suppression_inds)
    survivors = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not survivors[i]:
            continue
        for j in range(i+1, len(points)):
            if not survivors[j]:
                continue
            # Tính IoU đơn giản giữa 2 polygon (approximate bằng bbox hoặc full IoU)
            # Ở đây dùng approximate IoU bằng bbox để nhanh (gốc có use_bbox)
            p1 = points[i]
            p2 = points[j]
            d1 = dists[i]
            d2 = dists[j]

            # Bbox approximate
            r1 = np.max(d1) * 1.2  # radius max + margin
            r2 = np.max(d2) * 1.2
            bbox1 = [p1[0]-r1, p1[1]-r1, p1[0]+r1, p1[1]+r1]
            bbox2 = [p2[0]-r2, p2[1]-r2, p2[0]+r2, p2[1]+r2]

            inter_x1 = max(bbox1[0], bbox2[0])
            inter_y1 = max(bbox1[1], bbox2[1])
            inter_x2 = min(bbox1[2], bbox2[2])
            inter_y2 = min(bbox1[3], bbox2[3])

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

            iou = inter_area / min(area1, area2) if min(area1, area2) > 0 else 0

            if iou > nms_thresh:
                survivors[j] = False

    # Trả về points, prob, dist đã lọc
    return points[survivors], scores[survivors], dists[survivors]


def inference(model, image, prob_thresh=0.5, nms_thresh=0.5, device='cuda'):
    """
    Inference full image → instance segmentation
    image: np.array (H, W) hoặc (H, W, C) grayscale
    model: StarDist2D đã train
    """
    model.eval()
    with torch.no_grad():
        if image.ndim == 2:
            image = image[..., None]
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        prob, dist = model(img_tensor)

        prob = prob.squeeze(0).cpu().numpy()  # (1, H, W) → (H, W)
        dist = dist.squeeze(0).cpu().numpy()  # (n_rays, H, W) → (H, W, n_rays)

    # NMS
    points, scores, dist_filtered = non_maximum_suppression(
        dist.transpose(1, 2, 0),  # (H, W, n_rays)
        prob,
        grid=(1,1),
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh
    )

    # Chuyển thành label mask
    coord = dist_to_coord(dist_filtered, points)
    labels = polygons_to_label(coord, points, prob=scores, shape=image.shape[:2])

    return labels, {'points': points, 'prob': scores, 'dist': dist_filtered}


# ====================== Test inference đơn giản ======================
if __name__ == "__main__":
    # Giả lập image và model
    model = StarDist2D()
    dummy_img = np.random.rand(256, 256).astype(np.float32)
    labels, info = inference(model, dummy_img)
    print("Instance mask shape:", labels.shape)
    print("Số object detect:", len(np.unique(labels)) - 1)