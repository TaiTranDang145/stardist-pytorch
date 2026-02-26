import os
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from collections import namedtuple

# ====================== Các hàm phụ trợ từ code gốc ======================

def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError(f"{name or 'labels'} must be an array of non-negative integers.")
    if not (isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)):
        raise err
    if y.size == 0:
        return True
    if y.min() < 0:
        raise err
    if check_sequential:
        labels = np.unique(y)
        if not (set(labels) - {0} == set(range(1, 1 + labels.max()))):
            raise err
    return True


def label_overlap(x, y):
    _check_label_array(x, 'x', True)
    _check_label_array(y, 'y', True)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _safe_divide(x, y, eps=1e-10):
    out = np.zeros_like(x, dtype=np.float32)
    np.divide(x, y, out=out, where=np.abs(y) > eps)
    return out


def intersection_over_union(overlap):
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, n_pixels_pred + n_pixels_true - overlap)


def matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """
    Tính metrics giữa ground truth và prediction labels (sát gốc)
    """
    _check_label_array(y_true, 'y_true', True)
    _check_label_array(y_pred, 'y_pred', True)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)

    overlap = label_overlap(y_true, y_pred)
    scores = intersection_over_union(overlap)
    scores = scores[1:, 1:]  # bỏ background

    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    not_trivial = n_matched > 0 and np.any(scores >= thresh)
    if not_trivial:
        costs = -(scores >= thresh).astype(float) - scores / (2 * n_matched)
        true_ind, pred_ind = linear_sum_assignment(costs)
        match_ok = scores[true_ind, pred_ind] >= thresh
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0

    fp = n_pred - tp
    fn = n_true - tp

    sum_matched_score = np.sum(scores[true_ind, pred_ind][match_ok]) if not_trivial else 0.0
    mean_matched_score = sum_matched_score / tp if tp > 0 else 0.0
    mean_true_score = sum_matched_score / n_true if n_true > 0 else 0.0
    panoptic_quality = sum_matched_score / (tp + fp/2 + fn/2) if (tp + fp/2 + fn/2) > 0 else 0.0

    stats = {
        'criterion': criterion,
        'thresh': thresh,
        'fp': fp,
        'tp': tp,
        'fn': fn,
        'precision': tp / (tp + fp) if tp > 0 else 0.0,
        'recall': tp / (tp + fn) if tp > 0 else 0.0,
        'accuracy': tp / (tp + fp + fn) if tp > 0 else 0.0,
        'f1': (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0.0,
        'n_true': n_true,
        'n_pred': n_pred,
        'mean_true_score': mean_true_score,
        'mean_matched_score': mean_matched_score,
        'panoptic_quality': panoptic_quality,
    }

    if report_matches and not_trivial:
        stats.update({
            'matched_pairs': tuple((int(i), int(j)) for i, j in zip(true_ind + 1, pred_ind + 1) if match_ok[i]),
            'matched_scores': tuple(scores[true_ind, pred_ind][match_ok]),
        })
    else:
        stats.update({'matched_pairs': (), 'matched_scores': ()})

    return namedtuple('Matching', stats.keys())(*stats.values())


def evaluate_dataset(y_true_list, y_pred_list, thresh=0.5, criterion='iou', show_progress=True):
    """
    Đánh giá toàn bộ dataset (list of (y_true, y_pred))
    Trả về average metrics (PQ, precision, recall, f1, ...)
    """
    stats_list = []
    for y_true, y_pred in tqdm(zip(y_true_list, y_pred_list), total=len(y_true_list), desc="Evaluating"):
        stats = matching(y_true, y_pred, thresh=thresh, criterion=criterion)
        stats_list.append(stats)

    # Tính trung bình
    n_images = len(stats_list)
    avg_stats = {}
    for key in ['fp', 'tp', 'fn', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality']:
        avg_stats[key] = np.mean([s._asdict()[key] for s in stats_list])

    avg_stats['precision'] = avg_stats['tp'] / (avg_stats['tp'] + avg_stats['fp']) if (avg_stats['tp'] + avg_stats['fp']) > 0 else 0.0
    avg_stats['recall'] = avg_stats['tp'] / (avg_stats['tp'] + avg_stats['fn']) if (avg_stats['tp'] + avg_stats['fn']) > 0 else 0.0
    avg_stats['f1'] = (2 * avg_stats['tp']) / (2 * avg_stats['tp'] + avg_stats['fp'] + avg_stats['fn']) if (2 * avg_stats['tp'] + avg_stats['fp'] + avg_stats['fn']) > 0 else 0.0
    avg_stats['criterion'] = criterion
    avg_stats['thresh'] = thresh

    return namedtuple('DatasetEvaluation', avg_stats.keys())(*avg_stats.values())


# ====================== Test đánh giá đơn giản ======================
if __name__ == "__main__":
    # Giả lập ground truth và prediction (labels instance)
    y_true = np.zeros((256, 256), dtype=np.int32)
    y_true[50:150, 50:150] = 1  # object 1

    y_pred = np.zeros((256, 256), dtype=np.int32)
    y_pred[60:160, 60:160] = 1  # predict gần giống

    stats = matching(y_true, y_pred, thresh=0.5, criterion='iou')
    print("Single image metrics:")
    print(stats)

    # Test dataset
    y_true_list = [y_true, y_true]
    y_pred_list = [y_pred, y_pred]
    dataset_stats = evaluate_dataset(y_true_list, y_pred_list)
    print("\nDataset average:")
    print(dataset_stats)