import os
import glob
import torch
import numpy as np
import tifffile
from tqdm import tqdm
from models import StarDist2D
from postprocess import inference
from evaluate import evaluate_dataset

class EvalConfig:
    checkpoint_path = "checkpoints/best_model.pth"
    data_dir = "data/dsb2018/train/" # Dùng val split từ đây hoặc test set nếu có
    n_rays = 32
    grid = (1, 1)
    prob_thresh = 0.5
    nms_thresh = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    config = EvalConfig()
    
    # 1. Load Model
    model = StarDist2D(n_rays=config.n_rays, grid=config.grid).to(config.device)
    if not os.path.exists(config.checkpoint_path):
        print(f"Lỗi: Không tìm thấy checkpoint tại {config.checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    model.eval()
    print(f"Đã load model từ {config.checkpoint_path}")

    # 2. Load Data (Lấy danh sách ảnh và mask)
    # Lưu ý: Ở đây chúng ta lấy toàn bộ folder train để demo, 
    # trong thực tế bạn nên tách riêng folder test.
    image_paths = sorted(glob.glob(os.path.join(config.data_dir, "images/*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(config.data_dir, "masks/*.tif")))
    
    # Giới hạn số lượng ảnh đánh giá nếu muốn nhanh (ví dụ 20 ảnh đầu)
    image_paths = image_paths[:20]
    mask_paths = mask_paths[:20]

    y_true_list = []
    y_pred_list = []

    print(f"Đang chạy inference trên {len(image_paths)} ảnh...")
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        img = tifffile.imread(img_path)
        gt_mask = tifffile.imread(mask_path)
        
        # Inference
        pred_labels, _ = inference(
            model, img, 
            prob_thresh=config.prob_thresh, 
            nms_thresh=config.nms_thresh,
            device=config.device
        )
        
        y_true_list.append(gt_mask)
        y_pred_list.append(pred_labels)

    # 3. Chạy Evaluate
    print("\nĐang tính toán metrics (AP, F1, PQ)...")
    results = evaluate_dataset(y_true_list, y_pred_list, thresh=0.5)
    
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ (IoU threshold = 0.5)")
    print("="*30)
    print(f"Precision:         {results.precision:.4f}")
    print(f"Recall:            {results.recall:.4f}")
    print(f"F1 Score:          {results.f1:.4f}")
    print(f"Accuracy:          {results.accuracy:.4f}")
    print(f"Panoptic Quality:  {results.panoptic_quality:.4f}")
    print(f"Mean Matching IoU: {results.mean_matched_score:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
