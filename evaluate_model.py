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
    data_dir = "data/dsb2018/test/" 
    n_rays = 32
    grid = (1, 1)
    prob_thresh = 0.5
    nms_thresh = 0.3
    taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
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
    
    if len(image_paths) == 0:
        print(f"Lỗi: Không tìm thấy ảnh trong {config.data_dir}")
        return

    y_true_list = []
    y_pred_list = []

    print(f"Đang chạy inference trên toàn bộ {len(image_paths)} ảnh của bộ TEST...")
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

    # 3. Chạy Evaluate trên nhiều ngưỡng
    print("\n" + "="*80)
    print(f"{'Threshold':<10} | {'PQ':<10} | {'Accuracy':<10} | {'F1':<10} | {'Prec':<10} | {'Recall':<10}")
    print("-" * 80)
    
    for t in config.taus:
        results = evaluate_dataset(y_true_list, y_pred_list, thresh=t, show_progress=False)
        print(f"{t:<10.2f} | {results.panoptic_quality:<10.4f} | {results.accuracy:<10.4f} | {results.f1:<10.4f} | {results.precision:<10.4f} | {results.recall:<10.4f}")
    
    print("="*80)

if __name__ == "__main__":
    evaluate()
