import os
import torch
import numpy as np
import tifffile
from glob import glob
from tqdm import tqdm
from models import StarDist2D
from postprocess import inference
from stardist.matching import matching_dataset
import argparse

def evaluate_model(root_dir="data/dsb2018/train/", checkpoint="checkpoints/best_model.pth", prob_thresh=0.4, nms_thresh=0.3):
    """
    Chương trình đánh giá model Fourier StarDist bằng chỉ số mAP tiêu chuẩn.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang đánh giá mô hình trên {device}...")

    # 1. Load Model
    model = StarDist2D(n_harmonics=16).to(device)
    if not os.path.exists(checkpoint):
        print(f"Error: Không tìm thấy checkpoint tại {checkpoint}")
        return
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # 2. Load Validation Data
    # Lấy 20% cuối làm val (giống logic random_split nhưng lấy fixed subset cho stable eval)
    image_paths = sorted(glob(os.path.join(root_dir, "images/*.tif")))
    mask_paths = sorted(glob(os.path.join(root_dir, "masks/*.tif")))
    
    val_size = int(0.2 * len(image_paths))
    val_image_paths = image_paths[-val_size:]
    val_mask_paths = mask_paths[-val_size:]
    
    print(f"Số lượng ảnh đánh giá: {len(val_image_paths)}")

    Y_true = []
    Y_pred = []

    # 3. Chạy Inference cho toàn bộ tập Val
    for img_p, mask_p in tqdm(zip(val_image_paths, val_mask_paths), total=len(val_image_paths), desc="Eval Inference"):
        img = tifffile.imread(img_p)
        gt_mask = tifffile.imread(mask_p)
        
        # Chạy dự đoán (Sử dụng Fourier để render mượt)
        pred_mask, _ = inference(model, img, prob_thresh=prob_thresh, nms_thresh=nms_thresh, device=device)
        
        Y_true.append(gt_mask)
        Y_pred.append(pred_mask)

    # 4. Tính toán Metrics (IoU thresholds: 0.5 to 0.9)
    taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    stats = matching_dataset(Y_true, Y_pred, thresh=taus, show_progress=False)

    # 5. In kết quả cuối cùng
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ (FOURIER STARDIST)")
    print("="*50)
    print(f"{'Threshold':<10} | {'F1':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-"*50)
    
    ma_precision = []
    for s in stats:
        print(f"{s.thresh:<10.2f} | {s.f1:<8.4f} | {s.precision:<10.4f} | {s.recall:<8.4f}")
        ma_precision.append(s.precision)
    
    map_score = np.mean([s.f1 for s in stats])
    print("="*50)
    print(f">> MEAN AVERAGE PRECISION (mAP): {map_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Fourier StarDist model.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to checkpoint.')
    parser.add_argument('--prob_thresh', type=str, default=0.4, help='Probability threshold.')
    args = parser.parse_args()
    
    evaluate_model(checkpoint=args.checkpoint, prob_thresh=float(args.prob_thresh))
