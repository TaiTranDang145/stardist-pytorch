import torch
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from models import StarDist2D
from postprocess import inference, fourier_to_coord, dist_to_coord
from stardist import star_dist as stardist_func
from skimage.measure import regionprops

def final_bench(img_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Paths
    img_path = f"data/dsb2018/train/images/{img_id}"
    mask_path = img_path.replace("images", "masks")
    
    # 2. Load Data
    img = tifffile.imread(img_path)
    masks = tifffile.imread(mask_path)
    
    # 3. Model Inference (Epoch 10)
    model = StarDist2D(n_harmonics=16).to(device)
    if os.path.exists("checkpoints/best_model.pth"):
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    
    labels, info = inference(model, img, prob_thresh=0.3, nms_thresh=0.3, device=device)
    
    # 4. Generate Ground Truth Benchmarks
    # Lấy các tia (32 rays) chuẩn của StarDist
    dist_gt = stardist_func(masks, n_rays=32)
    props = regionprops(masks)
    gt_pts = np.array([p.centroid for p in props])
    gt_dst = np.array([dist_gt[int(p.centroid[0]), int(p.centroid[1])] for p in props])
    
    # Chuyển GT sang Fourier
    from fourier_utils import rays_to_fourier
    gt_f_complex = rays_to_fourier(gt_dst, n_harmonics=16)
    gt_f_cat = np.concatenate([gt_f_complex.real, gt_f_complex.imag], axis=-1)
    
    # 5. Plotting (3 Columns)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Col 1: Current Prediction (Red)
    axes[0].imshow(img, cmap='gray')
    p_pts = info['points']
    p_fourier = info['fourier']
    for i in range(len(p_pts)):
        # fourier_to_coord mặc định damping_sigma=8
        poly = fourier_to_coord(p_fourier[i:i+1], p_pts[i:i+1], n_samples=128)[0]
        poly_cl = np.concatenate([poly, poly[:, :1]], axis=1)
        axes[0].plot(poly_cl[1], poly_cl[0], color='red', lw=1.2)
    axes[0].set_title(f"1. Model Pred (10 Epochs)\nVisible result after scaling fix!")
    axes[0].axis('off')
    
    # Col 2: Author's Original (32 rays GT - Yellow)
    axes[1].imshow(img, cmap='gray')
    for i in range(len(gt_pts)):
        poly_ray = dist_to_coord(gt_dst[i:i+1], gt_pts[i:i+1])[0]
        poly_ray_cl = np.concatenate([poly_ray, poly_ray[:, :1]], axis=1)
        axes[1].plot(poly_ray_cl[1], poly_ray_cl[0], color='yellow', lw=1.2)
    axes[1].set_title("2. Original StarDist (Baseline)\nAuthor's 32-ray (Jagged)")
    axes[1].axis('off')
    
    # Col 3: Fourier StarDist (Smooth 128 samples GT - Cyan)
    axes[2].imshow(img, cmap='gray')
    for i in range(len(gt_pts)):
        poly_smooth = fourier_to_coord(gt_f_cat[i:i+1], gt_pts[i:i+1], n_samples=128)[0]
        poly_smooth_cl = np.concatenate([poly_smooth, poly_smooth[:, :1]], axis=1)
        axes[2].plot(poly_smooth_cl[1], poly_smooth_cl[0], color='cyan', lw=1.2)
    axes[2].set_title("3. Fourier StarDist (Target)\nOur Method - Smooth & Natural")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("final_benchmark_3column.png", dpi=150)
    print("Final benchmark saved to final_benchmark_3column.png")

if __name__ == "__main__":
    # Dùng ảnh mẫu đã biết
    img_id = "91cc2e0d4d6e2c1ad59a8d63bcbe3e2ea8bc7f8e642e942a0113450181e73379.tif"
    final_bench(img_id)
