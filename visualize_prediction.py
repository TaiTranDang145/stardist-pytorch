import os
import glob
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from models import StarDist2D
from postprocess import inference
from skimage.measure import find_contours

class VisConfig:
    checkpoint_path = "checkpoints/best_model.pth"
    data_dir = "data/dsb2018/test/"
    n_rays = 32
    grid = (1, 1)
    prob_thresh = 0.5
    nms_thresh = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 10
    save_path = "prediction_visualization.png"

def visualize():
    config = VisConfig()
    
    # 1. Load Model
    model = StarDist2D(n_rays=config.n_rays, grid=config.grid).to(config.device)
    if not os.path.exists(config.checkpoint_path):
        print(f"Error: No checkpoint found at {config.checkpoint_path}")
        return
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    model.eval()
    print(f"Loaded model from {config.checkpoint_path}")

    # 2. Get Images
    image_paths = sorted(glob.glob(os.path.join(config.data_dir, "images/*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(config.data_dir, "masks/*.tif")))
    
    indices = np.linspace(0, len(image_paths) - 1, config.num_samples, dtype=int)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating output folder: {output_dir}")

    for idx in indices:
        img_path = image_paths[idx]
        mask_path = mask_paths[idx]
        
        img = tifffile.imread(img_path)
        gt_mask = tifffile.imread(mask_path)
        
        # Inference
        labels, _ = inference(
            model, img, 
            prob_thresh=config.prob_thresh, 
            nms_thresh=config.nms_thresh,
            device=config.device
        )
        
        # Plotting individual image
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image {idx}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='nipy_spectral')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(img, cmap='gray')
        # Draw outlines
        for l in np.unique(labels):
            if l == 0: continue
            mask = (labels == l)
            contours = find_contours(mask, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)
        
        plt.title("Prediction (Outlines)")
        plt.axis('off')

        save_name = os.path.join(output_dir, f"prediction_{idx}.png")
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()
        print(f"Saved: {save_name}")

    print(f"\nAll {config.num_samples} predictions saved to '{output_dir}/' folder.")

if __name__ == "__main__":
    visualize()
