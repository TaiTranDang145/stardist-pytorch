import torch
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from models import StarDist2D
from postprocess import inference, fourier_to_coord
from dataset import create_dataloaders

def run_demo():
    """
    Script demo chính thức để kiểm tra kết quả phân đoạn Fourier StarDist.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = StarDist2D(n_harmonics=16).to(device)
    checkpoint = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded weights from {checkpoint}")
    else:
        print("Warning: No checkpoint found. Running with random weights.")

    # 2. Load Sample Image from Validation Set
    _, val_loader = create_dataloaders(batch_size=1)
    # Lấy ảnh có tế bào
    for images, prob_gt, dist_gt, fgt in val_loader:
        if prob_gt.max() > 0.5: break
    
    img_tensor = images[0]
    img = img_tensor.permute(1, 2, 0).numpy()
    
    # 3. Inference
    # Sử dụng use_fourier=True để render biên mượt
    labels, info = inference(model, img, prob_thresh=0.3, nms_thresh=0.3, device=device)
    
    # 4. Visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Col 1: Original Image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')
    
    # Col 2: Predicted Instance Labels (Colorized)
    from skimage.color import label2rgb
    axes[1].imshow(label2rgb(labels, bg_label=0))
    axes[1].set_title(f"2. Instance Labels\n({len(np.unique(labels))-1} objects)")
    axes[1].axis('off')
    
    # Col 3: Fourier Boundaries Overlay
    axes[2].imshow(img, cmap='gray')
    pts = info['points']
    f_coeffs = info['fourier']
    for i in range(len(pts)):
        # fourier_to_coord mặc định damping_sigma=8 để cực mượt
        poly = fourier_to_coord(f_coeffs[i:i+1], pts[i:i+1], n_samples=128)[0]
        poly_closed = np.concatenate([poly, poly[:, :1]], axis=1)
        axes[2].plot(poly_closed[1], poly_closed[0], color='red', lw=1.2)
    axes[2].set_title("3. Fourier Boundaries\n(Smooth Reconstruction)")
    axes[2].axis('off')
    
    # Col 4: Zoom for detail
    if len(pts) > 0:
        idx = 0 # Lấy tế bào đầu tiên để zoom
        y, x = pts[idx]
        axes[3].imshow(img, cmap='gray')
        for i in range(len(pts)):
            poly = fourier_to_coord(f_coeffs[i:i+1], pts[i:i+1], n_samples=128)[0]
            axes[3].plot(poly[1], poly[0], color='cyan', lw=1.5)
        
        # Zoom vào vùng quanh tâm tế bào đó
        axes[3].set_xlim(x-30, x+30)
        axes[3].set_ylim(y+30, y-30)
        axes[3].set_title("4. Zoomed View (Smoothness Check)")
    else:
        axes[3].text(0.5, 0.5, "No detections", ha='center')
    
    plt.tight_layout()
    plot_path = "fourier_stardist_demo.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Demo result saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_demo()
