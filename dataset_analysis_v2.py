import os
import tifffile
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

ARTIFACT_DIR = "/home/lok/.gemini/antigravity/brain/92f1c56c-c2a4-494a-90f7-c3651889695b"

def get_stats(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))
    
    sizes = []
    num_instances = []
    
    for img_p, mask_p in zip(image_paths, mask_paths):
        img = tifffile.imread(img_p)
        mask = tifffile.imread(mask_p)
        
        sizes.append(f"{img.shape[1]}x{img.shape[0]}")
        num_instances.append(len(np.unique(mask)) - 1) # exclude 0 for background
        
    return sizes, num_instances, image_paths, mask_paths

def main():
    train_image_dir = "data/dsb2018/train/images"
    train_mask_dir = "data/dsb2018/train/masks"
    
    print(f"Loading data from {train_image_dir}...")
    train_sizes, train_instances, train_img_paths, train_mask_paths = get_stats(train_image_dir, train_mask_dir)
    
    test_image_dir = "data/dsb2018/test/images"
    
    test_image_paths = sorted(glob(os.path.join(test_image_dir, "*.tif")))
    test_sizes = []
    for img_p in test_image_paths:
        img = tifffile.imread(img_p)
        test_sizes.append(f"{img.shape[1]}x{img.shape[0]}")
        
    print("Generating size distribution plots...")
    # Plot size distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    def plot_dist(ax, pairs, title, color):
        counts = Counter(pairs)
        sorted_pairs = sorted(counts.keys(), key=lambda x: [int(i) for i in x.split('x')])
        sorted_counts = [counts[p] for p in sorted_pairs]
        ax.bar(sorted_pairs, sorted_counts, color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Image Size (WxH)')
        ax.set_ylabel('Number of Images')
        ax.tick_params(axis='x', rotation=45)
    
    plot_dist(axes[0], train_sizes, 'Train Dataset: Image Size Distribution', 'blue')
    plot_dist(axes[1], test_sizes, 'Test Dataset: Image Size Distribution', 'orange')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "size_distribution.png"))
    plt.close()
    
    print("Generating instance distribution plots...")
    # Plot number of instances distribution (for train)
    plt.figure(figsize=(10, 6))
    plt.hist(train_instances, bins=50, color='green', alpha=0.7)
    plt.title('Train Dataset: Distribution of Number of Instances (Cells) per Image')
    plt.xlabel('Number of Instances')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "instances_distribution.png"))
    plt.close()
    
    print("Generating sample visualisations...")
    # Visualize samples
    indices = random.sample(range(len(train_img_paths)), min(5, len(train_img_paths)))
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4*len(indices)))
    
    for i, idx in enumerate(indices):
        img_path = train_img_paths[idx]
        mask_path = train_mask_paths[idx]
        
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # Color objects differently, overlay on original image
        mask_colored = np.ma.masked_where(mask == 0, mask)
        axes[i, 1].imshow(img, cmap='gray')
        axes[i, 1].imshow(mask_colored, cmap='nipy_spectral', alpha=0.5)
        axes[i, 1].set_title("Instance Mask Overlay")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "dataset_samples.png"), bbox_inches='tight')
    plt.close()
    
    print("Writing markdown report...")
    # Save a markdown report
    report = f"""# Dataset Statistics
    
## Training Set
- Total Images: {{len(train_sizes)}}
- Average instances (cells) per image: {{np.mean(train_instances):.2f}}
- Min instances: {{np.min(train_instances)}}
- Max instances: {{np.max(train_instances)}}
- Standard Deviation: {{np.std(train_instances):.2f}}

## Test Set
- Total Images: {{len(test_sizes)}}

## Visualizations
### Image Size Distribution
![Size Distribution]({{ARTIFACT_DIR}}/size_distribution.png)

### Distribution of Number of Cells
![Instances Distribution]({{ARTIFACT_DIR}}/instances_distribution.png)

### Examples of Original Images and Masks
![Dataset Samples]({{ARTIFACT_DIR}}/dataset_samples.png)
"""
    with open(os.path.join(ARTIFACT_DIR, "dataset_report.md"), "w") as f:
        f.write(report)
        
    print("Dataset analysis complete! Outputs saved to the artifact directory.")

if __name__ == "__main__":
    main()
