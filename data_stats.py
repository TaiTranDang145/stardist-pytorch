import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

from collections import Counter

def get_image_sizes(image_dir):
    pairs = []
    image_paths = glob(os.path.join(image_dir, "*.tif"))
    
    for path in image_paths:
        with Image.open(path) as img:
            w, h = img.size
            pairs.append(f"{w}x{h}")
    return pairs

def plot_distribution(ax, pairs, title, color):
    if not pairs:
        ax.set_title(f"{title} (No Data)")
        return

    counts = Counter(pairs)
    # Sort by size (W then H) for consistent X axis
    sorted_pairs = sorted(counts.keys(), key=lambda x: [int(i) for i in x.split('x')])
    sorted_counts = [counts[p] for p in sorted_pairs]

    ax.bar(sorted_pairs, sorted_counts, color=color, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Image Size (WxH)')
    ax.set_ylabel('Number of Images (Count)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def main():
    train_dir = "data/dsb2018/train/images"
    test_dir = "data/dsb2018/test/images"
    
    print(f"Analyzing {train_dir}...")
    train_pairs = get_image_sizes(train_dir)
    print(f"Analyzing {test_dir}...")
    test_pairs = get_image_sizes(test_dir)
    
    print("\nTraining Statistics (Unique Sizes):")
    train_counts = Counter(train_pairs)
    for size, count in sorted(train_counts.items()):
        print(f"  {size}: {count}")
    
    print("\nTesting Statistics (Unique Sizes):")
    test_counts = Counter(test_pairs)
    for size, count in sorted(test_counts.items()):
        print(f"  {size}: {count}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_distribution(axes[0], train_pairs, 'Train Dataset: Image Size Distribution', 'blue')
    plot_distribution(axes[1], test_pairs, 'Test Dataset: Image Size Distribution', 'orange')
    
    plt.tight_layout()
    output_filename = "data_size_distribution_v2.png"
    plt.savefig(output_filename)
    print(f"\nHistogram saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
