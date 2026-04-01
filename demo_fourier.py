import numpy as np
import matplotlib.pyplot as plt
from fourier_utils import rays_to_fourier, fourier_to_rays

def main():
    # 1. Tạo 32 tia giả lập (có chút nhiễu cho thực tế)
    n_rays = 32
    theta = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    
    # Tế bào hình hơi elip + một chút móp méo
    rays = 15 + 5 * np.cos(1 * theta) + 2 * np.sin(2 * theta)
    # Thêm nhiễu răng cưa (noise)
    np.random.seed(0)
    rays += np.random.normal(0, 0.8, n_rays)
    
    # 2. Reconstruct với các số lượng Harmonics khác nhau
    harmonics_to_test = [2, 4, 8, 16]
    
    fig, axes = plt.subplots(1, len(harmonics_to_test) + 1, figsize=(20, 5), subplot_kw={'projection': 'polar'})
    
    # Vẽ bản gốc (Đa giác 32 cạnh)
    axes[0].plot(theta, rays, 'r.-', label='Original (32 points)')
    axes[0].fill(theta, rays, alpha=0.1, color='red')
    axes[0].set_title("Original StarDist\n(32 Rays - Jagged)")
    
    for i, n in enumerate(harmonics_to_test):
        coeffs = rays_to_fourier(rays, n_harmonics=n)
        
        # Tái tạo lại với 200 tia (cho thật mịn)
        n_dense = 200
        theta_dense = np.linspace(0, 2*np.pi, n_dense, endpoint=False)
        reconstructed = fourier_to_rays(coeffs, n_rays=n_dense)
        
        axes[i+1].plot(theta_dense, reconstructed, 'b-')
        axes[i+1].fill(theta_dense, reconstructed, alpha=0.1, color='blue')
        axes[i+1].set_title(f"Fourier: {n} Harmonics\n(Smooth)")
        
    plt.tight_layout()
    plt.savefig("fourier_recon_demo.png")
    print("Đã lưu hình ảnh demo tại fourier_recon_demo.png")

if __name__ == "__main__":
    main()
