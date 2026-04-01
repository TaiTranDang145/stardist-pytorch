import numpy as np
import torch
import matplotlib.pyplot as plt

def rays_to_fourier(rays, n_harmonics=10):
    """
    Chuyển đổi các tia (radial distances) sang hệ số Fourier.
    Args:
        rays: mảng numpy (..., n_rays)
        n_harmonics: số lượng harmonics bậc cao muốn giữ lại.
    Returns:
        coeffs: mảng phức (..., n_harmonics + 1)
    """
    n_rays = rays.shape[-1]
    # Thực hiện Real FFT vì rays là số thực
    coeffs = np.fft.rfft(rays, axis=-1)
    
    # Chỉ giữ lại số lượng harmonics mong muốn
    if n_harmonics is not None:
        return coeffs[..., :n_harmonics + 1]
    return coeffs

def fourier_to_rays(coeffs, n_rays=32):
    """
    Chuyển đổi ngược từ hệ số Fourier sang các tia.
    Args:
        coeffs: mảng phức (..., n_harmonics + 1)
        n_rays: số lượng tia muốn tái tạo (độ mịn)
    Returns:
        rays: mảng thực (..., n_rays)
    """
    # Inverse Real FFT
    # n=n_rays để đảm bảo đầu ra có đúng số lượng tia mong muốn (nội suy)
    rays = np.fft.irfft(coeffs, n=n_rays, axis=-1)
    return rays

def visualize_fourier_reconstruction(rays, harmonics_list=[2, 4, 8, 16]):
    """
    Trực quan hóa việc tái tạo hình dạng với số lượng harmonics khác nhau.
    """
    plt.figure(figsize=(15, 5))
    n_rays = len(rays)
    theta = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    
    # Hình gốc
    plt.subplot(1, len(harmonics_list) + 1, 1, projection='polar')
    plt.plot(theta, rays, 'r-', label='Original (32 rays)')
    plt.fill(theta, rays, alpha=0.2)
    plt.title("Original StarDist")
    
    for i, n in enumerate(harmonics_list):
        coeffs = rays_to_fourier(rays, n_harmonics=n)
        # Tái tạo với 128 tia để thấy độ mượt
        reconstructed = fourier_to_rays(coeffs, n_rays=128)
        theta_new = np.linspace(0, 2*np.pi, 128, endpoint=False)
        
        plt.subplot(1, len(harmonics_list) + 1, i + 2, projection='polar')
        plt.plot(theta_new, reconstructed, 'b-')
        plt.fill(theta_new, reconstructed, alpha=0.2)
        plt.title(f"{n} Harmonics")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Demo: Tạo một hình dạng "méo" giả lập 32 tia
    np.random.seed(42)
    base_rays = 10 + 3 * np.sin(np.linspace(0, 2*np.pi, 32, endpoint=False))
    noise = np.random.normal(0, 0.5, 32)
    rays = base_rays + noise
    
    print(f"Original rays: {rays.shape}")
    coeffs = rays_to_fourier(rays, n_harmonics=8)
    print(f"Fourier coefficients (8 harmonics): {coeffs.shape}")
    
    # Thử trực quan hóa (nếu có môi trường hiển thị)
    # visualize_fourier_reconstruction(rays)
