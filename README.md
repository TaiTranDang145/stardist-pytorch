# Fourier StarDist: Dynamic Rays for Smooth Cell Segmentation

Dự án này triển khai kiến trúc **Fourier StarDist**, thay đổi cách biểu diễn hình dạng tế bào từ các tia (rays) truyền thống sang các hệ số Fourier (Fourier Coefficients). Điều này mang lại độ mượt mà tối ưu cho biên tế bào và khả năng thích ứng linh hoạt với độ phức tạp của hình dạng.

## ✨ Tính năng nổi bật
- **Fourier StarDist**: Sử dụng hệ số Fourier thay vì 32 tia cố định, cho phép tái tạo biên mượt mà với độ phân giải bất kỳ (mặc định 128 điểm).
- **Dynamic Rays**: Tự động điều chỉnh độ chi tiết của biên ở giai đoạn hậu xử lý mà không cần thay đổi model.
- **Complexity-based Gating**: Nhánh dự đoán độ phức tạp (Complexity Head) giúp mô hình tối ưu hóa việc phân hóa giữa các tế bào tròn và tế bào có hình dạng phức tạp.
- **Smooth Reconstruction**: Tích hợp bộ lọc Gaussian Low-pass trong `fourier_utils.py` để loại bỏ hiện tượng răng cưa/rung (ringing artifacts).

## 🚀 Hướng dẫn nhanh

### 1. Cài đặt môi trường
Yêu cầu Python 3.10+. Cài đặt các phụ thuộc:
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện (Training)
Để bắt đầu huấn luyện dài hạn trên GPU:
1. Kiểm tra dữ liệu trong `data/dsb2018/`.
2. Chạy lệnh:
```bash
python train.py
```
*Lưu ý: File `train.py` đã được cấu hình mặc định 100 epochs với Batch Size 16 và các tham số tối ưu.*

### 3. Chạy Demo & Kiểm tra độ mượt
Để chạy thử inference và xem kết quả so sánh:
```bash
python demo_inference.py
```
Kết quả trực quan sẽ được lưu tại `fourier_stardist_demo.png`.

## 📂 Cấu trúc mã nguồn
- `models.py`: Kiến trúc U-Net cải tiến với Fourier & Complexity heads.
- `dataset.py`: DataLoader hỗ trợ tính toán Fourier & Complexity Ground Truth.
- `fourier_utils.py`: Thư viện xử lý FFT và làm mượt biên (Damping).
- `postprocess.py`: Hậu xử lý NMS và render Polygon hiệu năng cao.
- `loss.py`: Hệ thống hàm mất mát đa nhiệm (Prob, Dist, Fourier, Complexity).

---
*Dự án được phát triển nhằm nâng cao độ chính xác hình thái trong phân đoạn ảnh y sinh.*
