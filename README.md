# Fourier StarDist: Dynamic Rays for Cell Segmentation

Dự án này là phiên bản cải tiến của **StarDist**, tích hợp cơ chế **Fourier-based Shape Modeling** (với Dynamic Rays) để đạt được độ mượt mà tối ưu cho biên tế bào và khả năng thích ứng linh hoạt với độ phức tạp của hình dạng.

## 🚀 Tính năng nổi bật

- **Fourier StarDist**: Thay vì dự đoán 32 tia rời rạc, mô hình dự đoán các hệ số Fourier Complex. Điều này cho phép tái tạo biên tế bào mượt mà với độ phân giải bất kỳ.
- **Dynamic Rays**: Số lượng mẫu điểm trên biên có thể điều chỉnh linh hoạt ở giai đoạn inference mà không cần train lại model.
- **Complexity-based Gating**: Nhánh dự đoán độ phức tạp (Complexity Head) giúp mô hình nhận diện các tế bào có hình dạng bất thường, hỗ trợ hậu xử lý thông minh hơn.
- **Adaptive Loss**: Hệ thống loss đa thành phần kết hợp Prob, Distance, Fourier Coefficients và Complexity Score.

## 📦 Cài đặt

Để bắt đầu, hãy tạo môi trường ảo và cài đặt tất cả các dependencies thông qua file `requirements.txt`:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Linux)
source .venv/bin/activate

# Cài đặt toàn bộ dependencies
pip install -r requirements.txt
```

## 🛠 Cấu trúc Project

- `models.py`: Kiến trúc StarDist2D cải tiến với Fourier & Complexity heads.
- `dataset.py`: DataLoader cho DSB2018 hỗ trợ tính toán Fourier & Complexity Ground Truth.
- `fourier_utils.py`: Thư viện lõi xử lý biến đổi Fourier cho đa giác.
- `loss.py`: Định nghĩa các hàm loss cho kiến trúc Fourier.
- `postprocess.py`: Hậu xử lý NMS và render polygon từ hệ số Fourier.
- `train.py`: Quy trình huấn luyện với logging TensorBoard chi tiết.
- `demo_fourier.py`: Script minh họa khả năng tái tạo hình dạng mượt mà của Fourier.

## 🏋️‍♂️ Huấn luyện

Đảm bảo dữ liệu DSB2018 nằm trong `data/dsb2018/train/`. Sau đó chạy:
```bash
python train.py
```

Theo dõi quá trình huấn luyện bằng TensorBoard:
```bash
tensorboard --logdir=tensorboard_logs
```

## 🔍 Inference & Visualization

Để chạy thử inference trên ảnh mới và xem kết quả dựng biên từ Fourier:
```bash
python postprocess.py
```

Sử dụng `visualize_prediction.py` để xem các heatmap xác suất và khoảng cách.

## 📊 Kết quả Demo

| Gốc (32 Rays) | Fourier Reconstruction (16 Harmonics) |
| :---: | :---: |
| Răng cưa, đa giác thô | Đường cong mượt mà, tự nhiên |

*Xem chi tiết tại [fourier_recon_demo.png](fourier_recon_demo.png)*
