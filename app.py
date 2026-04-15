import gradio as gr
import torch
import numpy as np
import os
from models import StarDist2D
from postprocess import inference
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries

# Cấu hình thiết bị (GPU nếu có, không thì dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình
model = StarDist2D(n_rays=32, grid=(1, 1)).to(device)

# Tải trọng số tốt nhất
model_path = os.path.abspath("best_model.pth")
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Đã tải mô hình thành công từ {model_path}.")
except Exception as e:
    print(f"⚠️ Cảnh báo: Không thể tải mô hình. Vui lòng kiểm tra lại đường dẫn '{model_path}'. Lỗi: {e}")

model.eval()

def segment_cells(input_img):
    if input_img is None:
        return None, None
        
    # Chuyển đổi ảnh sang ảnh xám (grayscale)
    if len(input_img.shape) == 3:
        if input_img.shape[2] == 4:
            input_img = input_img[:, :, :3] # Bỏ kênh alpha
        img_gray = np.mean(input_img, axis=-1)
    else:
        img_gray = input_img.copy()

    # Thực hiện dự đoán
    labels, _ = inference(
        model, img_gray, 
        prob_thresh=0.5, 
        nms_thresh=0.3,
        device=device
    )
    
    # 1. Chuẩn hoá ảnh đầu vào thành uint8 để hiển thị trên trình duyệt (sửa lỗi ko hiện file .tif)
    img_display = input_img.copy()
    if img_display.dtype != np.uint8:
        pmin = np.percentile(img_display, 1)
        pmax = np.percentile(img_display, 99)
        img_display = np.clip((img_display - pmin) / (pmax - pmin + 1e-8), 0, 1)
        img_display = (img_display * 255).astype(np.uint8)
    
    # 2. Tạo ảnh kết quả: copy từ ảnh gốc chuẩn hoá và vẽ viền đỏ trực tiếp vào pixel Numpy
    if len(img_display.shape) == 2:
        img_result = np.stack((img_display,)*3, axis=-1)
    else:
        img_result = img_display.copy()
        
    # Lấy ra các đường viền của label và tô màu đỏ (255, 0, 0)
    boundaries = find_boundaries(labels, mode='thick')
    img_result[boundaries] = [255, 0, 0]
    
    return img_display, img_result

# Xây dựng giao diện Web với Gradio
with gr.Blocks(title="Phân Đoạn Tế Bào bằng StarDist", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Phân Đoạn Tế Bào (Cell Segmentation)")
    gr.Markdown("Vui lòng tải ảnh tế bào lên để mô hình thực hiện phân đoạn.")
    
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(show_label=False)
            submit_btn = gr.Button("🔍 Bắt đầu phân đoạn", variant="primary")
            
        with gr.Column():
            # Sử dụng gr.Image thay vì gr.Plot để 2 bức ảnh có cùng kích thước song song hiển thị
            image_out = gr.Image(show_label=False)
            
    submit_btn.click(fn=segment_cells, inputs=image_in, outputs=[image_in, image_out])

if __name__ == "__main__":
    print("🚀 Đang khởi động giao diện...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
