import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def find_best_run(step2_output_dir):
    """
    Tìm Run có loss cuối cùng thấp nhất trong thư mục đầu ra của Step 2.
    """
    run_dirs = glob.glob(os.path.join(step2_output_dir, "Run*"))
    if not run_dirs:
        return None
        
    best_run = None
    min_loss = float('inf')
    
    for run_dir in run_dirs:
        loss_file = os.path.join(run_dir, "Epoch_UnsupervisedLoss.csv")
        if os.path.exists(loss_file):
            try:
                # Đọc dòng cuối cùng để lấy loss cuối của quá trình train
                df = pd.read_csv(loss_file)
                if not df.empty:
                    final_loss = df.iloc[-1]["UnsupervisedLoss"]
                    if final_loss < min_loss:
                        min_loss = final_loss
                        best_run = run_dir
            except Exception as e:
                print(f"Cảnh báo: Không thể đọc file loss tại {run_dir}: {e}")
                
    return best_run

def visualize_communities(input_dir, output_tcn_dir, output_viz_dir):
    """
    Đọc kết quả TCN và vẽ bản đồ không gian các cộng đồng tế bào.
    """
    input_dir = Path(input_dir)
    output_tcn_dir = Path(output_tcn_dir)
    output_viz_dir = Path(output_viz_dir)
    output_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả các thư mục đầu ra của Step 2
    step2_dirs = glob.glob(os.path.join(output_tcn_dir, "Step2_Output_*"))
    
    if not step2_dirs:
        print(f"Không tìm thấy thư mục Step2_Output_* nào trong {output_tcn_dir}")
        return

    print(f"Bắt đầu visualize cộng đồng cho {len(step2_dirs)} ảnh...")
    
    success_count = 0
    
    for step2_dir in step2_dirs:
        image_name = os.path.basename(step2_dir).replace("Step2_Output_", "")
        print(f"\nĐang xử lý ảnh: {image_name}")
        
        # 1. Tìm Run tốt nhất
        best_run = find_best_run(step2_dir)
        if not best_run:
            print(f"  -> Cảnh báo: Không tìm thấy Run hợp lệ cho {image_name}. Bỏ qua.")
            continue
            
        print(f"  -> Chọn {os.path.basename(best_run)} (Loss thấp nhất)")
        
        # 2. Đọc Assign Matrix (Soft Membership)
        assign_file = os.path.join(best_run, "TCN_AssignMatrix1.csv")
        if not os.path.exists(assign_file):
            print(f"  -> Lỗi: Không tìm thấy TCN_AssignMatrix1.csv tại {best_run}. Bỏ qua.")
            continue
            
        # Ma trận có kích thước (N_cells, N_communities)
        assign_matrix = np.loadtxt(assign_file, delimiter=',')
        
        # Lấy nhãn cụm bằng argmax
        if assign_matrix.ndim == 1: # Trường hợp chỉ có 1 cell? 
            communities = np.argmax(assign_matrix)
            communities = np.array([communities])
        else:
            communities = np.argmax(assign_matrix, axis=1)
            
        # 3. Đọc Coordinates tương ứng
        coord_file = input_dir / f"{image_name}_Coordinates.txt"
        if not coord_file.exists():
            print(f"  -> Lỗi: Không tìm thấy file tọa độ {coord_file.name} trong thư mục input. Bỏ qua.")
            continue
            
        # Tọa độ (x, y)
        coords = np.loadtxt(coord_file, delimiter='\t')
        
        # Đảm bảo số lượng cell khớp nhau
        if len(coords) != len(communities):
            print(f"  -> Lỗi: Số lượng cell không khớp! Coords: {len(coords)}, Matrix: {len(communities)}")
            continue
            
        # 4. Vẽ biểu đồ
        plt.figure(figsize=(10, 8))
        
        # Sử dụng colormap để phân biệt các cộng đồng (communities)
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=communities, cmap='tab20', s=20, alpha=0.9)
        
        plt.colorbar(scatter, label='Community ID')
        plt.title(f"Spatial Communities (TCN Clusters)\nImage: {image_name}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        
        # Đảo ngược trục Y vì tọa độ ảnh gốc nằm góc trái trên
        plt.gca().invert_yaxis()
        plt.axis('equal')
        
        # Lưu kết quả
        output_path = output_viz_dir / f"{image_name}_communities.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"  -> Đã lưu bản đồ cộng đồng: {output_path.name}")
        success_count += 1
        
    print(f"\nHoàn tất! Đã tạo thành công {success_count} bản đồ cộng đồng tế bào.")
    print(f"Kết quả được lưu tại: {output_viz_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CytoCommunity TCN Results.')
    parser.add_argument('--input_dir', type=str, required=True, help='Thư mục chứa *_Coordinates.txt (ví dụ: final_input_for_cyto)')
    parser.add_argument('--output_tcn_dir', type=str, required=True, help='Thư mục chứa kết quả từ Step 2 (ví dụ: ~/CytoCommunity/output_TCN)')
    parser.add_argument('--output_viz_dir', type=str, default='output/communities', help='Thư mục lưu hình ảnh visualization')
    
    args = parser.parse_args()
    
    # Mở rộng path người dùng (tilde ~)
    target_input_dir = os.path.expanduser(args.input_dir)
    target_output_tcn_dir = os.path.expanduser(args.output_tcn_dir)
    target_output_viz_dir = os.path.expanduser(args.output_viz_dir)
    
    visualize_communities(target_input_dir, target_output_tcn_dir, target_output_viz_dir)
