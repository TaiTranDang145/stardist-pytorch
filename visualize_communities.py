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

def visualize_communities(input_dir, output_tcn_dir, graph_dir, output_viz_dir, image_dir=None):
    """
    Đọc kết quả TCN và vẽ bản đồ không gian các cộng đồng tế bào, bao gồm ảnh nền và các liên kết.
    """
    input_dir = Path(input_dir)
    output_tcn_dir = Path(output_tcn_dir)
    graph_dir = Path(graph_dir)
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
            
        assign_matrix = np.loadtxt(assign_file, delimiter=',')
        if assign_matrix.ndim == 1:
            communities = np.argmax(assign_matrix)
            communities = np.array([communities])
        else:
            communities = np.argmax(assign_matrix, axis=1)
            
        # 3. Đọc Coordinates
        coord_file = input_dir / f"{image_name}_Coordinates.txt"
        if not coord_file.exists():
            print(f"  -> Lỗi: Không tìm thấy file tọa độ {coord_file.name}. Bỏ qua.")
            continue
        coords = np.loadtxt(coord_file, delimiter='\t')
        
        # 4. Đọc Edges từ Step 1
        edge_file = graph_dir / f"{image_name}_EdgeIndex.txt"
        edges = None
        if edge_file.exists():
            edges = np.loadtxt(edge_file, dtype=int, delimiter='\t')
        else:
            print(f"  -> Cảnh báo: Không tìm thấy file liên kết {edge_file.name}. Sẽ không vẽ các đường nối.")

        # 5. Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Thử nạp ảnh nền nếu có
        if image_dir:
            from skimage.io import imread
            # Tìm ảnh (thử các đuôi phổ biến)
            img = None
            for ext in ['.png', '.tif', '.tiff', '.jpg']:
                img_path = Path(image_dir) / f"{image_name}{ext}"
                if img_path.exists():
                    img = imread(img_path)
                    break
            if img is not None:
                ax.imshow(img, alpha=0.5) # Vẽ ảnh mờ đi một chút để nổi bật tế bào

        # Vẽ các liên kết (Edges)
        if edges is not None:
            for edge in edges:
                u, v = edge
                # Lấy tọa độ (x, y) của 2 node
                pos_u = coords[u]
                pos_v = coords[v]
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 
                        color='gray', linewidth=0.5, alpha=0.3, zorder=1)

        # Vẽ tế bào
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=communities, 
                             cmap='tab20', s=40, edgecolors='black', linewidths=0.5, zorder=2)
        
        plt.colorbar(scatter, ax=ax, label='Community ID')
        ax.set_title(f"Spatial Communities (TCN Clusters)\nImage: {image_name}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Nếu không có ảnh nền thì đảo ngược trục Y cho đúng hệ tọa độ ảnh
        if img is None:
            ax.invert_yaxis()
        
        ax.axis('equal')
        
        # Lưu kết quả
        output_path = output_viz_dir / f"{image_name}_communities_rich.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"  -> Đã lưu bản đồ phong phú: {output_path.name}")
        success_count += 1
        
    print(f"\nHoàn tất! Đã tạo thành công {success_count} bản đồ cộng đồng tế bào.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CytoCommunity TCN Results with Background & Edges.')
    parser.add_argument('--input_dir', type=str, required=True, help='Thư mục chứa *_Coordinates.txt')
    parser.add_argument('--output_tcn_dir', type=str, required=True, help='Thư mục chứa kết quả từ Step 2')
    parser.add_argument('--graph_dir', type=str, required=True, help='Thư mục chứa *_EdgeIndex.txt từ Step 1')
    parser.add_argument('--image_dir', type=str, default=None, help='(Tùy chọn) Thư mục chứa ảnh gốc (.png, .tif)')
    parser.add_argument('--output_viz_dir', type=str, default='output/communities', help='Thư mục lưu hình ảnh')
    
    args = parser.parse_args()
    
    visualize_communities(
        os.path.expanduser(args.input_dir),
        os.path.expanduser(args.output_tcn_dir),
        os.path.expanduser(args.graph_dir),
        os.path.expanduser(args.output_viz_dir),
        os.path.expanduser(args.image_dir) if args.image_dir else None
    )
