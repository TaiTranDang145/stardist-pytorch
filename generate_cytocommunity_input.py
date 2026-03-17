"""
Script trích xuất đặc trưng hình thái học của tế bào từ mask (ví dụ: chạy bằng StarDist)
và ảnh gốc, dùng cho việc xây dựng đầu vào CytoCommunity (Chế độ Unsupervised).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

from skimage.measure import regionprops
from skimage.io import imread
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Tắt cảnh báo KMeans memory leak trên Windows hoặc lỗi OpenMP
warnings.filterwarnings('ignore')

def process_dataset(input_mask_dir, input_image_dir, output_dir, num_clusters=5):
    # Dùng pathlib để quản lý đường dẫn cho gọn và sạch
    input_mask_dir = Path(input_mask_dir)
    input_image_dir = Path(input_image_dir)
    output_dir = Path(output_dir)
    
    # 7. Tạo thư mục output nếu chưa có
    output_label_dir = output_dir / "CellTypeLabels"
    output_coord_dir = output_dir / "Coordinates"
    output_vis_dir = output_dir / "visualization"
    
    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_coord_dir.mkdir(parents=True, exist_ok=True)
    output_vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy tất cả file mask (hỗ trợ .tif, .tiff, .npy)
    valid_exts = ['.tif', '.tiff', '.npy']
    mask_files = [f for f in input_mask_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
    
    if not mask_files:
        print(f"Không tìm thấy file mask nào trong {input_mask_dir}.")
        return
        
    print(f"Bắt đầu xử lý {len(mask_files)} file masks...\n")
    
    success_count = 0
    
    for mask_path in mask_files:
        print(f"Đang xử lý: {mask_path.name}")
        
        # Đọc file mask
        if mask_path.suffix.lower() == '.npy':
            mask = np.load(mask_path)
        else:
            mask = imread(mask_path)
            
        # Suy luận tên ảnh gốc từ mask (gỡ bỏ '_labels' nếu có)
        # VD: Image_0032_labels.tif -> Image_0032
        base_name = mask_path.stem
        if base_name.endswith('_labels'):
            img_base = base_name.replace('_labels', '')
        else:
            img_base = base_name
            
        # Đọc ảnh gốc để tính mean_intensity
        image = None
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            potential_img_path = input_image_dir / (img_base + ext)
            if potential_img_path.exists():
                image = imread(potential_img_path)
                break
                
        if image is None:
            print(f"  -> Cảnh báo: Không tìm thấy ảnh gốc cho {img_base}. Tính mean_intensity = 0.")
            image = np.zeros_like(mask)
            
        # 1. Trích xuất morphology features
        props = regionprops(mask, intensity_image=image)
        
        # Xử lý lỗi nếu mask trống (không có tế bào)
        if len(props) == 0:
            print(f"  -> Bỏ qua: Không tìm thấy tế bào nào trong mask này.")
            continue
            
        features = []
        centroids = []
        
        # 2. Các features bắt buộc
        for prop in props:
            area = prop.area
            perimeter = prop.perimeter
            
            # Xử lý ngoại lệ circularity khi perimeter = 0 (trường hợp cực hiếm)
            if perimeter == 0:
                circularity = 0.0
            else:
                circularity = 4 * math.pi * area / (perimeter ** 2)
                
            intensity = prop.mean_intensity
            # Nếu ảnh có nhiều kênh màu, lấy trung bình các kênh để ra 1 giá trị intensity chung
            if isinstance(intensity, (np.ndarray, list, tuple)):
                intensity = np.mean(intensity)
                
            feat = [
                area,
                perimeter,
                prop.eccentricity,
                circularity,
                intensity if intensity is not None else 0.0,
                prop.major_axis_length,
                prop.minor_axis_length,
                prop.solidity,
                prop.extent
            ]
            
            features.append(feat)
            centroids.append(prop.centroid) # tọa độ dạng (row, col) = (y, x)
            
        features = np.array(features)
        centroids = np.array(centroids)
        
        # 3. Chuẩn hóa tất cả features
        scaler = StandardScaler()
        # Nếu chỉ có lèo tèo vài cell, scaler hoặc KMeans có thể không hoạt động tốt, tuy nhiên ở đây xử lý chung
        if len(features) < 2:
            print(f"  -> Chú ý: Chỉ có {len(features)} tế bào, quá ít để gom cụm!")
            features_scaled = features
            labels_kmeans = [0] * len(features)
        else:
            features_scaled = scaler.fit_transform(features)
            
            # 4. Thực hiện clustering bằng KMeans
            n_clusters_actual = min(num_clusters, len(features_scaled))
            kmeans = KMeans(n_clusters=n_clusters_actual, n_init='auto', random_state=42)
            labels_kmeans = kmeans.fit_predict(features_scaled)
            
        # In ra bảng phân bổ số lượng tế bào theo cluster
        unique_labels, counts = np.unique(labels_kmeans, return_counts=True)
        print("  -> Phân bổ cluster:")
        for label, count in zip(unique_labels, counts):
            print(f"     Type{label + 1}: {count} tế bào")
            
        # 5. Gán label dạng string (Thứ tự tự động khớp 100% với regionprops list)
        cell_types = [f"Type{label + 1}" for label in labels_kmeans]
        
        # 6. Ghi output ra txt format CytoCommunity: [tên_ảnh]_CellTypeLabel.txt
        output_txt_path = output_label_dir / f"{img_base}_CellTypeLabel.txt"
        with open(output_txt_path, 'w') as f:
            for ct in cell_types:
                f.write(f"{ct}\n")
                
        # 7. Ghi output tọa độ (Coordinates) ra txt format CytoCommunity: [tên_ảnh]_Coordinates.txt
        output_coord_path = output_coord_dir / f"{img_base}_Coordinates.txt"
        # centroids[:, [1, 0]] dùng để tráo đổi thứ tự từ (y, x) thành (x, y) để tương thích với output x\ty
        np.savetxt(output_coord_path, centroids[:, [1, 0]], fmt='%.6f', delimiter='\t')
        print(f"  -> Đã lưu tọa độ: {output_coord_path} với {len(centroids)} tế bào")
                
        # 8. Bonus: Vẽ hình visualize nhanh
        plt.figure(figsize=(8, 8))
        # tọa độ gốc của matrix img là (row, col) nên vẽ sẽ là Y=row, X=col
        y, x = centroids[:, 0], centroids[:, 1]
        
        # Vẽ scatter plot (tô màu theo cluster_id)
        scatter = plt.scatter(x, y, c=labels_kmeans, cmap='Set1', s=15, alpha=0.8)
        
        plt.title(f"Unsupervised Clustering ({len(features)} cells)\nImage: {img_base}")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        # Lật trục Y vì tọa độ gốc của ảnh nằm góc trái trên
        plt.gca().invert_yaxis()
        plt.axis('equal') # Giữ tỉ lệ không để bị méo
        
        output_png_path = output_vis_dir / f"{img_base}_cluster.png"
        plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        success_count += 1
        print(f"  -> Đã lưu output text, tọa độ và hình ảnh!")
        
    print("-" * 50)
    print(f"Đã tạo {success_count} file CellTypeLabel.txt và {success_count} file Coordinates.txt thành công!")
    print(f"Các file được lưu tại thư mục: {output_dir.absolute()}")

if __name__ == "__main__":
    # ================= CẤU HÌNH BIẾN =================
    # Khai báo đường dẫn: Chỉnh sửa lại cho đúng thư mục của bạn
    
    # 1. Thư mục chứa Mask xuất ra từ StarDist
    INPUT_MASK_DIR = "data/dsb2018/test/masks"
    
    # 2. Thư mục chứa Original Image
    # (Nếu không có, cứ để path tới thư mục chứa ảnh. Nếu không tìm thấy, code tự set mean_intensity=0)
    INPUT_IMAGE_DIR = "data/dsb2018/test/images"
    
    # 3. Thư mục đầu ra
    OUTPUT_DIR = "output"
    
    # 4. Số cụm Type (Mặc định: 5)
    NUM_CLUSTERS = 5
    # =================================================
    
    process_dataset(INPUT_MASK_DIR, INPUT_IMAGE_DIR, OUTPUT_DIR, num_clusters=NUM_CLUSTERS)
