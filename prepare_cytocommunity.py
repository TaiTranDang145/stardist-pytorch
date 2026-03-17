import os
import shutil
from pathlib import Path

# Màu sắc ANSI để in cho đẹp
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"

def count_lines(filepath):
    """
    Hàm đếm số dòng trong một file text.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"{COLOR_RED}Lỗi khi đọc file {filepath}: {e}{COLOR_RESET}")
        return -1

def prepare_cytocommunity_input(output_parent_dir, final_input_dir):
    """
    Gộp file CellTypeLabel và Coordinates, kiểm tra tính đồng nhất 
    và copy vào thư mục final input.
    """
    output_parent_dir = Path(output_parent_dir)
    final_input_dir = Path(final_input_dir)
    
    label_dir = output_parent_dir / "CellTypeLabels"
    coord_dir = output_parent_dir / "Coordinates"
    
    # 3. Tạo thư mục đích nếu chưa có
    final_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra thư mục nguồn
    if not label_dir.exists() or not coord_dir.exists():
        print(f"{COLOR_RED}LỖI: Không tìm thấy thư mục CellTypeLabels hoặc Coordinates trong {output_parent_dir}!{COLOR_RESET}")
        return
        
    # Lấy danh sách file Coordinate
    coord_files = list(coord_dir.glob("*_Coordinates.txt"))
    if not coord_files:
        print(f"{COLOR_RED}LỖI: Không tìm thấy file *_Coordinates.txt nào trong {coord_dir}!{COLOR_RESET}")
        return

    # 1. Tự động trích xuất base_name và tạo ImageNameList.txt
    valid_base_names = []
    
    # Dictionary lưu trữ thông tin kiểm tra
    stats = {
        'valid_images': 0,
        'invalid_images': 0,
        'failed_names': [],
        'cell_counts': []
    }
    
    print(f"{COLOR_CYAN}Bắt đầu kiểm tra tính nhất quán dữ liệu cho {len(coord_files)} files...{COLOR_RESET}\n")
    
    # 2. Kiểm tra tính nhất quán dữ liệu
    for coord_file in coord_files:
        # Tên file dạng: Image_0032_Coordinates.txt -> lấy phần trước "_Coordinates"
        base_name = coord_file.name.replace("_Coordinates.txt", "")
        
        # Đường dẫn file label tương ứng
        label_file = label_dir / f"{base_name}_CellTypeLabel.txt"
        
        if not label_file.exists():
            print(f"{COLOR_RED}[WARNING] Thiếu file Label: {base_name}_CellTypeLabel.txt không tồn tại!{COLOR_RESET}")
            stats['invalid_images'] += 1
            stats['failed_names'].append(base_name)
            continue
            
        # Đếm số dòng
        coord_count = count_lines(coord_file)
        label_count = count_lines(label_file)
        
        if coord_count == -1 or label_count == -1:
            stats['invalid_images'] += 1
            stats['failed_names'].append(base_name)
            continue
            
        # Kiểm tra khớp cell count
        if coord_count != label_count:
            print(f"{COLOR_RED}[WARNING] Lỗi lệch số lượng {base_name}: Coordinates có {coord_count} dòng, Label có {label_count} dòng!{COLOR_RESET}")
            stats['invalid_images'] += 1
            stats['failed_names'].append(base_name)
            continue
            
        # Nếu rỗng (0 dòng)
        if coord_count == 0:
            print(f"{COLOR_YELLOW}[SKIP] Bỏ qua {base_name}: File rỗng (không có tế bào).{COLOR_RESET}")
            stats['invalid_images'] += 1
            stats['failed_names'].append(base_name)
            continue
            
        # Hợp lệ -> lưu lại
        valid_base_names.append(base_name)
        stats['valid_images'] += 1
        stats['cell_counts'].append(coord_count)
        
        # Co-py các file đã hợp lệ thành công qua thư mục "final_input_for_cyto" (copy đè nếu có)
        shutil.copy2(coord_file, final_input_dir / coord_file.name)
        shutil.copy2(label_file, final_input_dir / label_file.name)

    # Sắp xếp theo alphabetical order / numeric order
    valid_base_names.sort()

    # Ghi ImageNameList.txt vào mục đích luôn
    list_path = final_input_dir / "ImageNameList.txt"
    with open(list_path, 'w', encoding='utf-8') as f:
        for name in valid_base_names:
            f.write(f"{name}\n")

    # In báo cáo tổng
    print(f"\n{COLOR_CYAN}================ BÁO CÁO KIỂM TRA DỮ LIỆU ================{COLOR_RESET}")
    print(f"Tổng số ảnh quét:      {COLOR_CYAN}{len(coord_files)}{COLOR_RESET}")
    print(f"Số ảnh hợp lệ:         {COLOR_GREEN}{stats['valid_images']}{COLOR_RESET}")
    print(f"Số ảnh lỗi/bỏ qua:     {COLOR_RED}{stats['invalid_images']}{COLOR_RESET}")
    
    if stats['failed_names']:
        print(f"[{COLOR_RED}WARNING{COLOR_RESET}] Các ảnh bị lỗi: {', '.join(stats['failed_names'][:5])}" + ("..." if len(stats['failed_names']) > 5 else ""))
        
    if stats['valid_images'] > 0:
        avg_cells = sum(stats['cell_counts']) / len(stats['cell_counts'])
        min_cells = min(stats['cell_counts'])
        max_cells = max(stats['cell_counts'])
        
        print("\n{COLOR_CYAN}--- THÔNG KÊ SỐ LƯỢNG TẾ BÀO ---{COLOR_RESET}".format(COLOR_CYAN=COLOR_CYAN, COLOR_RESET=COLOR_RESET))
        print(f"Trung bình: {COLOR_GREEN}{avg_cells:.2f}{COLOR_RESET} tế bào/ảnh")
        print(f"Tối thiểu:  {min_cells} tế bào")
        print(f"Tối đa:     {max_cells} tế bào")
        
        # 4. Gợi ý tham số K-NN
        suggested_k = 8
        if avg_cells < 20:
            suggested_k = 5
        elif avg_cells > 100:
            suggested_k = 10
            
        print(f"\n{COLOR_GREEN}>> Gợi ý số tham số k (Nearest Neighbors) cho K-NN graph Step 1: k = {suggested_k} (nằm trong khoảng 5-10 tùy mật độ).{COLOR_RESET}")
    
    print(f"{COLOR_CYAN}=========================================================={COLOR_RESET}\n")

    if stats['valid_images'] > 0:
        print(f"{COLOR_GREEN}Đã gom xong {stats['valid_images']} bộ dữ liệu vào thư mục: {final_input_dir.absolute()}{COLOR_RESET}\n")
        
        # In hướng dẫn lệnh Terminal
        input_abs_path = final_input_dir.absolute()
        
        print(f"{COLOR_YELLOW}Bây giờ bạn có thể kết nối với thư mục repo CytoCommunity của bạn và chạy: {COLOR_RESET}\n")
        
        print(f"1. Tạo đồ thị tế bào (Spatial Graphs):")
        print(f"   {COLOR_CYAN}python Step1_ConstructCellularSpatialGraphs.py --input_dir {input_abs_path} --output_dir ~/CytoCommunity/output_graphs --k {suggested_k}{COLOR_RESET}\n")
        
        print(f"2. Chạy học Unsupervised TCN (Graph Neural Network):")
        print(f"   {COLOR_CYAN}python Step2_TCNLearning_Unsupervised.py --graph_dir ~/CytoCommunity/output_graphs/ --output_dir ~/CytoCommunity/output_TCN/ --Num_TCN 6 --Num_Run 20 --lr 0.001 --epochs 300 --device cuda{COLOR_RESET}\n")
        
        print(f"3. Bonus: Lấy membership sau đó có thể vizualize bằng script (Vị dụ nếu repo có visualize.py):")
        print(f"   {COLOR_CYAN}python visualize.py --membership_dir ~/CytoCommunity/output_TCN/ --coord_dir {input_abs_path}{COLOR_RESET}\n")

if __name__ == "__main__":
    # ================= CẤU HÌNH BIẾN =================
    # Thư mục chứa các output vừa xuất từ bước 1
    OUTPUT_PARENT_DIR = "output"
    
    # 3. Thư mục tổng hợp gọn gàng cho bước CytoCommunity
    FINAL_INPUT_DIR = "final_input_for_cyto"
    # =================================================
    
    prepare_cytocommunity_input(OUTPUT_PARENT_DIR, FINAL_INPUT_DIR)
