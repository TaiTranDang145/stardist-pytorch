import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset import create_dataloaders, StarDistDataset2D, augmenter
from models import StarDist2D
from loss import total_loss, kld_metric
from evaluate import evaluate_instances

def repeater(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TrainConfig:
    # --- Cấu hình Training ---
    epochs = 10           
    steps_per_epoch = 100          # train_steps_per_epoch gốc
    batch_size = 16               # Tăng lên 16 cho RTX 4060
    learning_rate = 0.0003         # train_learning_rate gốc
    patch_size = (256, 256)
    n_rays = 32
    n_harmonics = 16
    foreground_prob = 0.9          # train_foreground_only
    reg_weight = 1e-4              # train_background_reg
    loss_weights = (1.0, 0.2, 0.2, 0.1)
    save_dir = "checkpoints"
    log_dir = "tensorboard_logs"
    checkpoint_interval = 20       # save mỗi 50 epoch
    early_stop_patience = 40       # tương đương patience trong ReduceLR
    resume = True                  # Tự động load checkpoint nếu có
    
    # --- Cấu hình Đánh giá (Metrics) ---
    eval_prob_thresh = 0.4
    eval_nms_thresh = 0.3
    eval_iou_thresh = 0.5
    
    # --- Cấu hình Kiến trúc Model ---
    # Lưu ý: Nếu thay đổi các thông số này, bạn không thể load checkpoint cũ (size mismatch)
    unet_n_depth = 2               # Khớp với checkpoint cũ
    unet_n_filter_base = 16        # Khớp với checkpoint cũ
    net_conv_after_unet = 64       # Khớp với checkpoint cũ
    
    # --- Thiết bị ---
    device = pick_device()
    num_workers = 4 # Tăng tốc bằng multiprocessing


def train():
    config = TrainConfig()

    checkpoint_path = os.path.join(config.save_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(config.save_dir, "last_model.pth")

    # Chỉ xóa log nếu không resume hoặc không tìm thấy checkpoint
    if os.path.exists(config.log_dir):
        if not (config.resume and (os.path.exists(checkpoint_path) or os.path.exists(last_checkpoint_path))):
            import shutil
            shutil.rmtree(config.log_dir)
            print(f"TensorBoard logs tại: {config.log_dir} (đã xóa log cũ)")
        else:
            print(f"TensorBoard logs tại: {config.log_dir} (tiếp tục log cũ)")

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    print("Chạy lệnh: tensorboard --logdir=tensorboard_logs")

    # DataLoaders
    train_loader, val_loader, val_image_paths, val_mask_paths = create_dataloaders(
        root_dir="data/dsb2018/train/",
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        foreground_prob=config.foreground_prob,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == "cuda" else False,
    )

    # Model
    model = StarDist2D(
        n_channels_in=1,
        n_rays=config.n_rays,
        grid=(1, 1),
        unet_n_depth=config.unet_n_depth,
        unet_n_filter_base=config.unet_n_filter_base,
        net_conv_after_unet=config.net_conv_after_unet,
        n_harmonics=config.n_harmonics,
    ).to(config.device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=40,
        min_lr=1e-7
    )

    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0

    # LOAD CHECKPOINT (RESUME TOÀN BỘ TRẠNG THÁI)
    load_path = last_checkpoint_path if os.path.exists(last_checkpoint_path) else checkpoint_path
    if config.resume and os.path.exists(load_path):
        try:
            checkpoint = torch.load(load_path, map_location=config.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_val_loss = checkpoint.get("best_val_loss", float('inf'))
                patience_counter = checkpoint.get("patience_counter", 0)
                print(f"→ Đã load checkpoint từ {load_path} tại epoch {checkpoint['epoch']}.")
            else:
                missing, unexpected = model.load_state_dict(checkpoint, strict=False)
                print(f"→ Đã load state_dict từ {load_path}.")

            if missing:
                print(f"⚠  Head mới (khởi tạo random): {missing}")
            if unexpected:
                print(f"⚠  Key không dùng trong checkpoint: {unexpected}")

        except RuntimeError as e:
            print("\n" + "="*50)
            print("LỖI: Không thể nạp checkpoint do lệch cấu hình mạng!")
            print(f"Chi tiết lỗi: {e}")
            print("="*50)
            print("HƯỚNG DẪN KHẮC PHỤC:")
            print("1. Nếu muốn dùng model hiện tại (mạnh hơn): Hãy xóa thư mục 'checkpoints' và chạy lại.")
            print("2. Nếu muốn dùng tiếp model cũ: Chỉnh cấu hình trong TrainConfig về:")
            print("   unet_n_depth = 2")
            print("   unet_n_filter_base = 16")
            print("   net_conv_after_unet = 64")
            print("="*50 + "\n")
            return  # Dừng chương trình để người dùng xử lý

    elif config.resume:
        print("→ Không tìm thấy checkpoint cũ, bắt đầu train mới.")

    print(f"Training trên {config.device} | Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    train_iter = iter(repeater(train_loader))

    for epoch in range(start_epoch, config.epochs + 1):
        start_time = time.time()

        # Train một epoch
        model.train()
        train_loss = 0.0
        train_components = np.zeros(4)
        train_kld = 0.0
        train_steps = 0

        for step in tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch} [Train]"):
            images, prob_gt, dist_mask_gt, fourier_gt = next(train_iter)

            images = images.to(config.device)
            prob_gt = prob_gt.to(config.device)
            dist_mask_gt = dist_mask_gt.to(config.device)
            fourier_gt = fourier_gt.to(config.device)

            optimizer.zero_grad()
            prob_pred, dist_pred, fourier_pred, comp_pred = model(images)

            loss, components = total_loss(
                prob_pred, dist_pred, fourier_pred, comp_pred,
                prob_gt, dist_mask_gt, fourier_gt,
                loss_weights=config.loss_weights
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_components += np.array(components)
            train_kld += kld_metric(prob_gt, prob_pred).item()
            train_steps += 1

        train_loss /= train_steps
        train_components /= train_steps
        train_kld /= train_steps

        # Validation
        model.eval()
        val_loss = 0.0
        val_components = np.zeros(4)
        val_kld = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, prob_gt, dist_mask_gt, fourier_gt in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images = images.to(config.device)
                prob_gt = prob_gt.to(config.device)
                dist_mask_gt = dist_mask_gt.to(config.device)
                fourier_gt = fourier_gt.to(config.device)

                prob_pred, dist_pred, fourier_pred, comp_pred = model(images)
                loss, components = total_loss(
                    prob_pred, dist_pred, fourier_pred, comp_pred,
                    prob_gt, dist_mask_gt, fourier_gt,
                    loss_weights=config.loss_weights
                )

                val_loss += loss.item()
                val_components += np.array(components)
                val_kld += kld_metric(prob_gt, prob_pred).item()
                val_steps += 1

        val_loss /= val_steps
        val_components /= val_steps
        val_kld /= val_steps

        # Scheduler step
        scheduler.step(val_loss)

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss_Component/Prob_train", train_components[0], epoch)
        writer.add_scalar("Loss_Component/Dist_train", train_components[1], epoch)
        writer.add_scalar("Loss_Component/Fourier_train", train_components[2], epoch)
        writer.add_scalar("Loss_Component/Complexity_train", train_components[3], epoch)
        writer.add_scalar("KLD/train", train_kld, epoch)
        writer.add_scalar("KLD/val", val_kld, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # Đánh giá Metric F1, Precision, Recall
        print(f"--> Đang chạy inference tập Val (15 ảnh ngẫu nhiên) để tính AP (prob_thresh={config.eval_prob_thresh}, nms_thresh={config.eval_nms_thresh})...")
        import random
        eval_indices = random.sample(range(len(val_image_paths)), min(15, len(val_image_paths)))
        sub_val_img = [val_image_paths[i] for i in eval_indices]
        sub_val_mask = [val_mask_paths[i] for i in eval_indices]
        
        val_stat = evaluate_instances(
            model, sub_val_img, sub_val_mask,
            prob_thresh=config.eval_prob_thresh,
            nms_thresh=config.eval_nms_thresh,
            iou_thresh=config.eval_iou_thresh,
            device=config.device
        )
        
        writer.add_scalar("Metrics/Precision", val_stat.precision, epoch)
        writer.add_scalar("Metrics/Recall", val_stat.recall, epoch)
        writer.add_scalar("Metrics/F1", val_stat.f1, epoch)

        # Thời gian & print
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"Fourier: {train_components[2]:.4f} | "
              f"Complex: {train_components[3]:.4f} | "
              f"Prec: {val_stat.precision:.4f} - Rec: {val_stat.recall:.4f} - F1: {val_stat.f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.2f}s")

        # Chuẩn bị checkpoint dict
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }

        # Lưu last model
        torch.save(checkpoint_dict, last_checkpoint_path)

        # Save checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_dict["best_val_loss"] = best_val_loss
            checkpoint_dict["patience_counter"] = patience_counter
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"→ Saved best model (val_loss = {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping tại epoch {epoch}")
                break

        # Save periodic
        if epoch % 50 == 0:
            periodic_path = os.path.join(config.save_dir, f"model_epoch_{epoch}.pth")
            torch.save(checkpoint_dict, periodic_path)

    writer.close()
    print("Training hoàn tất!")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    train()