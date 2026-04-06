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
from loss import total_loss_with_boundary, kld_metric

class TrainConfig:
    # Training parameters
    epochs = 100
    steps_per_epoch = 100
    batch_size = 16
    learning_rate = 0.0003
    patch_size = (256, 256)
    n_rays = 32
    foreground_prob = 0.9
    reg_weight = 1e-4
    
    # === ĐIỀU CHỈNH TRỌNG SỐ LOSS TẠI ĐÂY ===
    # Format: (λ1_prob, λ2_dist, λ3_boundary)
    # λ1: BCE loss cho probability
    # λ2: MAE loss cho distance
    # λ3: Dice loss cho boundary
    loss_weights = (1.0, 0.2, 0.5)  # Recommended: cân bằng
    # Các tùy chọn khác:
    # loss_weights = (1.0, 0.2, 1.0)   # Ưu tiên boundary (tế bào dính chặt)
    # loss_weights = (1.0, 0.5, 0.3)   # Ưu tiên distance (tế bào tách rời)
    # loss_weights = (1.0, 0.3, 0.8)   # Custom tuning
    
    # === ATTENTION MECHANISMS CONFIGURATION ===
    use_attention = 'se'          # 'se', 'cbam', or None
    use_attention_gate = True     # True/False - Spatial attention
    use_boundary_head = True      # True/False - Boundary prediction head
    
    # Model architecture (Optimized - giảm 93.7% params: 2M → 126K)
    unet_n_depth = 2          # Giảm từ 3 → 2 layers
    unet_n_filter_base = 16   # Giảm từ 32 → 16 filters
    net_conv_after_unet = 32  # Giảm từ 128 → 32 filters
    
    # Checkpointing
    save_dir = "checkpoints"
    log_dir = "tensorboard_logs"
    checkpoint_interval = 20
    early_stop_patience = 40
    resume = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    config = TrainConfig()

    if os.path.exists(config.log_dir):
        import shutil
        shutil.rmtree(config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    print(f"TensorBoard logs tại: {config.log_dir} (đã xóa log cũ)")
    print("Chạy lệnh: tensorboard --logdir=tensorboard_logs")

    # DataLoaders
    # FIX: Windows multiprocessing issue - set num_workers=0
    train_loader, val_loader = create_dataloaders(
        root_dir="data/dsb2018/train/",
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        foreground_prob=config.foreground_prob,
        num_workers=0,  # Fix Windows multiprocessing error
        pin_memory=False,  # Set False cho stability trên Windows
    )

    # Model với Attention-Enhanced U-Net
    print(f"\n{'='*60}")
    print(f"KHỞI TẠO MODEL - Attention-Enhanced StarDist2D")
    print(f"{'='*60}")
    print(f"Attention Type: {config.use_attention}")
    print(f"Attention Gate: {config.use_attention_gate}")
    print(f"Boundary Head: {config.use_boundary_head}")
    print(f"Loss Weights: Prob={config.loss_weights[0]}, "
          f"Dist={config.loss_weights[1]}, Boundary={config.loss_weights[2]}")
    print(f"{'='*60}\n")
    
    model = StarDist2D(
        n_channels_in=1,
        n_rays=config.n_rays,
        grid=(1,1),
        unet_n_depth=config.unet_n_depth,
        unet_n_filter_base=config.unet_n_filter_base,
        net_conv_after_unet=config.net_conv_after_unet,
        use_attention=config.use_attention,
        use_attention_gate=config.use_attention_gate,
        use_boundary_head=config.use_boundary_head,
    ).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}\n")
    
    # LOAD CHECKPOINT (RESUME)
    checkpoint_path = os.path.join(config.save_dir, "best_model.pth")
    if config.resume and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
            print(f"→ Đã load checkpoint từ {checkpoint_path} để tiếp tục train.")
        except:
            print("→ Cấu trúc model thay đổi (Fourier), bắt đầu train mới.")
    elif config.resume:
        print("→ Không tìm thấy checkpoint cũ, bắt đầu train mới.")

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=40, min_lr=1e-7
    )

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training trên {config.device} | Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        # Train một epoch
        model.train()
        train_loss = 0.0
        train_components = np.zeros(3)  # prob, dist, boundary
        train_kld = 0.0
        train_steps = 0

        train_iter = iter(train_loader)
        for step in tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch} [Train]"):
            try:
                images, prob_gt, dist_mask_gt, fourier_gt = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, prob_gt, dist_mask_gt, fourier_gt = next(train_iter)

            images = images.to(config.device)
            prob_gt = prob_gt.to(config.device)
            dist_mask_gt = dist_mask_gt.to(config.device)
            # fourier_gt chứa boundary ở kênh cuối (tạm thời dùng complexity làm boundary)
            boundary_gt = fourier_gt[:, -1:].to(config.device)  # Lấy kênh cuối làm boundary

            optimizer.zero_grad()
            
            # Forward pass - model trả về dictionary
            outputs = model(images)
            prob_pred = outputs['prob']
            dist_pred = outputs['dist']
            boundary_pred = outputs['boundary']

            # Compute loss với boundary
            loss, components = total_loss_with_boundary(
                prob_pred, dist_pred, boundary_pred,
                prob_gt, dist_mask_gt, boundary_gt,
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
        val_components = np.zeros(3)  # prob, dist, boundary
        val_kld = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, prob_gt, dist_mask_gt, fourier_gt in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images = images.to(config.device)
                prob_gt = prob_gt.to(config.device)
                dist_mask_gt = dist_mask_gt.to(config.device)
                boundary_gt = fourier_gt[:, -1:].to(config.device)

                # Forward pass
                outputs = model(images)
                prob_pred = outputs['prob']
                dist_pred = outputs['dist']
                boundary_pred = outputs['boundary']
                
                loss, components = total_loss_with_boundary(
                    prob_pred, dist_pred, boundary_pred,
                    prob_gt, dist_mask_gt, boundary_gt,
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
        writer.add_scalar("Loss_Component/Boundary_train", train_components[2], epoch)
        writer.add_scalar("Loss_Component/Prob_val", val_components[0], epoch)
        writer.add_scalar("Loss_Component/Dist_val", val_components[1], epoch)
        writer.add_scalar("Loss_Component/Boundary_val", val_components[2], epoch)
        writer.add_scalar("KLD/train", train_kld, epoch)
        writer.add_scalar("KLD/val", val_kld, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # Thời gian & print
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"Prob: {train_components[0]:.4f} | "
              f"Dist: {train_components[1]:.4f} | "
              f"Boundary: {train_components[2]:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # Save checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pth"))
            print(f"→ Saved best model (val_loss = {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping tại epoch {epoch}")
                break

        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_dir, f"model_epoch_{epoch}.pth"))

    writer.close()
    print("Training hoàn tất!")


if __name__ == "__main__":
    train()