import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Import các phần bạn đã có
from dataset import create_dataloaders, StarDistDataset2D, augmenter
from models import StarDist2D
from loss import total_loss, kld_metric

# ====================== Config training (sát với Config2D gốc) ======================
class TrainConfig:
    epochs = 50                  # train_epochs gốc
    steps_per_epoch = 100          # train_steps_per_epoch gốc
    batch_size = 8               # train_batch_size gốc (RTX 3090 có thể tăng lên 8-16)
    learning_rate = 0.0003         # train_learning_rate gốc
    patch_size = (256, 256)
    n_rays = 32
    foreground_prob = 0.9          # train_foreground_only
    reg_weight = 1e-4              # train_background_reg
    loss_weights = (1.0, 0.2)      # train_loss_weights cho single-class
    save_dir = "checkpoints"
    log_dir = "tensorboard_logs"
    checkpoint_interval = 50       # save mỗi 50 epoch
    early_stop_patience = 40       # tương đương patience trong ReduceLR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    config = TrainConfig()

    # Tạo thư mục
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    print(f"TensorBoard logs tại: {config.log_dir}")
    print("Chạy lệnh: tensorboard --logdir=tensorboard_logs")

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        root_dir="data/dsb2018/train/",
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        foreground_prob=config.foreground_prob,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = StarDist2D(
        n_channels_in=1,
        n_rays=config.n_rays,
        grid=(1,1),
        unet_n_depth=3,
        unet_n_filter_base=32,
        net_conv_after_unet=128,
    ).to(config.device)

    # Optimizer & Scheduler (ReduceLROnPlateau như gốc)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=40,           # giống train_reduce_lr gốc
        min_lr=1e-7,
        verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training trên {config.device} | Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        # Train một epoch (steps_per_epoch giới hạn nếu cần)
        model.train()
        train_loss = 0.0
        train_kld = 0.0
        train_steps = 0

        train_iter = iter(train_loader)
        for step in tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch} [Train]"):
            try:
                images, prob_gt, dist_mask_gt = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, prob_gt, dist_mask_gt = next(train_iter)

            images = images.to(config.device)
            prob_gt = prob_gt.to(config.device)
            dist_mask_gt = dist_mask_gt.to(config.device)

            optimizer.zero_grad()
            prob_pred, dist_pred = model(images)

            loss = total_loss(
                prob_pred, dist_pred,
                prob_gt, dist_mask_gt,
                loss_weights=config.loss_weights
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_kld += kld_metric(prob_gt, prob_pred).item()
            train_steps += 1

        train_loss /= train_steps
        train_kld /= train_steps

        # Validation (full val loader)
        model.eval()
        val_loss = 0.0
        val_kld = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, prob_gt, dist_mask_gt in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images = images.to(config.device)
                prob_gt = prob_gt.to(config.device)
                dist_mask_gt = dist_mask_gt.to(config.device)

                prob_pred, dist_pred = model(images)
                loss = total_loss(
                    prob_pred, dist_pred,
                    prob_gt, dist_mask_gt,
                    loss_weights=config.loss_weights
                )

                val_loss += loss.item()
                val_kld += kld_metric(prob_gt, prob_pred).item()
                val_steps += 1

        val_loss /= val_steps
        val_kld /= val_steps

        # Scheduler step
        scheduler.step(val_loss)

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("KLD/train", train_kld, epoch)
        writer.add_scalar("KLD/val", val_kld, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # Thời gian & print
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train KLD: {train_kld:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val KLD: {val_kld:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.2f}s")

        # Save checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pth"))
            print(f"→ Saved best model (val_loss = {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 40:  # giống patience gốc
                print(f"Early stopping tại epoch {epoch}")
                break

        # Save periodic
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_dir, f"model_epoch_{epoch}.pth"))

    writer.close()
    print("Training hoàn tất!")


if __name__ == "__main__":
    train()