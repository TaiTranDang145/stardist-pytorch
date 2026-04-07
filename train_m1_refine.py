import csv
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import create_dataloaders
from loss import kld_metric, total_loss
from models import StarDist2D


def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrialConfig:
    name: str
    learning_rate: float
    optimizer_name: str
    weight_decay: float
    loss_weights: Tuple[float, float]
    batch_size: int = 2
    patch_size: Tuple[int, int] = (192, 192)
    unet_n_depth: int = 2
    unet_n_filter_base: int = 16
    net_conv_after_unet: int = 64
    grad_clip: float = 0.0
    epochs: int = 6
    steps_per_epoch: int = 30
    foreground_prob: float = 0.9
    n_rays: int = 32
    split_seed: int = 42


def build_trials() -> List[TrialConfig]:
    return [
        TrialConfig("ref01_adam_lr6e4", 6e-4, "adam", 0.0, (1.0, 0.2)),
        TrialConfig("ref02_adam_lr5e4", 5e-4, "adam", 0.0, (1.0, 0.2)),
        TrialConfig("ref03_adam_lr7e4", 7e-4, "adam", 0.0, (1.0, 0.2)),
        TrialConfig("ref04_adamw_wd1e4", 6e-4, "adamw", 1e-4, (1.0, 0.2)),
        TrialConfig("ref05_adamw_wd5e5", 5e-4, "adamw", 5e-5, (1.0, 0.2)),
        TrialConfig("ref06_dist018", 6e-4, "adam", 0.0, (1.0, 0.18)),
        TrialConfig("ref07_dist025", 6e-4, "adam", 0.0, (1.0, 0.25)),
        TrialConfig("ref08_filter24_adamw", 4e-4, "adamw", 1e-4, (1.0, 0.2), unet_n_filter_base=24, net_conv_after_unet=96),
    ]


def make_optimizer(cfg: TrialConfig, model: torch.nn.Module):
    if cfg.optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def train_one_trial(cfg: TrialConfig, device: torch.device, output_dir: Path) -> Dict[str, float]:
    seed_everything(42)

    if device.type == "cuda":
        num_workers = 2
        pin_memory = True
    else:
        # M1/MPS và CPU: ổn định hơn với num_workers=0
        num_workers = 0
        pin_memory = False

    train_loader, val_loader = create_dataloaders(
        root_dir="data/dsb2018/train/",
        patch_size=cfg.patch_size,
        batch_size=cfg.batch_size,
        foreground_prob=cfg.foreground_prob,
        num_workers=num_workers,
        pin_memory=pin_memory,
        split_seed=cfg.split_seed,
    )

    model = StarDist2D(
        n_channels_in=1,
        n_rays=cfg.n_rays,
        grid=(1, 1),
        unet_n_depth=cfg.unet_n_depth,
        unet_n_filter_base=cfg.unet_n_filter_base,
        net_conv_after_unet=cfg.net_conv_after_unet,
    ).to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    start = time.time()

    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_kld", "val_loss", "val_kld", "lr"])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_iter = iter(train_loader)
            train_loss = 0.0
            train_kld = 0.0

            for _ in tqdm(range(cfg.steps_per_epoch), desc=f"{cfg.name} Epoch {epoch} [Train]", leave=False):
                try:
                    images, prob_gt, dist_mask_gt = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    images, prob_gt, dist_mask_gt = next(train_iter)

                images = images.to(device)
                prob_gt = prob_gt.to(device)
                dist_mask_gt = dist_mask_gt.to(device)

                optimizer.zero_grad(set_to_none=True)
                prob_pred, dist_pred = model(images)
                loss = total_loss(
                    prob_pred,
                    dist_pred,
                    prob_gt,
                    dist_mask_gt,
                    loss_weights=cfg.loss_weights,
                )
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                train_loss += loss.item()
                train_kld += kld_metric(prob_gt, prob_pred).item()

            train_loss /= cfg.steps_per_epoch
            train_kld /= cfg.steps_per_epoch

            model.eval()
            val_loss = 0.0
            val_kld = 0.0
            val_steps = 0
            with torch.no_grad():
                for images, prob_gt, dist_mask_gt in tqdm(val_loader, desc=f"{cfg.name} Epoch {epoch} [Val]", leave=False):
                    images = images.to(device)
                    prob_gt = prob_gt.to(device)
                    dist_mask_gt = dist_mask_gt.to(device)
                    prob_pred, dist_pred = model(images)
                    loss = total_loss(
                        prob_pred,
                        dist_pred,
                        prob_gt,
                        dist_mask_gt,
                        loss_weights=cfg.loss_weights,
                    )
                    val_loss += loss.item()
                    val_kld += kld_metric(prob_gt, prob_pred).item()
                    val_steps += 1

            val_loss /= max(val_steps, 1)
            val_kld /= max(val_steps, 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "best_model.pth")

            writer.writerow(
                [epoch, train_loss, train_kld, val_loss, val_kld, optimizer.param_groups[0]["lr"]]
            )

    elapsed = time.time() - start
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    result = {
        "name": cfg.name,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "seconds": float(elapsed),
        "optimizer": cfg.optimizer_name,
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
        "learning_rate": float(cfg.learning_rate),
        "loss_w_prob": float(cfg.loss_weights[0]),
        "loss_w_dist": float(cfg.loss_weights[1]),
        "patch_h": int(cfg.patch_size[0]),
        "patch_w": int(cfg.patch_size[1]),
        "depth": int(cfg.unet_n_depth),
        "filter_base": int(cfg.unet_n_filter_base),
        "conv_after_unet": int(cfg.net_conv_after_unet),
        "params_million": float(params_m),
    }

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def run_refine() -> None:
    device = pick_device()
    print(f"Using device: {device}")

    output_dir = Path("m1_refine_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials()
    results: List[Dict[str, float]] = []

    for i, cfg in enumerate(trials, start=1):
        print(f"\n[{i}/{len(trials)}] {cfg.name}")
        result = train_one_trial(cfg, device, output_dir)
        results.append(result)
        print(
            f"{cfg.name}: best_val_loss={result['best_val_loss']:.4f}, "
            f"epoch={result['best_epoch']}, time={result['seconds']:.1f}s"
        )

    results_sorted = sorted(results, key=lambda x: x["best_val_loss"])
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(results_sorted)

    best_json = output_dir / "best_config.json"
    with best_json.open("w") as f:
        json.dump(results_sorted[0], f, indent=2)

    print("\n=== Top 3 refine runs ===")
    for r in results_sorted[:3]:
        print(
            f"{r['name']}: val={r['best_val_loss']:.4f}, opt={r['optimizer']}, "
            f"lr={r['learning_rate']}, wd={r['weight_decay']}, dist_w={r['loss_w_dist']}"
        )
    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {best_json}")


if __name__ == "__main__":
    run_refine()
