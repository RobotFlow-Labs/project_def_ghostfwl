#!/usr/bin/env python3
"""KITTI LiDAR ghost detection training — GPU-optimized pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/train_kitti.py --config configs/kitti_train.toml
    CUDA_VISIBLE_DEVICES=6 python scripts/train_kitti.py --config configs/kitti_train.toml --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from anima_def_ghostfwl.data.kitti_dataset import KITTIVoxelDataset
from anima_def_ghostfwl.data.kitti_voxelize import cache_kitti_voxels
from anima_def_ghostfwl.models.ghost_detector_3d import GhostDetector3D


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, mode: str = "min"):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> None:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        # Always keep best
        best_val, best_path = self.history[0]
        import shutil
        shutil.copy2(best_path, self.save_dir / "best.pth")


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta)
            if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def find_batch_size(model: nn.Module, device: torch.device, target_util: float = 0.65) -> int:
    """Binary search for optimal batch size targeting VRAM utilization."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory

    low, high = 1, 64
    best_bs = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.randn(mid, 2, 256, 256, 32, device=device)
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(dummy)
                loss = out.sum()
            loss.backward()
            model.zero_grad()
            del dummy, out, loss
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated(device)
            util = peak / total_memory
            print(f"  [BATCH] bs={mid}: {peak / 1e9:.2f}GB / "
                  f"{total_memory / 1e9:.2f}GB = {util:.1%}")

            if util <= target_util:
                best_bs = mid
                low = mid + 1
            else:
                high = mid - 1
        except RuntimeError:
            high = mid - 1
            torch.cuda.empty_cache()

    return best_bs


def train(config: dict, *, dry_run: bool = False, resume: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CONFIG] Using device: {device}")
    print(f"[CONFIG] CUDA: {torch.version.cuda}")

    # Log GPU info
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {props.name}, {props.total_memory / 1e9:.1f}GB VRAM")

    torch.manual_seed(config["training"]["seed"])

    # Step 0: Ensure voxel cache exists
    cache_dir = Path(config["data"]["cache_dir"])
    velodyne_dir = Path(config["data"]["velodyne_dir"])
    if not list(cache_dir.glob("*.pt")):
        print("[VOXELIZE] Cache empty — building GPU voxel cache...")
        cache_kitti_voxels(velodyne_dir, cache_dir, device=str(device))

    # Step 1: Load datasets
    train_ds = KITTIVoxelDataset(cache_dir, split="train", augment=True)
    val_ds = KITTIVoxelDataset(cache_dir, split="val")

    print(f"[DATA] train={len(train_ds)} samples, val={len(val_ds)} samples")

    # Step 2: Build model
    model_cfg = config.get("model", {})
    model = GhostDetector3D(
        in_channels=model_cfg.get("in_channels", 2),
        num_classes=model_cfg.get("num_classes", 3),
        base_ch=model_cfg.get("base_channels", 32),
    ).to(device)
    print(f"[MODEL] {model.param_count:,} parameters ({model.trainable_param_count:,} trainable)")

    # Step 3: Auto batch size
    train_cfg = config["training"]
    if train_cfg.get("batch_size") == "auto":
        batch_size = find_batch_size(model, device, target_util=0.65)
    else:
        batch_size = int(train_cfg["batch_size"])
    print(f"[BATCH] batch_size={batch_size}")

    if dry_run:
        print("[DRY RUN] Config validated. Exiting.")
        return

    # Step 4: DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Step 5: Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 200),
        total_steps=total_steps,
    )
    scaler = GradScaler("cuda")

    # Step 6: Checkpoint + early stopping
    ckpt_cfg = config["checkpoint"]
    ckpt_mgr = CheckpointManager(
        Path(ckpt_cfg["output_dir"]),
        keep_top_k=ckpt_cfg.get("keep_top_k", 2),
        mode=ckpt_cfg.get("mode", "min"),
    )

    es_cfg = config.get("early_stopping", {})
    early_stop = EarlyStopping(
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.0001),
    )

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[RESUME] from epoch {start_epoch}, step {global_step}")

    # Focal-like class weights: empty is ~95% of voxels, ghost is rare
    class_weights = torch.tensor([0.05, 1.0, 3.0], device=device)

    # Log setup
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir / "metrics.jsonl"

    # Print training header
    print(f"\n[TRAIN] epochs={train_cfg['epochs']}, lr={train_cfg['learning_rate']}, "
          f"optimizer=AdamW, scheduler=cosine_warmup")
    save_n = ckpt_cfg.get('save_every_n_steps', 200)
    print(f"[TRAIN] total_steps={total_steps}, save_every={save_n}")

    # Check VRAM at start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        mem_used = torch.cuda.memory_allocated(device)
        mem_total = torch.cuda.get_device_properties(0).total_memory
        vram_pct = mem_used / mem_total
        print(f"[VRAM] start: {mem_used / 1e9:.2f}GB / "
              f"{mem_total / 1e9:.2f}GB = {vram_pct:.1%}")

    # Step 7: Training loop
    model = torch.compile(model)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.perf_counter()

        for batch in train_loader:
            voxel = batch["voxel"].to(device, non_blocking=True)
            target = batch["ghost_label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(voxel)
                loss = F.cross_entropy(
                    logits, target, weight=class_weights, ignore_index=-1
                )

            if torch.isnan(loss):
                print(f"[FATAL] Loss is NaN at epoch {epoch}, step {global_step} — stopping")
                return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Save checkpoint
            if global_step % ckpt_cfg.get("save_every_n_steps", 200) == 0:
                val_loss = validate(model, val_loader, device, class_weights)
                ckpt_mgr.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": val_loss,
                        "config": config,
                    },
                    val_loss,
                    global_step,
                )
                print(
                    f"  [CKPT] step={global_step} val_loss={val_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.perf_counter() - t_epoch
        steps_per_sec = epoch_steps / max(elapsed, 1e-9)

        # Validation
        val_loss = validate(model, val_loader, device, class_weights)

        # VRAM check
        if device.type == "cuda":
            mem_used = torch.cuda.max_memory_allocated(device)
            mem_total = torch.cuda.get_device_properties(0).total_memory
            vram_pct = mem_used / mem_total * 100
        else:
            vram_pct = 0

        log_line = (
            f"[Epoch {epoch + 1}/{train_cfg['epochs']}] "
            f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"steps/sec={steps_per_sec:.1f} VRAM={vram_pct:.0f}%"
        )
        print(log_line)

        # Log metrics
        with open(metrics_file, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "steps_per_sec": steps_per_sec,
                "vram_pct": vram_pct,
                "global_step": global_step,
            }) + "\n")

        # Early stopping
        if early_stop.step(val_loss):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs. Stopping.")
            break

    # Final checkpoint
    ckpt_mgr.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "config": config,
        },
        val_loss,
        global_step,
    )
    print(f"\n[DONE] Training complete. Best val_loss={early_stop.best:.4f}")
    print(f"[DONE] Best checkpoint: {ckpt_mgr.save_dir / 'best.pth'}")


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        voxel = batch["voxel"].to(device, non_blocking=True)
        target = batch["ghost_label"].to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(voxel)
            loss = F.cross_entropy(logits, target, weight=class_weights, ignore_index=-1)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost-FWL KITTI training")
    parser.add_argument("--config", type=str, default="configs/kitti_train.toml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    print(json.dumps(config, indent=2, default=str))

    train(config, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
