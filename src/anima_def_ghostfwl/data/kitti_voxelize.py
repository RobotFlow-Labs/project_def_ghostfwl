"""GPU-accelerated KITTI LiDAR voxelization pipeline.

Converts raw KITTI velodyne .bin point clouds into 3D occupancy voxel grids
on GPU, caches as .pt tensors. Shared cache at:
  /mnt/forge-data/shared_infra/datasets/kitti_voxel_cache/

Voxel grid: 256x256x32 covering [-51.2, 51.2] x [-51.2, 51.2] x [-5, 3] meters
Resolution: 0.4m x 0.4m x 0.25m (matches common LiDAR detection configs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# KITTI LiDAR voxel grid configuration
VOXEL_CONFIG = {
    "x_range": (-51.2, 51.2),
    "y_range": (-51.2, 51.2),
    "z_range": (-5.0, 3.0),
    "voxel_size": (0.4, 0.4, 0.25),
    "grid_shape": (256, 256, 32),  # (X, Y, Z)
}


def load_kitti_velodyne(bin_path: str | Path) -> np.ndarray:
    """Load a KITTI velodyne .bin file as (N, 4) float32 [x, y, z, reflectance]."""
    return np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)


@torch.no_grad()
def voxelize_pointcloud_gpu(
    points: Tensor,
    *,
    x_range: tuple[float, float] = VOXEL_CONFIG["x_range"],
    y_range: tuple[float, float] = VOXEL_CONFIG["y_range"],
    z_range: tuple[float, float] = VOXEL_CONFIG["z_range"],
    voxel_size: tuple[float, float, float] = VOXEL_CONFIG["voxel_size"],
    grid_shape: tuple[int, int, int] = VOXEL_CONFIG["grid_shape"],
) -> Tensor:
    """Voxelize a point cloud entirely on GPU using scatter.

    Args:
        points: (N, 4) float tensor [x, y, z, reflectance] on GPU

    Returns:
        (2, Gx, Gy, Gz) float tensor — channel 0: occupancy, channel 1: mean reflectance
    """
    device = points.device
    xyz = points[:, :3]
    refl = points[:, 3]

    # Filter to bounds
    mask = (
        (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] < x_range[1])
        & (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] < y_range[1])
        & (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] < z_range[1])
    )
    xyz = xyz[mask]
    refl = refl[mask]

    if len(xyz) == 0:
        return torch.zeros(2, *grid_shape, device=device)

    # Compute voxel indices
    vx = ((xyz[:, 0] - x_range[0]) / voxel_size[0]).long().clamp(0, grid_shape[0] - 1)
    vy = ((xyz[:, 1] - y_range[0]) / voxel_size[1]).long().clamp(0, grid_shape[1] - 1)
    vz = ((xyz[:, 2] - z_range[0]) / voxel_size[2]).long().clamp(0, grid_shape[2] - 1)

    # Flat index for scatter
    flat_idx = vx * (grid_shape[1] * grid_shape[2]) + vy * grid_shape[2] + vz
    flat_size = grid_shape[0] * grid_shape[1] * grid_shape[2]

    # Occupancy: count points per voxel
    counts = torch.zeros(flat_size, device=device)
    counts.scatter_add_(0, flat_idx, torch.ones_like(refl))

    # Reflectance sum for mean
    refl_sum = torch.zeros(flat_size, device=device)
    refl_sum.scatter_add_(0, flat_idx, refl)

    # Normalize: occupancy (binary) and mean reflectance
    occupancy = (counts > 0).float().reshape(grid_shape)
    mean_refl = torch.where(counts > 0, refl_sum / counts, torch.zeros_like(counts))
    mean_refl = mean_refl.reshape(grid_shape)

    return torch.stack([occupancy, mean_refl], dim=0)  # (2, Gx, Gy, Gz)


def cache_kitti_voxels(
    velodyne_dir: str | Path,
    cache_dir: str | Path,
    *,
    device: str = "cuda",
    max_files: int | None = None,
) -> int:
    """Batch-voxelize all KITTI velodyne scans and save as .pt cache.

    Returns number of files cached.
    """
    velodyne_dir = Path(velodyne_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    if max_files:
        bin_files = bin_files[:max_files]

    dev = torch.device(device)
    cached = 0

    for i, bin_path in enumerate(bin_files):
        stem = bin_path.stem
        cache_path = cache_dir / f"{stem}.pt"

        if cache_path.exists():
            cached += 1
            continue

        points_np = load_kitti_velodyne(bin_path)
        points_gpu = torch.from_numpy(points_np).to(dev)
        voxel = voxelize_pointcloud_gpu(points_gpu)

        # Save as CPU tensor for portability
        torch.save(voxel.cpu(), cache_path)
        cached += 1

        if (i + 1) % 500 == 0:
            print(f"[VOXELIZE] {i + 1}/{len(bin_files)} cached")

    print(f"[VOXELIZE] Done: {cached} files in {cache_dir}")
    return cached


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="KITTI GPU voxelization cache builder")
    parser.add_argument(
        "--velodyne-dir",
        type=str,
        default="/mnt/forge-data/datasets/kitti/training/velodyne",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/forge-data/shared_infra/datasets/kitti_voxel_cache",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    n = cache_kitti_voxels(
        args.velodyne_dir,
        args.cache_dir,
        device=args.device,
        max_files=args.max_files,
    )
    elapsed = time.perf_counter() - t0
    print(f"[VOXELIZE] {n} files in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.1f} files/sec)")
