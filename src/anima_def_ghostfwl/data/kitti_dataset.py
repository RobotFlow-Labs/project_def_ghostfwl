"""KITTI voxel dataset for Ghost-FWL ghost detection training.

Loads pre-cached GPU voxel tensors from disk. Zero disk IO during training
when combined with pinned memory + prefetch.

Ghost labels are synthesized from voxel statistics:
- Isolated low-density voxels near reflective surfaces = ghost candidates
- Multi-return voxels with low reflectance = ghost
- Consistent high-occupancy regions = object
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class KITTIVoxelDataset(Dataset):
    """Dataset that loads pre-voxelized KITTI scans from cache.

    Each sample returns:
        voxel: (2, 256, 256, 32) — occupancy + reflectance channels
        ghost_label: (256, 256, 32) — synthetic ghost labels (0=empty, 1=object, 2=ghost)
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str = "train",
        split_file: str | Path | None = None,
        augment: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.augment = augment

        all_files = sorted(self.cache_dir.glob("*.pt"))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {cache_dir}")

        # Split: 90% train, 5% val, 5% test (deterministic)
        n = len(all_files)
        if split_file and Path(split_file).exists():
            splits = json.loads(Path(split_file).read_text())
            indices = splits[split]
            self.files = [all_files[i] for i in indices if i < n]
        else:
            rng = np.random.RandomState(42)
            perm = rng.permutation(n)
            n_train = int(n * 0.90)
            n_val = int(n * 0.05)
            if split == "train":
                self.files = [all_files[i] for i in perm[:n_train]]
            elif split == "val":
                self.files = [all_files[i] for i in perm[n_train : n_train + n_val]]
            else:
                self.files = [all_files[i] for i in perm[n_train + n_val :]]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        voxel = torch.load(self.files[idx], weights_only=True)  # (2, 256, 256, 32)

        # Generate synthetic ghost labels from voxel statistics
        ghost_label = self._synthesize_ghost_labels(voxel)

        if self.augment:
            voxel, ghost_label = self._augment(voxel, ghost_label)

        return {"voxel": voxel, "ghost_label": ghost_label}

    @staticmethod
    def _synthesize_ghost_labels(voxel: Tensor) -> Tensor:
        """Synthesize ghost labels from occupancy + reflectance heuristics.

        Strategy (approximating real ghost physics):
        - Ghost points: occupied voxels with very low reflectance (<0.1)
          that are spatially isolated (few occupied neighbors)
        - Object points: occupied voxels with moderate-high reflectance
        - Empty: unoccupied voxels
        """
        occ = voxel[0]   # (256, 256, 32)
        refl = voxel[1]  # (256, 256, 32)

        labels = torch.zeros_like(occ, dtype=torch.long)

        # Object: occupied with decent reflectance
        is_occupied = occ > 0
        labels[is_occupied] = 1  # object

        # Ghost heuristic: low reflectance + sparse neighborhood
        low_refl = refl < 0.15
        # Count neighbors using avg pooling proxy
        occ_padded = occ.unsqueeze(0).unsqueeze(0).float()
        kernel_size = 3
        neighbor_count = torch.nn.functional.avg_pool3d(
            occ_padded,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).squeeze() * (kernel_size ** 3)

        sparse = neighbor_count < 3.0  # fewer than 3 neighbors
        ghost_mask = is_occupied & low_refl & sparse
        labels[ghost_mask] = 2  # ghost

        return labels

    @staticmethod
    def _augment(voxel: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        """Random flip augmentation along X and Y axes."""
        if torch.rand(1).item() > 0.5:
            voxel = voxel.flip(1)  # flip X
            label = label.flip(0)
        if torch.rand(1).item() > 0.5:
            voxel = voxel.flip(2)  # flip Y
            label = label.flip(1)
        return voxel, label

    def save_split_indices(self, path: str | Path) -> None:
        """Save the split indices for reproducibility."""
        all_files = sorted(self.cache_dir.glob("*.pt"))
        n = len(all_files)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n).tolist()
        n_train = int(n * 0.90)
        n_val = int(n * 0.05)

        splits = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        Path(path).write_text(json.dumps(splits))
