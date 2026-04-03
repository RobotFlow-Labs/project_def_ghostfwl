"""Paper-faithful preprocessing for Ghost-FWL voxel volumes.

Reference: paper section 5.1, appendix D.1/D.2.
Pipeline: raw (400,512,700) -> crop Y -> crop front -> downsample T -> crop XY -> tensor (1,256,128,128).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def crop_y_axis(
    voxel: np.ndarray, *, top: int = 90, bottom: int = 90
) -> np.ndarray:
    """Remove top and bottom bins from the Y (second) axis."""
    return voxel[:, top : voxel.shape[1] - bottom, :]


def crop_histogram_front(voxel: np.ndarray, *, front: int = 25) -> np.ndarray:
    """Remove the first `front` bins from the histogram (third) axis."""
    return voxel[:, :, front:]


def downsample_histogram_axis(voxel: np.ndarray, target_t: int) -> np.ndarray:
    """Linearly interpolate the histogram axis to `target_t` bins."""
    source_t = voxel.shape[2]
    if source_t == target_t:
        return voxel
    source_indices = np.linspace(0, source_t - 1, target_t)
    lower = np.floor(source_indices).astype(int)
    upper = np.minimum(lower + 1, source_t - 1)
    fraction = (source_indices - lower).astype(np.float32)
    return (
        voxel[:, :, lower] * (1.0 - fraction[np.newaxis, np.newaxis, :])
        + voxel[:, :, upper] * fraction[np.newaxis, np.newaxis, :]
    )


def crop_xy_patch(
    voxel: np.ndarray,
    *,
    target_x: int = 128,
    target_y: int = 128,
    start_x: int = 0,
    start_y: int = 0,
) -> np.ndarray:
    """Extract a fixed-size XY crop from the volume."""
    return voxel[start_x : start_x + target_x, start_y : start_y + target_y, :]


class FWLPreprocessor:
    """Full preprocessing chain matching the paper defaults."""

    def __init__(
        self,
        *,
        crop_top: int = 90,
        crop_bottom: int = 90,
        crop_front: int = 25,
        target_t: int = 256,
        target_x: int = 128,
        target_y: int = 128,
    ) -> None:
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_front = crop_front
        self.target_t = target_t
        self.target_x = target_x
        self.target_y = target_y

    def __call__(
        self, voxel: np.ndarray, *, start_x: int = 0, start_y: int = 0
    ) -> Tensor:
        v = crop_y_axis(voxel, top=self.crop_top, bottom=self.crop_bottom)
        v = crop_histogram_front(v, front=self.crop_front)
        v = downsample_histogram_axis(v, self.target_t)
        v = crop_xy_patch(
            v,
            target_x=self.target_x,
            target_y=self.target_y,
            start_x=start_x,
            start_y=start_y,
        )
        # Output shape: (H, W, T) -> model expects (C, T, H, W)
        tensor = torch.from_numpy(v).float().permute(2, 0, 1).unsqueeze(0)
        return tensor
