"""Losses for Ghost-FWL pretraining and finetuning."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anima_def_ghostfwl.data.labels import LABEL_ID_TO_NAME

from .patch_embed import PatchGridSpec, patchify_volume

PAPER_FOCAL_ALPHA_CLASS_ORDER: tuple[float, float, float, float] = (0.25, 0.7, 0.05, 0.0001)
PAPER_FOCAL_ALPHA_BY_LABEL: tuple[float, ...] = tuple(
    {
        "noise": 0.0001,
        "object": 0.25,
        "glass": 0.05,
        "ghost": 0.7,
    }[LABEL_ID_TO_NAME[label_id]]
    for label_id in sorted(LABEL_ID_TO_NAME)
)


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float | Sequence[float] | Tensor = PAPER_FOCAL_ALPHA_BY_LABEL,
    gamma: float = 2.0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Compute multiclass focal loss over dense logits."""

    ce_loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=ignore_index)
    valid_mask = targets != ignore_index
    safe_targets = targets.masked_fill(~valid_mask, 0)
    pt = torch.exp(-ce_loss)

    if isinstance(alpha, (float, int)):
        alpha_t = logits.new_full(targets.shape, float(alpha))
    else:
        alpha_tensor = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device)
        alpha_t = alpha_tensor[safe_targets]
    alpha_t = alpha_t.masked_fill(~valid_mask, 0.0)

    loss = alpha_t * (1.0 - pt) ** gamma * ce_loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    denom = valid_mask.sum().clamp_min(1)
    return loss.sum() / denom


class PaperFocalLoss(nn.Module):
    """Reference focal loss wrapper with the paper class weights."""

    def __init__(
        self,
        *,
        alpha: Sequence[float] | Tensor = PAPER_FOCAL_ALPHA_BY_LABEL,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_index=self.ignore_index,
        )


def patchify_peak_targets(peak_tensor: Tensor, *, spec: PatchGridSpec) -> Tensor:
    """Aggregate `[B, K, H, W]` peak targets into patch space `[B, N, K]`."""

    batch, peaks, height, width = peak_tensor.shape
    _, patch_h, patch_w = spec.patch_size
    _, grid_h, grid_w = spec.grid_shape
    if (height, width) != (spec.voxel_size[1], spec.voxel_size[2]):
        raise ValueError(
            "Expected peak target size "
            f"{(spec.voxel_size[1], spec.voxel_size[2])}, got {(height, width)}"
        )
    return (
        peak_tensor.reshape(batch, peaks, grid_h, patch_h, grid_w, patch_w)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, spec.num_patches, peaks, patch_h * patch_w)
        .mean(dim=-1)
    )


class FWLMAELoss(nn.Module):
    """Joint loss for masked reconstruction and patch-level peak regression."""

    def __init__(
        self,
        *,
        patch_spec: PatchGridSpec,
        position_weight: float = 1.0,
        height_weight: float = 1.0,
        width_weight: float = 0.5,
        reconstruction_weight: float = 1.0,
        position_loss: str = "l1",
        height_loss: str = "l1",
        width_loss: str = "l1",
        reconstruction_loss: str = "mse",
    ) -> None:
        super().__init__()
        self.patch_spec = patch_spec
        self.position_weight = position_weight
        self.height_weight = height_weight
        self.width_weight = width_weight
        self.reconstruction_weight = reconstruction_weight
        self.position_loss_fn = self._make_loss(position_loss)
        self.height_loss_fn = self._make_loss(height_loss)
        self.width_loss_fn = self._make_loss(width_loss)
        self.reconstruction_loss_fn = self._make_loss(reconstruction_loss)

    @staticmethod
    def _make_loss(name: str) -> nn.Module:
        normalized = name.lower()
        if normalized == "mse":
            return nn.MSELoss(reduction="none")
        if normalized == "l1":
            return nn.L1Loss(reduction="none")
        if normalized == "smooth_l1":
            return nn.SmoothL1Loss(reduction="none")
        raise ValueError(f"Unsupported loss type: {name}")

    def forward(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
        *,
        input_volume: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        mask = predictions.get("mask") if mask is None else mask
        if mask is None:
            raise ValueError("Mask must be provided explicitly or via predictions['mask']")

        target_patches = patchify_volume(input_volume, self.patch_spec)
        reconstruction_target = target_patches[mask]
        reconstruction_pred = predictions["reconstruction"].reshape(-1, target_patches.shape[-1])
        reconstruction_loss = self.reconstruction_loss_fn(
            reconstruction_pred,
            reconstruction_target.reshape(-1, target_patches.shape[-1]),
        ).mean()

        peak_pos_target = patchify_peak_targets(targets["peak_positions"], spec=self.patch_spec)
        peak_hgt_target = patchify_peak_targets(targets["peak_heights"], spec=self.patch_spec)
        peak_wid_target = patchify_peak_targets(targets["peak_widths"], spec=self.patch_spec)

        position_loss = self.position_loss_fn(
            predictions["peak_positions"][mask],
            peak_pos_target[mask],
        ).mean()
        height_loss = self.height_loss_fn(
            predictions["peak_heights"][mask],
            peak_hgt_target[mask],
        ).mean()
        width_loss = self.width_loss_fn(
            predictions["peak_widths"][mask],
            peak_wid_target[mask],
        ).mean()

        total = (
            self.position_weight * position_loss
            + self.height_weight * height_loss
            + self.width_weight * width_loss
            + self.reconstruction_weight * reconstruction_loss
        )
        components = {
            "position_loss": position_loss,
            "height_loss": height_loss,
            "width_loss": width_loss,
            "reconstruction_loss": reconstruction_loss,
            "total_loss": total,
        }
        return total, components
