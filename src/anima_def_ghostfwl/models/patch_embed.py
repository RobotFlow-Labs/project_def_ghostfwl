"""Patch embedding and patch-grid helpers for Ghost-FWL volumes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PatchGridSpec:
    """Patch geometry derived from voxel and patch sizes."""

    voxel_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(v % p != 0 for v, p in zip(self.voxel_size, self.patch_size, strict=True)):
            raise ValueError(
                f"Voxel size {self.voxel_size} must be divisible by patch size {self.patch_size}"
            )

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(v // p for v, p in zip(self.voxel_size, self.patch_size, strict=True))

    @property
    def num_patches(self) -> int:
        depth, height, width = self.grid_shape
        return depth * height * width

    @property
    def patch_volume(self) -> int:
        patch_d, patch_h, patch_w = self.patch_size
        return patch_d * patch_h * patch_w


def _sincos_1d(values: Tensor, dim: int) -> Tensor:
    if dim <= 0:
        return values.new_zeros((values.numel(), 0))
    usable_dim = dim - (dim % 2)
    if usable_dim == 0:
        return values.new_zeros((values.numel(), dim))
    omega = torch.arange(usable_dim // 2, device=values.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(usable_dim // 2, 1)))
    angles = values.reshape(-1, 1).float() * omega.reshape(1, -1)
    emb = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat((emb, values.new_zeros((values.numel(), dim - emb.shape[1]))), dim=1)
    return emb


def build_3d_sincos_pos_embed(spec: PatchGridSpec, embed_dim: int) -> Tensor:
    """Build a deterministic 3D sin/cos positional embedding table."""

    grid_d, grid_h, grid_w = spec.grid_shape
    z, y, x = torch.meshgrid(
        torch.arange(grid_d, dtype=torch.float32),
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack((z.reshape(-1), y.reshape(-1), x.reshape(-1)), dim=1)

    base_dim = (embed_dim // 3) // 2 * 2
    dims = [base_dim, base_dim, base_dim]
    dims[-1] += embed_dim - sum(dims)

    embeddings = [
        _sincos_1d(coords[:, axis], dims[axis]) for axis in range(3)
    ]
    return torch.cat(embeddings, dim=1).unsqueeze(0)


def build_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Create a per-sample boolean patch mask with a fixed masked count."""

    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"Mask ratio must be within [0, 1], got {mask_ratio}")

    num_masked = int(round(num_patches * mask_ratio))
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
    for batch_idx in range(batch_size):
        order = torch.randperm(num_patches, device=device, generator=generator)
        mask[batch_idx, order[:num_masked]] = True
    return mask


def patchify_volume(volume: Tensor, spec: PatchGridSpec) -> Tensor:
    """Convert `[B, C, D, H, W]` voxels into flattened patch tokens."""

    batch, channels, depth, height, width = volume.shape
    if (depth, height, width) != spec.voxel_size:
        raise ValueError(
            f"Expected voxel size {spec.voxel_size}, got {(depth, height, width)}"
        )

    grid_d, grid_h, grid_w = spec.grid_shape
    patch_d, patch_h, patch_w = spec.patch_size
    return (
        volume.reshape(batch, channels, grid_d, patch_d, grid_h, patch_h, grid_w, patch_w)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(batch, spec.num_patches, channels * spec.patch_volume)
    )


def reshape_patch_logits_to_volume(
    patch_logits: Tensor,
    *,
    spec: PatchGridSpec,
    num_classes: int,
) -> Tensor:
    """Invert flattened patch predictions back to dense voxel logits."""

    batch, num_patches, patch_output = patch_logits.shape
    if num_patches != spec.num_patches:
        raise ValueError(f"Expected {spec.num_patches} patches, got {num_patches}")
    expected_output = num_classes * spec.patch_volume
    if patch_output != expected_output:
        raise ValueError(f"Expected patch output {expected_output}, got {patch_output}")

    grid_d, grid_h, grid_w = spec.grid_shape
    patch_d, patch_h, patch_w = spec.patch_size
    depth, height, width = spec.voxel_size
    return (
        patch_logits.reshape(
            batch,
            grid_d,
            grid_h,
            grid_w,
            num_classes,
            patch_d,
            patch_h,
            patch_w,
        )
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(batch, num_classes, depth, height, width)
    )


class VoxelPatchEmbed(nn.Module):
    """3D convolutional patch embedding for full-waveform voxel histograms."""

    def __init__(
        self,
        voxel_size: tuple[int, int, int] = (256, 128, 128),
        patch_size: tuple[int, int, int] = (256, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.spec = PatchGridSpec(voxel_size=voxel_size, patch_size=patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @property
    def num_patches(self) -> int:
        return self.spec.num_patches

    @property
    def patch_volume(self) -> int:
        return self.spec.patch_volume

    def forward(self, x: Tensor) -> Tensor:
        batch, _, depth, height, width = x.shape
        if (depth, height, width) != self.spec.voxel_size:
            raise ValueError(
                f"Expected voxel size {self.spec.voxel_size}, got {(depth, height, width)}"
            )
        return self.proj(x).flatten(2).transpose(1, 2)
