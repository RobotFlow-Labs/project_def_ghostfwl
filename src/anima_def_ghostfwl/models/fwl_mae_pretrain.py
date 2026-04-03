"""Paper-shaped FWL-MAE pretraining backbone for Ghost-FWL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .patch_embed import (
    PatchGridSpec,
    VoxelPatchEmbed,
    build_3d_sincos_pos_embed,
    build_patch_mask,
)


@dataclass(frozen=True)
class FWLMAEConfig:
    """Paper-default architecture values for the Ghost-FWL MAE."""

    voxel_size: tuple[int, int, int] = (256, 128, 128)
    patch_size: tuple[int, int, int] = (256, 16, 16)
    in_chans: int = 1
    encoder_embed_dim: int = 768
    encoder_depth: int = 6
    encoder_num_heads: int = 6
    decoder_embed_dim: int = 384
    decoder_depth: int = 6
    decoder_num_heads: int = 6
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.70
    max_peaks: int = 4
    histogram_bins: int = 256
    dropout: float = 0.0

    @property
    def patch_spec(self) -> PatchGridSpec:
        return PatchGridSpec(voxel_size=self.voxel_size, patch_size=self.patch_size)


class TransformerBlock(nn.Module):
    """A compact transformer block with batch-first semantics."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class GhostFWLEncoder(nn.Module):
    """Patch encoder used by both pretraining and finetuning models."""

    def __init__(self, config: FWLMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VoxelPatchEmbed(
            voxel_size=config.voxel_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.encoder_embed_dim,
        )
        pos_embed = build_3d_sincos_pos_embed(self.patch_embed.spec, config.encoder_embed_dim)
        self.register_buffer("pos_embed", pos_embed, persistent=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.encoder_embed_dim,
                    num_heads=config.encoder_num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(config.encoder_embed_dim)

    @property
    def patch_spec(self) -> PatchGridSpec:
        return self.patch_embed.spec

    def patch_tokens(self, voxel: Tensor) -> Tensor:
        tokens = self.patch_embed(voxel)
        return tokens + self.pos_embed.to(device=voxel.device, dtype=tokens.dtype)

    def forward_full(self, voxel: Tensor) -> Tensor:
        tokens = self.patch_tokens(voxel)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_visible(self, voxel: Tensor, mask: Tensor) -> Tensor:
        tokens = self.patch_tokens(voxel)
        batch, _, channels = tokens.shape
        visible = tokens[~mask].reshape(batch, -1, channels)
        for block in self.blocks:
            visible = block(visible)
        return self.norm(visible)


class GhostFWLDecoder(nn.Module):
    """Masked-patch decoder for waveform reconstruction."""

    def __init__(self, config: FWLMAEConfig, patch_spec: PatchGridSpec) -> None:
        super().__init__()
        self.config = config
        self.patch_spec = patch_spec
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.decoder_embed_dim,
                    num_heads=config.decoder_num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(config.decoder_embed_dim)
        self.head = nn.Linear(config.decoder_embed_dim, patch_spec.patch_volume)

    def forward(self, x: Tensor, num_masked_tokens: int) -> Tensor:
        for block in self.blocks:
            x = block(x)
        decoded = self.head(self.norm(x))
        if num_masked_tokens == 0:
            return decoded[:, :0]
        return decoded[:, -num_masked_tokens:]


class PeakPositionHead(nn.Module):
    def __init__(self, embed_dim: int, *, max_peaks: int, histogram_bins: int) -> None:
        super().__init__()
        self.max_peaks = max_peaks
        self.histogram_bins = histogram_bins
        self.head = nn.Linear(embed_dim, max_peaks)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.head(x)) * float(self.histogram_bins - 1)


class PositivePeakHead(nn.Module):
    def __init__(self, embed_dim: int, *, max_peaks: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, max_peaks)
        self.activation = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.head(x))


class FWLMAEPretrain(nn.Module):
    """Joint waveform reconstruction + peak regression model."""

    def __init__(self, config: FWLMAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or FWLMAEConfig()
        self.patch_spec = self.config.patch_spec
        self.encoder = GhostFWLEncoder(self.config)
        self.encoder_to_decoder = nn.Linear(
            self.config.encoder_embed_dim,
            self.config.decoder_embed_dim,
            bias=False,
        )
        decoder_pos_embed = build_3d_sincos_pos_embed(
            self.patch_spec,
            self.config.decoder_embed_dim,
        )
        self.register_buffer("decoder_pos_embed", decoder_pos_embed, persistent=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.decoder_embed_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.decoder = GhostFWLDecoder(self.config, self.patch_spec)
        self.peak_position_head = PeakPositionHead(
            self.config.decoder_embed_dim,
            max_peaks=self.config.max_peaks,
            histogram_bins=self.config.histogram_bins,
        )
        self.peak_width_head = PositivePeakHead(
            self.config.decoder_embed_dim,
            max_peaks=self.config.max_peaks,
        )
        self.peak_height_head = PositivePeakHead(
            self.config.decoder_embed_dim,
            max_peaks=self.config.max_peaks,
        )

    def build_mask(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return build_patch_mask(
            batch_size=batch_size,
            num_patches=self.patch_spec.num_patches,
            mask_ratio=self.config.mask_ratio,
            device=device,
            generator=generator,
        )

    def forward(self, voxel: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        batch_size = voxel.shape[0]
        if mask is None:
            mask = self.build_mask(batch_size, device=voxel.device)

        visible_tokens = self.encoder.forward_visible(voxel, mask)
        visible_tokens = self.encoder_to_decoder(visible_tokens)

        decoder_pos = self.decoder_pos_embed.to(device=voxel.device, dtype=visible_tokens.dtype)
        decoder_pos = decoder_pos.expand(batch_size, -1, -1)
        channels = visible_tokens.shape[-1]
        visible_pos = decoder_pos[~mask].reshape(batch_size, -1, channels)
        masked_pos = decoder_pos[mask].reshape(batch_size, -1, channels)

        visible_with_pos = visible_tokens + visible_pos
        masked_with_pos = self.mask_token.to(dtype=visible_tokens.dtype) + masked_pos
        decoder_sequence = torch.cat((visible_with_pos, masked_with_pos), dim=1)
        num_masked = masked_with_pos.shape[1]

        aligned_sequence = torch.empty(
            (batch_size, self.patch_spec.num_patches, channels),
            device=voxel.device,
            dtype=visible_tokens.dtype,
        )
        aligned_sequence[~mask] = visible_with_pos.reshape(-1, channels)
        aligned_sequence[mask] = masked_with_pos.reshape(-1, channels)

        return {
            "mask": mask,
            "reconstruction": self.decoder(decoder_sequence, num_masked),
            "peak_positions": self.peak_position_head(aligned_sequence),
            "peak_widths": self.peak_width_head(aligned_sequence),
            "peak_heights": self.peak_height_head(aligned_sequence),
        }

    def reconstruct_voxel(
        self,
        reconstruction: Tensor,
        mask: Tensor,
        *,
        channels: int = 1,
    ) -> Tensor:
        """Place masked patch reconstructions back into voxel space."""

        batch_size = reconstruction.shape[0]
        depth, height, width = self.patch_spec.voxel_size
        patch_d, patch_h, patch_w = self.patch_spec.patch_size
        grid_d, grid_h, grid_w = self.patch_spec.grid_shape
        volume = reconstruction.new_zeros((batch_size, channels, depth, height, width))

        masked_indices = mask.nonzero(as_tuple=False)
        for flat_idx, (batch_idx, patch_idx) in enumerate(masked_indices):
            d_idx = int(patch_idx) // (grid_h * grid_w)
            spatial_idx = int(patch_idx) % (grid_h * grid_w)
            h_idx = spatial_idx // grid_w
            w_idx = spatial_idx % grid_w
            patch = reconstruction[batch_idx, flat_idx % reconstruction.shape[1]].reshape(
                patch_d,
                patch_h,
                patch_w,
            )
            d_start, h_start, w_start = d_idx * patch_d, h_idx * patch_h, w_idx * patch_w
            volume[
                batch_idx,
                0,
                d_start : d_start + patch_d,
                h_start : h_start + patch_h,
                w_start : w_start + patch_w,
            ] = patch
        return volume
