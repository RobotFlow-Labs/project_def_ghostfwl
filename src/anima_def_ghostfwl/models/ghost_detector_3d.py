"""3D convolutional ghost detector for voxelized LiDAR point clouds.

Adapted from Ghost-FWL paper architecture for KITTI-scale voxel grids.
Input: (B, 2, 256, 256, 32) — occupancy + reflectance
Output: (B, 3, 256, 256, 32) — per-voxel logits for empty/object/ghost

Uses depthwise-separable 3D convolutions for efficiency on L4 GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DSConv3d(nn.Module):
    """Depthwise-separable 3D convolution for efficiency."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class EncoderBlock(nn.Module):
    """Encoder block: 2x DSConv3d + residual + downsample."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = DSConv3d(in_ch, out_ch)
        self.conv2 = DSConv3d(out_ch, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.down = nn.MaxPool3d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv1(x)
        h = self.conv2(h) + self.skip(x)
        return self.down(h), h  # downsampled, skip


class DecoderBlock(nn.Module):
    """Decoder block: upsample + concat skip + 2x DSConv3d."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.conv1 = DSConv3d(in_ch + skip_ch, out_ch)
        self.conv2 = DSConv3d(out_ch, out_ch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # Handle size mismatch from pooling
        if x.shape != skip.shape:
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class GhostDetector3D(nn.Module):
    """3D U-Net style ghost detector for voxelized LiDAR.

    Architecture sized for 256x256x32 input on L4 GPU (23GB):
    - 4 encoder stages: 2→32→64→128→256
    - Bottleneck: 256
    - 4 decoder stages: 256→128→64→32
    - Head: 32→3 (empty/object/ghost)

    ~2.1M parameters, ~14-16GB VRAM at batch 8-12.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_ch)        # 256→128
        self.enc2 = EncoderBlock(base_ch, base_ch * 2)        # 128→64
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)    # 64→32
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)    # 32→16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DSConv3d(base_ch * 8, base_ch * 8),
            DSConv3d(base_ch * 8, base_ch * 8),
        )

        # Decoder
        self.dec4 = DecoderBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec3 = DecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = DecoderBlock(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = DecoderBlock(base_ch, base_ch, base_ch)

        # Classification head
        self.head = nn.Conv3d(base_ch, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, 2, 256, 256, 32)

        Returns:
            (B, 3, 256, 256, 32) — logits for empty/object/ghost
        """
        d1, s1 = self.enc1(x)
        d2, s2 = self.enc2(d1)
        d3, s3 = self.enc3(d2)
        d4, s4 = self.enc4(d3)

        b = self.bottleneck(d4)

        u4 = self.dec4(b, s4)
        u3 = self.dec3(u4, s3)
        u2 = self.dec2(u3, s2)
        u1 = self.dec1(u2, s1)

        return self.head(u1)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
