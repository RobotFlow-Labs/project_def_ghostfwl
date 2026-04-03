"""Frozen-encoder Ghost-FWL finetune classifier."""

from __future__ import annotations

import copy

from torch import Tensor, nn

from .fwl_mae_pretrain import FWLMAEConfig, FWLMAEPretrain, GhostFWLEncoder
from .patch_embed import PatchGridSpec, reshape_patch_logits_to_volume


class FrozenEncoderGhostClassifier(nn.Module):
    """Patch-based dense classifier built on top of the MAE encoder."""

    def __init__(
        self,
        config: FWLMAEConfig | None = None,
        *,
        num_classes: int = 4,
        encoder: GhostFWLEncoder | None = None,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or FWLMAEConfig()
        self.num_classes = num_classes
        self.patch_spec: PatchGridSpec = self.config.patch_spec
        self.encoder = encoder or GhostFWLEncoder(self.config)
        hidden_dim = max(self.config.encoder_embed_dim // 2, 32)
        self.head = nn.Sequential(
            nn.Linear(self.config.encoder_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes * self.patch_spec.patch_volume),
        )
        self.set_encoder_frozen(freeze_encoder)

    @classmethod
    def from_pretrain(
        cls,
        pretrain_model: FWLMAEPretrain,
        *,
        num_classes: int = 4,
        freeze_encoder: bool = True,
    ) -> FrozenEncoderGhostClassifier:
        return cls(
            config=pretrain_model.config,
            num_classes=num_classes,
            encoder=copy.deepcopy(pretrain_model.encoder),
            freeze_encoder=freeze_encoder,
        )

    def set_encoder_frozen(self, frozen: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen

    def forward(self, voxel: Tensor) -> Tensor:
        encoded = self.encoder.forward_full(voxel)
        patch_logits = self.head(encoded)
        return reshape_patch_logits_to_volume(
            patch_logits,
            spec=self.patch_spec,
            num_classes=self.num_classes,
        )
