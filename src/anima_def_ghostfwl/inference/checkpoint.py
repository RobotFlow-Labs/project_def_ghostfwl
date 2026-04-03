"""Checkpoint loading and single-window prediction wrappers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from anima_def_ghostfwl.models.fwl_classifier import FrozenEncoderGhostClassifier
from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig


def _coerce_config(payload: Any) -> FWLMAEConfig:
    if isinstance(payload, FWLMAEConfig):
        return payload
    if isinstance(payload, dict):
        return FWLMAEConfig(**payload)
    return FWLMAEConfig()


@dataclass
class LoadedPredictor:
    """Loaded classifier plus metadata needed by inference code."""

    model: FrozenEncoderGhostClassifier
    config: FWLMAEConfig
    device: torch.device

    def _window_to_tensor(self, window: np.ndarray) -> Tensor:
        expected_shape = (
            self.config.voxel_size[1],
            self.config.voxel_size[2],
            self.config.voxel_size[0],
        )
        if window.shape != expected_shape:
            raise ValueError(f"Expected window shape {expected_shape}, got {window.shape}")
        tensor = torch.from_numpy(window).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict_probabilities(self, window: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._window_to_tensor(window))
            probabilities = torch.softmax(logits, dim=1).squeeze(0).permute(2, 3, 1, 0)
        return probabilities.cpu().numpy()

    def predict_labels(self, window: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_probabilities(window)
        max_prob = probabilities.max(axis=-1)
        labels = probabilities.argmax(axis=-1).astype(np.int32)
        labels[max_prob < threshold] = 0
        return labels


def load_predictor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedPredictor:
    """Load a frozen-encoder classifier checkpoint deterministically."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    config = _coerce_config(checkpoint.get("config"))
    model = FrozenEncoderGhostClassifier(config=config, freeze_encoder=False)
    model.load_state_dict(state_dict, strict=True)
    predictor = LoadedPredictor(
        model=model.to(device),
        config=config,
        device=torch.device(device),
    )
    return predictor


def save_checkpoint(
    checkpoint_path: str | Path,
    model: FrozenEncoderGhostClassifier,
    *,
    config: FWLMAEConfig,
) -> None:
    """Small helper used by tests and local smoke runs."""

    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        Path(checkpoint_path),
    )
