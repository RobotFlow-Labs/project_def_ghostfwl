"""ANIMA AnimaNode subclass for Ghost-FWL serving.

This is the standard ANIMA serving entry point that wraps the Ghost-FWL
inference pipeline into the AnimaNode lifecycle (setup_inference + process).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from anima_def_ghostfwl.inference.checkpoint import LoadedPredictor, load_predictor
from anima_def_ghostfwl.inference.postprocess import labels_to_point_cloud
from anima_def_ghostfwl.inference.sliding_window import infer_tiled
from anima_def_ghostfwl.version import __version__


class GhostFWLServeNode:
    """AnimaNode-compatible serving wrapper for Ghost-FWL.

    Follows the ANIMA 3-layer Docker pattern:
      - setup_inference(): load weights from HF or local path
      - process(): run ghost detection on input volume
      - get_status(): report node health
    """

    def __init__(self) -> None:
        self._predictor: LoadedPredictor | None = None
        self._device: str = os.environ.get("ANIMA_DEVICE", "cpu")
        self._threshold: float = float(os.environ.get("ANIMA_THRESHOLD", "0.5"))
        self._frames_processed: int = 0

    def setup_inference(self) -> None:
        """Load model weights from configured path."""
        checkpoint_path = os.environ.get("ANIMA_CHECKPOINT_PATH", "")
        weight_dir = os.environ.get("ANIMA_WEIGHT_DIR", "/data/weights")

        # Try explicit checkpoint path first, then weight dir
        candidates = [
            Path(checkpoint_path) if checkpoint_path else None,
            Path(weight_dir) / "best.pth",
            Path(weight_dir) / "model.safetensors",
        ]

        for path in candidates:
            if path is not None and path.is_file():
                self._predictor = load_predictor(path, device=self._device)
                return

        raise FileNotFoundError(
            f"No checkpoint found. Checked: {[str(p) for p in candidates]}"
        )

    def process(self, input_data: np.ndarray) -> dict[str, Any]:
        """Run ghost detection on an input volume."""
        if self._predictor is None:
            raise RuntimeError("Model not loaded — call setup_inference() first")

        window_shape = (
            self._predictor.config.voxel_size[1],
            self._predictor.config.voxel_size[2],
            self._predictor.config.voxel_size[0],
        )
        labels = infer_tiled(
            input_data,
            self._predictor,
            window_shape=window_shape,
            threshold=self._threshold,
        )
        points = labels_to_point_cloud(labels)
        ghost_count = int((labels == 3).sum())

        self._frames_processed += 1

        return {
            "denoised_points": points,
            "ghost_count": ghost_count,
            "total_voxels": int(labels.size),
            "surviving_points": len(points),
        }

    def get_status(self) -> dict[str, Any]:
        """Module-specific status fields for health endpoint."""
        return {
            "model_loaded": self._predictor is not None,
            "device": self._device,
            "threshold": self._threshold,
            "version": __version__,
            "frames_processed": self._frames_processed,
        }
