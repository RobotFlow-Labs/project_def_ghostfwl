"""Inference service wiring for Ghost-FWL API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from anima_def_ghostfwl.api.schemas import ClassSummary, PredictResponse
from anima_def_ghostfwl.data.labels import LABEL_ID_TO_NAME
from anima_def_ghostfwl.inference.checkpoint import LoadedPredictor, load_predictor
from anima_def_ghostfwl.inference.postprocess import labels_to_point_cloud
from anima_def_ghostfwl.inference.sliding_window import infer_tiled


@dataclass
class DenoiseService:
    """Manages model lifecycle and runs inference for API requests."""

    checkpoint_path: Path | None = None
    device: str = "cpu"
    _predictor: LoadedPredictor | None = field(default=None, init=False, repr=False)

    @property
    def is_ready(self) -> bool:
        return self._predictor is not None

    def load(self, checkpoint_path: Path | None = None, device: str | None = None) -> None:
        path = checkpoint_path or self.checkpoint_path
        dev = device or self.device
        if path is None:
            raise ValueError("No checkpoint path configured")
        self._predictor = load_predictor(path, device=dev)
        self.checkpoint_path = path
        self.device = dev

    def run(
        self,
        volume: np.ndarray,
        *,
        threshold: float = 0.5,
        output_dir: Path | None = None,
    ) -> PredictResponse:
        if self._predictor is None:
            raise RuntimeError("Service not ready — call load() first")

        window_shape = (
            self._predictor.config.voxel_size[1],
            self._predictor.config.voxel_size[2],
            self._predictor.config.voxel_size[0],
        )
        labels = infer_tiled(
            volume, self._predictor, window_shape=window_shape, threshold=threshold
        )

        points = labels_to_point_cloud(labels)
        ghost_count = int((labels == 3).sum())

        class_counts = {}
        for cls_id, cls_name in LABEL_ID_TO_NAME.items():
            class_counts[cls_name] = int((labels == cls_id).sum())

        output_path = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "denoised_points.npy", points)
            output_path = str(output_dir / "denoised_points.npy")

        return PredictResponse(
            denoised_points_count=len(points),
            ghost_points_removed=ghost_count,
            class_summary=ClassSummary(**class_counts),
            output_path=output_path,
        )
