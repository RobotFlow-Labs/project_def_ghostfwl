"""Postprocessing helpers for ghost removal outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def threshold_predictions(probabilities: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Apply a confidence threshold to class probabilities."""

    labels = probabilities.argmax(axis=-1).astype(np.int32)
    max_prob = probabilities.max(axis=-1)
    labels[max_prob < threshold] = 0
    return labels


def ghost_mask_from_labels(labels: np.ndarray, *, ghost_label: int = 3) -> np.ndarray:
    return labels == ghost_label


def labels_to_point_cloud(labels: np.ndarray, *, ghost_label: int = 3) -> np.ndarray:
    """Convert dense label voxels into surviving point coordinates `[N, 3]`."""

    keep_mask = labels != ghost_label
    return np.argwhere(keep_mask).astype(np.int32)


def write_point_cloud_artifact(path: str | Path, points: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, points)
    return target
