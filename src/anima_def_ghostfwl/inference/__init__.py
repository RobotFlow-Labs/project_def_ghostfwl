"""Inference utilities for Ghost-FWL."""

from .checkpoint import LoadedPredictor, load_predictor
from .postprocess import ghost_mask_from_labels, labels_to_point_cloud, threshold_predictions
from .sliding_window import extract_window, generate_window_positions, infer_tiled

__all__ = [
    "LoadedPredictor",
    "extract_window",
    "generate_window_positions",
    "ghost_mask_from_labels",
    "infer_tiled",
    "labels_to_point_cloud",
    "load_predictor",
    "threshold_predictions",
]
