"""Non-overlapping sliding-window inference for full-frame Ghost-FWL volumes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .checkpoint import LoadedPredictor

WindowPredictor = Callable[[np.ndarray], np.ndarray]


def generate_window_positions(
    volume_shape: tuple[int, int, int],
    window_shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Generate non-overlapping XY window starts with full-T crops."""

    height, width, depth = volume_shape
    win_h, win_w, win_t = window_shape
    if win_t != depth:
        raise ValueError("Window depth must match the preprocessed temporal depth")

    positions: list[tuple[int, int, int]] = []
    for start_h in range(0, max(height, 1), win_h):
        for start_w in range(0, max(width, 1), win_w):
            positions.append((start_h, start_w, 0))
    return positions


def extract_window(
    volume: np.ndarray,
    *,
    start: tuple[int, int, int],
    window_shape: tuple[int, int, int],
) -> np.ndarray:
    """Extract a zero-padded window from an HWT volume."""

    start_h, start_w, start_t = start
    win_h, win_w, win_t = window_shape
    end_h = min(start_h + win_h, volume.shape[0])
    end_w = min(start_w + win_w, volume.shape[1])
    end_t = min(start_t + win_t, volume.shape[2])

    window = np.zeros(window_shape, dtype=volume.dtype)
    cropped = volume[start_h:end_h, start_w:end_w, start_t:end_t]
    window[: cropped.shape[0], : cropped.shape[1], : cropped.shape[2]] = cropped
    return window


def infer_tiled(
    volume: np.ndarray,
    predictor: LoadedPredictor | WindowPredictor,
    *,
    window_shape: tuple[int, int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Run tiled inference and merge predictions back into the valid region."""

    if isinstance(predictor, LoadedPredictor):
        def predict_fn(window: np.ndarray) -> np.ndarray:
            return predictor.predict_labels(window, threshold=threshold)
    else:
        predict_fn = predictor

    output = np.zeros(volume.shape, dtype=np.int32)
    for start_h, start_w, start_t in generate_window_positions(volume.shape, window_shape):
        window = extract_window(
            volume,
            start=(start_h, start_w, start_t),
            window_shape=window_shape,
        )
        prediction = predict_fn(window)
        end_h = min(start_h + window_shape[0], volume.shape[0])
        end_w = min(start_w + window_shape[1], volume.shape[1])
        end_t = min(start_t + window_shape[2], volume.shape[2])
        output[start_h:end_h, start_w:end_w, start_t:end_t] = prediction[
            : end_h - start_h,
            : end_w - start_w,
            : end_t - start_t,
        ]
    return output
