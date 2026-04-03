"""Paper-faithful denoising metrics from Ghost-FWL section 5.1 and appendix E.1."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from anima_def_ghostfwl.data.labels import LABEL_NAME_TO_ID


def peak_recall(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    *,
    ghost_label: int = LABEL_NAME_TO_ID["ghost"],
) -> float:
    """Fraction of true ghost peaks correctly identified.

    Paper section 5.1: recall = TP / (TP + FN) where positives are ghost peaks.
    """
    gt_ghost = ground_truth_labels == ghost_label
    total_ghost = gt_ghost.sum()
    if total_ghost == 0:
        return 1.0
    pred_ghost = predicted_labels == ghost_label
    true_positives = (pred_ghost & gt_ghost).sum()
    return float(true_positives / total_ghost)


def ghost_removal_rate(
    pred_points: np.ndarray,
    gt_ghost_points: np.ndarray,
    *,
    radius: float = 0.001,
) -> float:
    """Fraction of ground-truth ghost points removed from the denoised cloud.

    Paper section 5.1, appendix E.1: for each GT ghost point, check if any
    predicted point survives within `radius`. If not, the ghost was removed.
    """
    if len(gt_ghost_points) == 0:
        return 1.0
    if len(pred_points) == 0:
        return 1.0

    tree = KDTree(pred_points)
    distances, _ = tree.query(gt_ghost_points, k=1)
    removed = (distances > radius).sum()
    return float(removed / len(gt_ghost_points))


def per_class_accuracy(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    *,
    num_classes: int = 4,
    ignore_label: int = -1,
) -> dict[str, float]:
    """Per-class accuracy breakdown for all label categories."""
    from anima_def_ghostfwl.data.labels import LABEL_ID_TO_NAME

    results: dict[str, float] = {}
    for cls_id in range(num_classes):
        name = LABEL_ID_TO_NAME.get(cls_id, f"class_{cls_id}")
        mask = (ground_truth_labels == cls_id) & (ground_truth_labels != ignore_label)
        total = mask.sum()
        if total == 0:
            results[name] = 0.0
            continue
        correct = ((predicted_labels == cls_id) & mask).sum()
        results[name] = float(correct / total)
    return results


def ghost_false_positive_rate(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    *,
    ghost_label: int = LABEL_NAME_TO_ID["ghost"],
    object_label: int = LABEL_NAME_TO_ID["object"],
) -> float:
    """Fraction of real object peaks incorrectly classified as ghost.

    Paper section 5.2.2: ghost FP rate among object detections.
    """
    gt_object = ground_truth_labels == object_label
    total_object = gt_object.sum()
    if total_object == 0:
        return 0.0
    false_ghost = ((predicted_labels == ghost_label) & gt_object).sum()
    return float(false_ghost / total_object)


def slam_ate(
    estimated_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
) -> float:
    """Absolute Trajectory Error — mean L2 distance between aligned poses.

    Paper section 5.2.1: ATE after Umeyama alignment.
    """
    if len(estimated_trajectory) != len(ground_truth_trajectory):
        raise ValueError(
            f"Trajectory lengths differ: {len(estimated_trajectory)} vs "
            f"{len(ground_truth_trajectory)}"
        )
    errors = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, axis=1)
    return float(errors.mean())


def slam_rte(
    estimated_trajectory: np.ndarray,
    ground_truth_trajectory: np.ndarray,
) -> float:
    """Relative Trajectory Error — mean L2 of consecutive displacement differences.

    Paper section 5.2.1.
    """
    if len(estimated_trajectory) != len(ground_truth_trajectory):
        raise ValueError(
            f"Trajectory lengths differ: {len(estimated_trajectory)} vs "
            f"{len(ground_truth_trajectory)}"
        )
    est_delta = np.diff(estimated_trajectory, axis=0)
    gt_delta = np.diff(ground_truth_trajectory, axis=0)
    errors = np.linalg.norm(est_delta - gt_delta, axis=1)
    return float(errors.mean())
