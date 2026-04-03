"""Tests for Ghost-FWL evaluation metrics."""

import numpy as np

from anima_def_ghostfwl.eval.metrics import (
    ghost_false_positive_rate,
    ghost_removal_rate,
    peak_recall,
    per_class_accuracy,
    slam_ate,
    slam_rte,
)


def test_peak_recall_perfect() -> None:
    gt = np.array([0, 1, 2, 3, 3, 3])
    pred = np.array([0, 1, 2, 3, 3, 3])
    assert peak_recall(pred, gt) == 1.0


def test_peak_recall_no_ghosts() -> None:
    gt = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([0, 1, 2, 0, 1, 2])
    assert peak_recall(pred, gt) == 1.0


def test_peak_recall_partial() -> None:
    gt = np.array([3, 3, 3, 3])
    pred = np.array([3, 3, 0, 0])
    assert peak_recall(pred, gt) == 0.5


def test_ghost_removal_rate_all_removed() -> None:
    pred_points = np.array([[10.0, 10.0, 10.0]])
    gt_ghost = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    rate = ghost_removal_rate(pred_points, gt_ghost, radius=0.5)
    assert rate == 1.0


def test_ghost_removal_rate_none_removed() -> None:
    pred_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gt_ghost = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    rate = ghost_removal_rate(pred_points, gt_ghost, radius=0.5)
    assert rate == 0.0


def test_ghost_removal_rate_empty_ghost() -> None:
    pred_points = np.array([[0.0, 0.0, 0.0]])
    gt_ghost = np.empty((0, 3))
    assert ghost_removal_rate(pred_points, gt_ghost) == 1.0


def test_per_class_accuracy_perfect() -> None:
    gt = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 2, 3])
    acc = per_class_accuracy(pred, gt)
    assert all(v == 1.0 for v in acc.values())


def test_ghost_fp_rate_zero() -> None:
    gt = np.array([1, 1, 1, 3])
    pred = np.array([1, 1, 1, 3])
    assert ghost_false_positive_rate(pred, gt) == 0.0


def test_ghost_fp_rate_nonzero() -> None:
    gt = np.array([1, 1, 1, 1])
    pred = np.array([3, 1, 1, 1])
    assert ghost_false_positive_rate(pred, gt) == 0.25


def test_slam_ate_identical() -> None:
    traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert slam_ate(traj, traj) == 0.0


def test_slam_ate_offset() -> None:
    est = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert abs(slam_ate(est, gt) - 1.0) < 1e-6


def test_slam_rte_identical() -> None:
    traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert slam_rte(traj, traj) == 0.0
