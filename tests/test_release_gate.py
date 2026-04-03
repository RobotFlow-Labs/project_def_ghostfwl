"""Tests for release gate evaluation."""

from __future__ import annotations

from anima_def_ghostfwl.ops.observability import (
    CheckpointFingerprint,
    DegradationMonitor,
    InferenceEvent,
)
from anima_def_ghostfwl.ops.release_gate import (
    ReleaseGateConfig,
    evaluate_release_gate,
)


def test_release_gate_all_pass() -> None:
    metrics = {
        "recall": 0.76,
        "ghost_removal_rate": 0.92,
        "ghost_fp_rate": 0.013,
        "slam_ate": 0.24,
        "slam_rte": 0.24,
    }
    result = evaluate_release_gate(metrics)
    assert result.passed is True
    assert all(result.details.values())


def test_release_gate_recall_fail() -> None:
    metrics = {"recall": 0.5, "ghost_removal_rate": 0.95}
    result = evaluate_release_gate(metrics)
    assert result.passed is False
    assert result.details["recall"] is False


def test_release_gate_custom_config() -> None:
    cfg = ReleaseGateConfig(min_recall=0.90)
    metrics = {"recall": 0.85}
    result = evaluate_release_gate(metrics, config=cfg)
    assert result.passed is False


def test_release_gate_empty_metrics_fail() -> None:
    result = evaluate_release_gate({})
    assert result.passed is False


def test_inference_event_json() -> None:
    event = InferenceEvent(frame_id="test", latency_ms=12.5)
    j = event.to_json()
    assert "test" in j
    assert "12.5" in j


def test_checkpoint_fingerprint_missing_file() -> None:
    fp = CheckpointFingerprint.from_file("/tmp/nonexistent_ckpt.pth")
    assert fp.sha256 == "MISSING"


def test_degradation_monitor_normal() -> None:
    mon = DegradationMonitor(window_size=10)
    for _ in range(5):
        mon.record(ghost_count=10, total_count=1000)
    assert mon.mean_ghost_rate < 0.05
    assert mon.check_degradation() is False


def test_degradation_monitor_alert() -> None:
    mon = DegradationMonitor(window_size=5)
    for _ in range(5):
        mon.record(ghost_count=600, total_count=1000)
    assert mon.check_degradation(max_ghost_rate=0.5) is True
