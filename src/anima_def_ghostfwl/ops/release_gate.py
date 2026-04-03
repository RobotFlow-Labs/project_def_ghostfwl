"""Release gates — paper-derived metric thresholds for production promotion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReleaseGateConfig:
    """Minimum metric thresholds for production release."""

    min_recall: float = 0.73
    min_ghost_removal_rate: float = 0.90
    max_ghost_fp_rate: float = 0.02
    max_slam_ate: float = 0.30
    max_slam_rte: float = 0.30


@dataclass
class GateResult:
    """Result of a release gate evaluation."""

    passed: bool
    details: dict[str, bool]
    message: str


def evaluate_release_gate(
    metrics: dict[str, float],
    *,
    config: ReleaseGateConfig | None = None,
) -> GateResult:
    """Check training metrics against release thresholds.

    Returns a GateResult indicating whether the model is cleared for
    production deployment.
    """
    cfg = config or ReleaseGateConfig()

    checks = {
        "recall": metrics.get("recall", 0.0) >= cfg.min_recall,
        "ghost_removal_rate": metrics.get("ghost_removal_rate", 0.0) >= cfg.min_ghost_removal_rate,
        "ghost_fp_rate": metrics.get("ghost_fp_rate", 1.0) <= cfg.max_ghost_fp_rate,
        "slam_ate": metrics.get("slam_ate", 999.0) <= cfg.max_slam_ate,
        "slam_rte": metrics.get("slam_rte", 999.0) <= cfg.max_slam_rte,
    }

    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]

    if passed:
        message = "All release gates passed — cleared for production."
    else:
        message = f"Release blocked — failed gates: {', '.join(failed)}"

    return GateResult(passed=passed, details=checks, message=message)
