"""Benchmark report generation for Ghost-FWL evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PaperTargets:
    """Paper section 5 target values."""

    recall: float = 0.751
    ghost_removal_rate: float = 0.918
    slam_ate: float = 0.245
    slam_rte: float = 0.245
    ghost_fp_rate: float = 0.0134


@dataclass
class BenchmarkResult:
    """Single evaluation run results."""

    recall: float = 0.0
    ghost_removal_rate: float = 0.0
    ghost_fp_rate: float = 0.0
    slam_ate: float = 0.0
    slam_rte: float = 0.0
    threshold: float = 0.5
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def check_paper_gates(
    result: BenchmarkResult,
    targets: PaperTargets | None = None,
) -> dict[str, bool]:
    """Check each metric against paper-derived gates."""
    t = targets or PaperTargets()
    return {
        "recall": result.recall >= t.recall * 0.97,
        "ghost_removal_rate": result.ghost_removal_rate >= t.ghost_removal_rate * 0.97,
        "ghost_fp_rate": result.ghost_fp_rate <= t.ghost_fp_rate * 1.5,
        "slam_ate": result.slam_ate <= t.slam_ate * 1.25,
        "slam_rte": result.slam_rte <= t.slam_rte * 1.25,
    }


def generate_report(
    result: BenchmarkResult,
    *,
    output_path: str | Path | None = None,
    targets: PaperTargets | None = None,
) -> str:
    """Generate a markdown benchmark report."""
    t = targets or PaperTargets()
    gates = check_paper_gates(result, t)

    lines = [
        "# Ghost-FWL Benchmark Report",
        "",
        "## Denoising Metrics",
        "| Metric | Value | Paper Target | Gate |",
        "|--------|-------|-------------|------|",
        f"| Recall | {result.recall:.4f} | {t.recall} | {'PASS' if gates['recall'] else 'FAIL'} |",
        f"| Ghost Removal Rate | {result.ghost_removal_rate:.4f} | {t.ghost_removal_rate} | "
        f"{'PASS' if gates['ghost_removal_rate'] else 'FAIL'} |",
        f"| Ghost FP Rate | {result.ghost_fp_rate:.4f} | {t.ghost_fp_rate} | "
        f"{'PASS' if gates['ghost_fp_rate'] else 'FAIL'} |",
        "",
        "## SLAM Metrics",
        "| Metric | Value | Paper Target | Gate |",
        "|--------|-------|-------------|------|",
        f"| ATE | {result.slam_ate:.4f} | {t.slam_ate} | "
        f"{'PASS' if gates['slam_ate'] else 'FAIL'} |",
        f"| RTE | {result.slam_rte:.4f} | {t.slam_rte} | "
        f"{'PASS' if gates['slam_rte'] else 'FAIL'} |",
        "",
        f"## Threshold: {result.threshold}",
        "",
        f"## Overall: {'ALL PASS' if all(gates.values()) else 'SOME FAIL'}",
    ]

    if result.per_class_accuracy:
        lines.extend(["", "## Per-Class Accuracy"])
        for cls, acc in result.per_class_accuracy.items():
            lines.append(f"- {cls}: {acc:.4f}")

    report = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)

        json_path = out.with_suffix(".json")
        json_path.write_text(json.dumps(asdict(result), indent=2))

    return report
