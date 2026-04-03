"""Tests for benchmark report generation."""

from pathlib import Path

from anima_def_ghostfwl.eval.report import (
    BenchmarkResult,
    PaperTargets,
    check_paper_gates,
    generate_report,
)


def test_check_paper_gates_all_pass() -> None:
    result = BenchmarkResult(
        recall=0.76,
        ghost_removal_rate=0.92,
        ghost_fp_rate=0.013,
        slam_ate=0.24,
        slam_rte=0.24,
    )
    gates = check_paper_gates(result)
    assert all(gates.values())


def test_check_paper_gates_recall_fail() -> None:
    result = BenchmarkResult(recall=0.5)
    gates = check_paper_gates(result)
    assert gates["recall"] is False


def test_generate_report_contains_metrics() -> None:
    result = BenchmarkResult(
        recall=0.75,
        ghost_removal_rate=0.91,
        ghost_fp_rate=0.015,
        slam_ate=0.25,
        slam_rte=0.25,
    )
    report = generate_report(result)
    assert "Recall" in report
    assert "Ghost Removal Rate" in report
    assert "ATE" in report


def test_generate_report_writes_files(tmp_path: Path) -> None:
    result = BenchmarkResult(recall=0.75, ghost_removal_rate=0.91)
    output = tmp_path / "report.md"
    generate_report(result, output_path=output)
    assert output.exists()
    assert output.with_suffix(".json").exists()


def test_paper_targets_defaults() -> None:
    targets = PaperTargets()
    assert targets.recall == 0.751
    assert targets.ghost_removal_rate == 0.918
