"""Unified benchmark CLI for Ghost-FWL evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anima_def_ghostfwl.eval.report import BenchmarkResult, generate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ghost-FWL evaluation benchmark")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        print(json.dumps({"thresholds": args.threshold, "status": "dry_run"}, indent=2))
        return 0

    print(f"[EVAL] Predictions: {args.predictions_dir}")
    print(f"[EVAL] Ground truth: {args.ground_truth_dir}")
    print(f"[EVAL] Thresholds: {args.threshold}")

    # MOCK: Real evaluation requires dataset assets
    result = BenchmarkResult(
        recall=0.0,
        ghost_removal_rate=0.0,
        threshold=args.threshold[0],
        metadata={"note": "placeholder — requires real dataset assets"},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = generate_report(result, output_path=args.output_dir / "benchmark_report.md")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
