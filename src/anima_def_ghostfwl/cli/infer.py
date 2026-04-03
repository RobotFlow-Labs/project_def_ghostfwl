"""CLI entrypoint for Ghost-FWL tiled inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anima_def_ghostfwl.inference.checkpoint import load_predictor
from anima_def_ghostfwl.inference.postprocess import (
    ghost_mask_from_labels,
    labels_to_point_cloud,
    write_point_cloud_artifact,
)
from anima_def_ghostfwl.inference.sliding_window import infer_tiled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ghost-FWL tiled inference CLI")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    predictor = load_predictor(args.checkpoint, device=args.device)
    if args.dry_run:
        return 0

    volume = np.load(args.input_npy)
    window_shape = (
        predictor.config.voxel_size[1],
        predictor.config.voxel_size[2],
        predictor.config.voxel_size[0],
    )
    labels = infer_tiled(volume, predictor, window_shape=window_shape, threshold=args.threshold)
    ghost_mask = ghost_mask_from_labels(labels)
    point_cloud = labels_to_point_cloud(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "predictions.npy", labels)
    np.save(args.output_dir / "ghost_mask.npy", ghost_mask)
    write_point_cloud_artifact(args.output_dir / "denoised_points.npy", point_cloud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
