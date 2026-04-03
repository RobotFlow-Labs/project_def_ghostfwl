"""Shared CLI builders for Ghost-FWL training entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig
from anima_def_ghostfwl.settings import GhostFWLSettings


def _common_parser(description: str) -> argparse.ArgumentParser:
    settings = GhostFWLSettings()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset_root)
    parser.add_argument("--checkpoint-dir", type=Path, default=settings.checkpoint_root)
    parser.add_argument("--device", default=settings.backend)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def build_pretrain_parser() -> argparse.ArgumentParser:
    config = FWLMAEConfig()
    parser = _common_parser("Ghost-FWL paper-faithful MAE pretraining entrypoint")
    parser.add_argument("--pretrain-root", type=Path, default=GhostFWLSettings().pretrain_root)
    parser.add_argument("--mask-ratio", type=float, default=config.mask_ratio)
    parser.add_argument("--encoder-embed-dim", type=int, default=config.encoder_embed_dim)
    parser.add_argument("--encoder-depth", type=int, default=config.encoder_depth)
    parser.add_argument("--encoder-heads", type=int, default=config.encoder_num_heads)
    parser.add_argument("--decoder-embed-dim", type=int, default=config.decoder_embed_dim)
    parser.add_argument("--decoder-depth", type=int, default=config.decoder_depth)
    parser.add_argument("--decoder-heads", type=int, default=config.decoder_num_heads)
    parser.add_argument("--patch-size", nargs=3, type=int, default=list(config.patch_size))
    parser.add_argument("--voxel-size", nargs=3, type=int, default=list(config.voxel_size))
    parser.add_argument("--max-peaks", type=int, default=config.max_peaks)
    return parser


def build_finetune_parser() -> argparse.ArgumentParser:
    config = FWLMAEConfig()
    parser = _common_parser("Ghost-FWL paper-faithful finetuning entrypoint")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--encoder-embed-dim", type=int, default=config.encoder_embed_dim)
    parser.add_argument("--encoder-depth", type=int, default=config.encoder_depth)
    parser.add_argument("--encoder-heads", type=int, default=config.encoder_num_heads)
    parser.add_argument("--patch-size", nargs=3, type=int, default=list(config.patch_size))
    parser.add_argument("--voxel-size", nargs=3, type=int, default=list(config.voxel_size))
    return parser


def namespace_to_dict(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def maybe_print_config(args: argparse.Namespace) -> None:
    if args.print_config or args.dry_run:
        print(json.dumps(namespace_to_dict(args), indent=2, sort_keys=True))


def ensure_path_exists(path: Path, *, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
