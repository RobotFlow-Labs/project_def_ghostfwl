#!/usr/bin/env python3
"""Ghost-FWL finetuning scaffold."""

from __future__ import annotations

from anima_def_ghostfwl.training.cli import (
    build_finetune_parser,
    ensure_path_exists,
    maybe_print_config,
)


def main() -> int:
    parser = build_finetune_parser()
    args = parser.parse_args()
    maybe_print_config(args)
    if args.dry_run:
        return 0

    ensure_path_exists(args.dataset_root, label="Dataset root")
    ensure_path_exists(args.checkpoint_dir, label="Checkpoint directory")
    if args.pretrained_checkpoint is not None:
        ensure_path_exists(args.pretrained_checkpoint, label="Pretrained checkpoint")
    raise SystemExit(
        "Finetuning dataset wiring is not implemented yet. "
        "The paper-default CLI and config surface are ready."
    )


if __name__ == "__main__":
    raise SystemExit(main())
