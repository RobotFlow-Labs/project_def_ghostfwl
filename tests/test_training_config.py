from anima_def_ghostfwl.training.cli import build_finetune_parser, build_pretrain_parser


def test_pretrain_parser_matches_paper_defaults() -> None:
    args = build_pretrain_parser().parse_args([])
    assert args.epochs == 100
    assert args.batch_size == 32
    assert args.lr == 1e-3
    assert args.weight_decay == 1e-2
    assert args.mask_ratio == 0.70
    assert args.patch_size == [256, 16, 16]
    assert args.voxel_size == [256, 128, 128]


def test_finetune_parser_defaults_to_frozen_encoder() -> None:
    args = build_finetune_parser().parse_args([])
    assert args.epochs == 100
    assert args.freeze_encoder is True
    assert args.num_classes == 4
