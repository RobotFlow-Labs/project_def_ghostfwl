from pathlib import Path

from anima_def_ghostfwl.settings import get_settings


def test_default_settings_match_paper_defaults() -> None:
    settings = get_settings()

    assert settings.codename == "DEF-GHOSTFWL"
    assert settings.project_name == "anima-def-ghostfwl"
    assert settings.raw_shape_xyz == (400, 512, 700)
    assert settings.model_shape_hwt == (128, 128, 256)
    assert settings.crop_top_bins == 90
    assert settings.crop_bottom_bins == 90
    assert settings.crop_front_bins == 25
    assert settings.pretrain_mask_ratio == 0.70


def test_settings_allow_override_paths() -> None:
    settings = get_settings(
        data_root=Path("/tmp/data"),
        dataset_root=Path("/tmp/data/ghost_dataset"),
        pretrain_root=Path("/tmp/data/mae_dataset"),
    )

    assert settings.data_root == Path("/tmp/data")
    assert settings.dataset_root.name == "ghost_dataset"
    assert settings.pretrain_root.name == "mae_dataset"
