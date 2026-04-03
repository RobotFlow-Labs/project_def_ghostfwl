"""Tests for Ghost-FWL export pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from anima_def_ghostfwl.export.exporter import (
    ExportManifest,
    export_checkpoint,
    run_export_pipeline,
)
from anima_def_ghostfwl.export.model_card import generate_model_card
from anima_def_ghostfwl.models.fwl_classifier import FrozenEncoderGhostClassifier
from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig


def _small_config() -> FWLMAEConfig:
    return FWLMAEConfig(
        voxel_size=(32, 32, 32),
        patch_size=(32, 16, 16),
        encoder_embed_dim=32,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )


def test_export_checkpoint_creates_file(tmp_path: Path) -> None:
    config = _small_config()
    model = FrozenEncoderGhostClassifier(config=config)
    path = export_checkpoint(model, tmp_path, config=config)
    assert path.exists()
    ckpt = torch.load(path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "config" in ckpt


def test_export_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = ExportManifest(formats={"pth": "model.pth"}, metrics={"recall": 0.75})
    manifest.save(tmp_path / "manifest.json")
    loaded = ExportManifest.load(tmp_path / "manifest.json")
    assert loaded.formats["pth"] == "model.pth"
    assert loaded.metrics["recall"] == 0.75


def test_run_export_pipeline(tmp_path: Path) -> None:
    config = _small_config()
    model = FrozenEncoderGhostClassifier(config=config)
    manifest = run_export_pipeline(model, tmp_path, config=config)
    assert "pth" in manifest.formats
    assert (tmp_path / "export_manifest.json").exists()


def test_model_card_generation() -> None:
    card = generate_model_card(metrics={"recall": 0.75})
    assert "Ghost-FWL" in card
    assert "0.75" in card


def test_model_card_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    generate_model_card(output_path=path)
    assert path.exists()
