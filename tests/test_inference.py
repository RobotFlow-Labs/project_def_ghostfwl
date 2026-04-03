import numpy as np

from anima_def_ghostfwl.inference.checkpoint import load_predictor, save_checkpoint
from anima_def_ghostfwl.inference.sliding_window import (
    extract_window,
    generate_window_positions,
    infer_tiled,
)
from anima_def_ghostfwl.models.fwl_classifier import FrozenEncoderGhostClassifier
from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig


def _tiny_config() -> FWLMAEConfig:
    return FWLMAEConfig(
        voxel_size=(8, 8, 8),
        patch_size=(8, 4, 4),
        encoder_embed_dim=16,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )


def test_load_predictor_is_deterministic(tmp_path) -> None:
    config = _tiny_config()
    model = FrozenEncoderGhostClassifier(config=config, freeze_encoder=False)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, config=config)

    predictor = load_predictor(checkpoint)
    window = np.random.rand(8, 8, 8).astype(np.float32)

    first = predictor.predict_labels(window)
    second = predictor.predict_labels(window)
    assert np.array_equal(first, second)


def test_generate_window_positions_are_non_overlapping() -> None:
    positions = generate_window_positions((20, 20, 8), (12, 12, 8))
    assert positions == [(0, 0, 0), (0, 12, 0), (12, 0, 0), (12, 12, 0)]


def test_extract_window_zero_pads_outside_valid_range() -> None:
    volume = np.ones((10, 10, 8), dtype=np.float32)
    window = extract_window(volume, start=(8, 8, 0), window_shape=(12, 12, 8))
    assert window.shape == (12, 12, 8)
    assert np.all(window[:2, :2] == 1.0)
    assert np.all(window[2:, 2:] == 0.0)


def test_infer_tiled_merges_valid_region_only() -> None:
    volume = np.zeros((20, 20, 8), dtype=np.float32)

    def predictor(window: np.ndarray) -> np.ndarray:
        return np.full(window.shape, 3, dtype=np.int32)

    labels = infer_tiled(volume, predictor, window_shape=(12, 12, 8), threshold=0.5)
    assert labels.shape == volume.shape
    assert np.all(labels == 3)
