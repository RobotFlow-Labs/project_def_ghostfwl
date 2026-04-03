import torch

from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig, FWLMAEPretrain
from anima_def_ghostfwl.models.losses import (
    PAPER_FOCAL_ALPHA_BY_LABEL,
    PAPER_FOCAL_ALPHA_CLASS_ORDER,
    FWLMAELoss,
    focal_loss,
)


def test_paper_focal_alpha_mapping_is_explicit() -> None:
    assert PAPER_FOCAL_ALPHA_CLASS_ORDER == (0.25, 0.7, 0.05, 0.0001)
    assert PAPER_FOCAL_ALPHA_BY_LABEL == (0.0001, 0.25, 0.05, 0.7)


def test_focal_loss_handles_ignore_index() -> None:
    logits = torch.randn(2, 4, 8, 8, 8)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    targets[:, 0] = -100
    loss = focal_loss(logits, targets)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_fwl_mae_loss_matches_model_outputs() -> None:
    config = FWLMAEConfig(
        voxel_size=(32, 32, 32),
        patch_size=(32, 16, 16),
        encoder_embed_dim=32,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        histogram_bins=32,
        max_peaks=2,
    )
    model = FWLMAEPretrain(config)
    voxel = torch.randn(2, 1, 32, 32, 32)
    predictions = model(voxel)
    targets = {
        "peak_positions": torch.rand(2, 2, 32, 32) * 31.0,
        "peak_heights": torch.rand(2, 2, 32, 32),
        "peak_widths": torch.rand(2, 2, 32, 32),
    }
    loss_fn = FWLMAELoss(patch_spec=config.patch_spec)
    total_loss, components = loss_fn(predictions, targets, input_volume=voxel)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(components["position_loss"])
    assert torch.isfinite(components["reconstruction_loss"])
