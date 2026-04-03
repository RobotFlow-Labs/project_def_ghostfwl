import torch

from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig, FWLMAEPretrain
from anima_def_ghostfwl.models.patch_embed import (
    PatchGridSpec,
    VoxelPatchEmbed,
    build_patch_mask,
)


def test_paper_patch_geometry_matches_appendix() -> None:
    spec = PatchGridSpec(voxel_size=(256, 128, 128), patch_size=(256, 16, 16))
    assert spec.grid_shape == (1, 8, 8)
    assert spec.num_patches == 64
    assert spec.patch_volume == 256 * 16 * 16


def test_patch_embedding_respects_patch_count() -> None:
    embed = VoxelPatchEmbed(
        voxel_size=(32, 32, 32),
        patch_size=(32, 16, 16),
        embed_dim=24,
    )
    voxel = torch.randn(2, 1, 32, 32, 32)
    tokens = embed(voxel)
    assert tokens.shape == (2, 4, 24)


def test_build_patch_mask_uses_fixed_mask_count() -> None:
    mask = build_patch_mask(batch_size=3, num_patches=10, mask_ratio=0.7)
    assert mask.dtype == torch.bool
    assert mask.shape == (3, 10)
    assert torch.all(mask.sum(dim=1) == 7)


def test_fwl_mae_forward_emits_reconstruction_and_peak_heads() -> None:
    config = FWLMAEConfig(
        voxel_size=(32, 32, 32),
        patch_size=(32, 16, 16),
        encoder_embed_dim=32,
        encoder_depth=2,
        encoder_num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=2,
        decoder_num_heads=4,
        max_peaks=3,
        histogram_bins=32,
    )
    model = FWLMAEPretrain(config)
    voxel = torch.randn(2, 1, 32, 32, 32)
    outputs = model(voxel)

    assert outputs["mask"].shape == (2, 4)
    assert outputs["peak_positions"].shape == (2, 4, 3)
    assert outputs["peak_widths"].shape == (2, 4, 3)
    assert outputs["peak_heights"].shape == (2, 4, 3)
    assert outputs["reconstruction"].shape == (2, 3, 32 * 16 * 16)


def test_reconstruct_voxel_places_masked_patches_back() -> None:
    config = FWLMAEConfig(
        voxel_size=(32, 32, 32),
        patch_size=(32, 16, 16),
        encoder_embed_dim=16,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )
    model = FWLMAEPretrain(config)
    mask = torch.tensor([[True, False, True, False]])
    reconstruction = torch.ones(1, 2, 32 * 16 * 16)

    volume = model.reconstruct_voxel(reconstruction, mask)

    assert volume.shape == (1, 1, 32, 32, 32)
    assert torch.count_nonzero(volume) > 0
