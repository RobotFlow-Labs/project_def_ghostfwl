import torch

from anima_def_ghostfwl.models.fwl_classifier import FrozenEncoderGhostClassifier
from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig, FWLMAEPretrain


def test_classifier_from_pretrain_freezes_encoder_by_default() -> None:
    config = FWLMAEConfig(
        voxel_size=(32, 32, 32),
        patch_size=(32, 16, 16),
        encoder_embed_dim=32,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )
    pretrain = FWLMAEPretrain(config)
    classifier = FrozenEncoderGhostClassifier.from_pretrain(pretrain)

    assert not any(parameter.requires_grad for parameter in classifier.encoder.parameters())
    assert all(parameter.requires_grad for parameter in classifier.head.parameters())


def test_classifier_forward_restores_dense_voxel_logits() -> None:
    config = FWLMAEConfig(
        voxel_size=(32, 32, 32),
        patch_size=(32, 16, 16),
        encoder_embed_dim=32,
        encoder_depth=2,
        encoder_num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )
    classifier = FrozenEncoderGhostClassifier(config=config, freeze_encoder=False)
    voxel = torch.randn(2, 1, 32, 32, 32)

    logits = classifier(voxel)

    assert logits.shape == (2, 4, 32, 32, 32)
