# PRD-02: Core Model

> Module: DEF-GHOSTFWL | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
The repo contains a paper-faithful implementation of FWL-MAE pretraining plus the frozen-encoder ghost-classification model used for denoising.

## Context (from paper)
The paper introduces FWL-MAE to learn FWL representations by reconstructing masked regions while also predicting peak position, amplitude, and width.
Paper reference: §4.1 "Full Waveform LiDAR Masked Autoencoder", §4.2 "Ghost Detection and Removal", appendix Table 7 and D.1/D.2.

Key implementation facts:
- Encoder: 6 transformer blocks, 6 attention heads.
- Decoder: 6 transformer blocks, 6 attention heads.
- Mask ratio: `70%`.
- Input tensor: `(H, W, T) = (128, 128, 256)`.
- Patch size: `(16, 16, 256)`.
- Embeddings: `Dencoder=768`, `Ddecoder=384`.
- Peak head predicts `K=4` peaks with `lambda_p=1.0`, `lambda_a=1.0`, `lambda_w=0.5`.
- Finetuning freezes the pretrained encoder and applies a lightweight 2-layer classifier head with focal loss.

## Acceptance Criteria
- [ ] FWL-MAE pretrain module reproduces the paper architecture and loss decomposition.
- [ ] Classifier module consumes the frozen encoder outputs and emits 4-class logits for every `(x, y, t)` location.
- [ ] Training loops exist for both pretraining and finetuning with AdamW and paper hyperparameters.
- [ ] Losses cover `LMSE`, `Lpeak-p`, `Lpeak-a`, `Lpeak-w`, and focal loss with the paper alpha/gamma values.
- [ ] Test: `uv run pytest tests/test_fwl_mae.py tests/test_classifier.py tests/test_losses.py tests/test_training_config.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_ghostfwl/models/patch_embed.py` | 3D patch embedding + PE | appendix Table 7 | ~120 |
| `src/anima_def_ghostfwl/models/fwl_mae_pretrain.py` | MAE encoder/decoder | §4.1 | ~260 |
| `src/anima_def_ghostfwl/models/peak_heads.py` | peak attribute heads | §4.1 | ~80 |
| `src/anima_def_ghostfwl/models/fwl_classifier.py` | frozen encoder + classifier head | §4.2 | ~180 |
| `src/anima_def_ghostfwl/models/losses.py` | MAE + focal losses | Eq. (1), Eq. (6) | ~140 |
| `src/anima_def_ghostfwl/training/pretrain.py` | pretrain loop | appendix D.1 | ~220 |
| `src/anima_def_ghostfwl/training/finetune.py` | finetune loop | appendix D.2 | ~220 |
| `scripts/train_pretrain.py` | CLI | appendix D.1 | ~40 |
| `scripts/train_finetune.py` | CLI | appendix D.2 | ~40 |
| `tests/test_fwl_mae.py` | model-shape tests | — | ~120 |
| `tests/test_classifier.py` | classifier tests | — | ~100 |
| `tests/test_losses.py` | loss tests | — | ~100 |
| `tests/test_training_config.py` | hyperparameter tests | — | ~60 |

## Architecture Detail (from paper)

### Inputs
- `voxel`: `Tensor[B, 1, 256, 128, 128]`
- `mask`: `BoolTensor[B, Npatch]`
- `peak_targets`: `Tensor[B, Npatch, K, 3]` for `(position, amplitude, width)`

### Outputs
- `mae_reconstruction`: `Tensor[B, Nmasked, 16*16*256]`
- `peak_predictions`: `Tensor[B, Npatch, K, 3]`
- `class_logits`: `Tensor[B, 4, 256, 128, 128]`

### Algorithm
```python
# Paper §4.1 / Appendix D.1
class FWLMAEPretrain(nn.Module):
    def forward(self, voxel, mask):
        patches = self.patch_embed(voxel)          # [B, Npatch, 768]
        encoded = self.encoder(select_unmasked(patches, mask))
        decoded = self.decoder(restore_mask_tokens(encoded, mask))
        peaks = self.peak_head(encoded_or_decoded)
        return {"reconstruction": decoded, "peaks": peaks}

class FrozenEncoderGhostClassifier(nn.Module):
    def forward(self, voxel):
        with torch.no_grad():
            features = self.encoder(voxel)
        logits = self.classifier_head(features)
        return logits
```

## Dependencies
```toml
timm = ">=1.0"
einops = ">=0.8"
scikit-learn = ">=1.5"
wandb = ">=0.19"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| unlabeled mobile frames | 8,933 frames | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/mae_dataset/` | pending |
| supervised annotated frames | 18,274 train+val+test total | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/ghost_dataset/` | pending |

## Test Plan
```bash
uv run pytest tests/test_fwl_mae.py tests/test_classifier.py tests/test_losses.py tests/test_training_config.py -v
uv run python scripts/train_pretrain.py --help
uv run python scripts/train_finetune.py --help
```

## References
- Paper: §4.1 "Full Waveform LiDAR Masked Autoencoder"
- Paper: §4.2 "Ghost Detection and Removal"
- Paper: Eq. (1), Eq. (6)
- Reference impl: `repositories/Ghost-FWL/src/models/FWLMAE_pretrain.py`
- Reference impl: `repositories/Ghost-FWL/src/models/FWLMAE.py`
- Reference impl: `repositories/Ghost-FWL/src/training/fwl_mae_pretrain.py`
- Reference impl: `repositories/Ghost-FWL/src/training/fwl_mae_finetune.py`
- Depends on: PRD-01
- Feeds into: PRD-03, PRD-04
