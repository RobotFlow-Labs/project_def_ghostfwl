# PRD-03: Inference Pipeline

> Module: DEF-GHOSTFWL | Priority: P0
> Depends on: PRD-01, PRD-02
> Status: ✅ Complete

## Objective
Checkpointed Ghost-FWL models can denoise new FWL tensors through the paper’s inference procedure and emit cleaned point clouds plus class volumes.

## Context (from paper)
The paper runs inference with the same `(128,128,256)` input size as training, sequential crops from `(x, y) = (0, 0)`, zero-padding beyond bounds, merge-back, upsampling, and threshold `0.5`.
Paper reference: appendix D.2 "Ghost Detection and Removal", §4.2 "Ghost Detection and Removal".

## Acceptance Criteria
- [x] Inference reproduces paper preprocessing and deterministic sliding-window coverage.
- [x] Predicted classes are assigned only when max probability exceeds `0.5`; otherwise mark as undefined/noise-safe state.
- [x] Ghost-labeled peaks are removable from waveform / point cloud outputs.
- [x] CLI can load a checkpoint and write denoised artifacts plus optional debug volumes.
- [x] Test: `uv run pytest tests/test_inference.py tests/test_postprocess.py tests/test_cli_infer.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_ghostfwl/inference/checkpoint.py` | model loading | appendix D.2 | ~80 |
| `src/anima_def_ghostfwl/inference/sliding_window.py` | tiled inference and merge | appendix D.2 | ~160 |
| `src/anima_def_ghostfwl/inference/postprocess.py` | thresholding and upsample merge | appendix D.2 | ~140 |
| `src/anima_def_ghostfwl/inference/remove_ghosts.py` | ghost point suppression | §4.2 | ~140 |
| `src/anima_def_ghostfwl/cli/infer.py` | CLI entrypoint | §4.2 | ~80 |
| `tests/test_inference.py` | sliding-window tests | — | ~100 |
| `tests/test_postprocess.py` | threshold / merge tests | — | ~90 |
| `tests/test_cli_infer.py` | CLI tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `voxel`: `Tensor[1, 256, 128, 128]`
- `checkpoint_path`: trained finetune weights
- `threshold`: `0.5`

### Outputs
- `class_probs`: `Tensor[256, 128, 128, 4]`
- `class_ids`: `Tensor[256, 128, 128]`
- `denoised_point_cloud`: `ndarray[N, 3|4]`

### Algorithm
```python
# Appendix D.2
def infer_full_frame(raw_voxel):
    voxel = preprocess(raw_voxel)
    windows = sliding_windows(voxel, patch_xy=(128, 128), stride=(128, 128))
    merged = merge_logits([model(win) for win in windows])
    probs = upsample_logits(merged, target_t=700)
    labels = threshold_argmax(probs, threshold=0.5)
    return remove_ghost_points(raw_voxel, labels)
```

## Dependencies
```toml
click = ">=8.1"
rich = ">=14.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| finetuned checkpoint | 1 file | `/Volumes/AIFlowDev/RobotFlowLabs/models/ghost_fwl/fwl_mae_classifier.ckpt` | pending |
| evaluation voxel sample | scene-level `.b2` frames | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/ghost_dataset/scene*/data/` | pending |

## Test Plan
```bash
uv run pytest tests/test_inference.py tests/test_postprocess.py tests/test_cli_infer.py -v
uv run python -m anima_def_ghostfwl.cli.infer --help
```

## References
- Paper: §4.2 "Ghost Detection and Removal"
- Reference impl: `repositories/Ghost-FWL/configs/config_estimate.yaml`
- Reference impl: `repositories/Ghost-FWL/scripts/run_estimate.py`
- Depends on: PRD-01, PRD-02
- Feeds into: PRD-04, PRD-05, PRD-06
