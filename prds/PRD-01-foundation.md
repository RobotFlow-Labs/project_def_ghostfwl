# PRD-01: Foundation & Config

> Module: DEF-GHOSTFWL | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
The repository exposes a clean `anima_def_ghostfwl` package with paper-faithful configuration, dataset manifests, and preprocessing utilities that can support both pretraining and supervised training.

## Context (from paper)
Ghost-FWL is a new task built on mobile full-waveform LiDAR data and requires careful preprocessing and peak-level labels.
Paper reference: §3.1 "Sensing System and Data Collection", §3.3 "Annotation", §5.1 "Ghost Denoising Evaluation".

Key requirements from the paper:
- Raw FWL histograms originate at `512 x 400 x 700` with about `1 ns` temporal resolution and `105 m` max range.
- Training data is preprocessed by removing the top and bottom 90 bins, removing the front 25 bins, downsampling T, and cropping to `(128, 128, 256)`.
- Peak annotations use four classes: Object, Glass, Ghost, Noise.

## Acceptance Criteria
- [ ] Package namespace is standardized on `src/anima_def_ghostfwl/` and top-level metadata no longer references `SHINIGAMI`.
- [ ] Settings objects encode dataset roots, split manifests, model shapes, and paper hyperparameters.
- [ ] Data IO can discover voxel `.b2`, annotation `.b2`, and peak `.npy` assets with deterministic split manifests.
- [ ] Preprocessing reproduces paper crops: top `90`, bottom `90`, front `25`, target `(128,128,256)`.
- [ ] Test: `uv run pytest tests/test_settings.py tests/test_data_io.py tests/test_preprocess.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_ghostfwl/__init__.py` | package identity | — | ~10 |
| `src/anima_def_ghostfwl/settings.py` | Pydantic settings / path registry | appendix D.1/D.2 | ~120 |
| `src/anima_def_ghostfwl/data/io.py` | `.b2` / `.npy` discovery and loading | §3.1, repo dataset docs | ~180 |
| `src/anima_def_ghostfwl/data/labels.py` | label maps and constants | §3.3 | ~40 |
| `src/anima_def_ghostfwl/data/splits.py` | scene split manifests | appendix Table 5, D.2 | ~100 |
| `src/anima_def_ghostfwl/data/preprocess.py` | crop/downsample/reorder pipeline | §5.1, appendix D.1/D.2 | ~200 |
| `configs/default.toml` | canonical module config | appendix D.1/D.2 | ~80 |
| `tests/test_settings.py` | settings tests | — | ~60 |
| `tests/test_data_io.py` | discovery tests | — | ~90 |
| `tests/test_preprocess.py` | shape / crop tests | — | ~110 |

## Architecture Detail (from paper)

### Inputs
- `raw_voxel`: `Tensor[X=400, Y=512, T=700]` from `.b2`
- `peak_file`: `ndarray[400*512, 3]` with per-ray peak tuples
- `annotation_voxel`: `Tensor[X=400, Y=512, T=700]`

### Outputs
- `model_voxel`: `Tensor[B, 1, 256, 128, 128]`
- `class_voxel`: `Tensor[B, 256, 128, 128]`
- `split_manifest`: structured list of scene/frame roots

### Algorithm
```python
# Paper §5.1 / Appendix D.1-D.2
class FWLPreprocessor:
    def __call__(self, voxel_xyz_t):
        voxel = crop_y(voxel_xyz_t, top=90, bottom=90)
        voxel = crop_t_front(voxel, front=25)
        voxel = downsample_t(voxel, target_t=256)
        voxel = random_or_deterministic_crop_xy(voxel, size=(128, 128))
        return to_model_layout(voxel)  # -> [1, 256, 128, 128]
```

## Dependencies
```toml
numpy = ">=1.26"
pydantic = ">=2.7"
torch = ">=2.0"
blosc2 = ">=2.7"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| Ghost-FWL supervised scenes | 24,412 frames | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/ghost_dataset/` | pending project-page archive |
| Ghost-FWL mobile pretrain set | 8,933 frames | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/mae_dataset/` | pending project-page archive |

## Test Plan
```bash
uv run pytest tests/test_settings.py tests/test_data_io.py tests/test_preprocess.py -v
uv run ruff check src/ tests/
```

## References
- Paper: §3.1 "Sensing System and Data Collection"
- Paper: §3.3 "Annotation"
- Paper: §5.1 "Ghost Denoising Evaluation"
- Reference impl: `repositories/Ghost-FWL/src/data/dataset_fwl.py`
- Reference impl: `repositories/Ghost-FWL/src/data/dataset_fwl_mae.py`
- Feeds into: PRD-02, PRD-03, PRD-04
