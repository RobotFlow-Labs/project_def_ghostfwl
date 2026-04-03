# DEF-GHOSTFWL: Ghost-FWL: LiDAR Ghost Object Detection — Implementation PRD
## ANIMA Wave-7 Module #12

**Status:** Planning complete, implementation not started
**Version:** 0.2
**Date:** 2026-04-03
**Paper:** Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal
**Paper Link:** https://arxiv.org/abs/2603.28224
**Repo:** https://github.com/Keio-CSG/Ghost-FWL
**Compute:** CUDA preferred, MLX acceptable for later inference adaptation
**Functional Name:** DEF-ghostfwl
**Stack:** Defense

## Build Plan — Executable PRDs

> Total PRDs: 7 | Tasks: 23 | Status: 0/23 complete

| # | PRD | Title | Priority | Tasks | Status |
|---|---|---|---|---|---|
| 1 | [PRD-01](prds/PRD-01-foundation.md) | Foundation & Config | P0 | 4 | ⬜ |
| 2 | [PRD-02](prds/PRD-02-core-model.md) | Core Model | P0 | 4 | ⬜ |
| 3 | [PRD-03](prds/PRD-03-inference.md) | Inference Pipeline | P0 | 3 | ⬜ |
| 4 | [PRD-04](prds/PRD-04-evaluation.md) | Evaluation | P1 | 3 | ⬜ |
| 5 | [PRD-05](prds/PRD-05-api-docker.md) | API & Docker | P1 | 3 | ⬜ |
| 6 | [PRD-06](prds/PRD-06-ros2.md) | ROS2 Integration | P1 | 3 | ⬜ |
| 7 | [PRD-07](prds/PRD-07-production.md) | Production | P2 | 3 | ⬜ |

## 1. Executive Summary
Ghost-FWL addresses a real failure mode in robotics and autonomous driving: ghost points created by multi-path LiDAR reflections through glass and reflective surfaces. The paper contributes both the first large-scale mobile full-waveform LiDAR dataset for this task and a paper baseline built around a transformer classifier with FWL-MAE self-supervised pretraining. For ANIMA, the correct strategy is to reproduce the paper pipeline first, including waveform preprocessing, split policy, frozen-encoder classification, and the denoising metrics that enable downstream SLAM and object-detection gains. Only after paper-faithful reproduction should we adapt the module for our own LiDAR stack and deployment surfaces.

## 2. Paper Verification Status
- [x] ArXiv ID verified
- [x] GitHub repo confirmed accessible and vendored locally in `repositories/Ghost-FWL/`
- [x] Paper read completely, including appendix implementation details
- [ ] Reference repo executed end-to-end on real data
- [ ] Datasets confirmed downloadable from a stable archive URL
- [ ] Metrics reproduced locally
- [x] No obvious red flags in method framing
- **Verdict:** VALID PAPER, but asset release is incomplete

## 3. What We Take From The Paper
- The exact FWL framing: raw full-waveform histograms plus peak-level labels for `object`, `glass`, `ghost`, and `noise`.
- The pretraining method FWL-MAE with masked spatial patches and peak attribute prediction `(position, amplitude, width)`.
- The supervised denoising model built from a frozen pretrained encoder and a lightweight classifier head.
- The preprocessing recipe: remove top/bottom `90` bins, remove front `25` bins, downsample T, crop to `(128,128,256)`.
- The split policy from appendix D.2: train/val on scenes `001,003,004,005,006,008,010`, test on `002,007,009`.
- The quantitative targets: Recall `0.751`, Ghost Removal Rate `0.918`, SLAM ATE/RTE `0.245`, Ghost FP `1.34%`.

## 4. What We Skip
- Synthetic-data generation; the paper explicitly motivates real waveform capture instead.
- Immediate MLX training parity; the paper was trained on a single RTX 6000 Ada and we need a CUDA reproduction first.
- Any paper extensions not required for baseline reproduction, such as temporal sequence modeling beyond the described static / mobile split.
- Direct recreation of the authors’ full visualization toolchain unless it is required for evaluation.

## 5. What We Adapt
- We will normalize the module namespace to `DEF-GHOSTFWL`; the current scaffold still refers to `SHINIGAMI`.
- We will convert YAML-heavy reference configs into ANIMA-style TOML + typed settings while preserving the same values.
- We will add container, API, and ROS2 surfaces around the paper inference path without changing denoising semantics.
- We will add release-gating logic tied to the paper metrics so productionization stays evidence-based.

## 6. Architecture

### Paper-Faithful Core
1. Load raw voxel grids from `.b2` files shaped `(400,512,700)` and optional peak `.npy` metadata.
2. Apply crop/downsample preprocessing to produce `(128,128,256)` waveform volumes.
3. Reorder into model tensor layout `Tensor[B,1,256,128,128]`.
4. Pretrain an FWL-MAE encoder with:
   - 6 encoder blocks / 6 heads
   - 6 decoder blocks / 6 heads
   - `Dencoder=768`, `Ddecoder=384`
   - mask ratio `0.70`
   - peak head predicting `K=4` peaks
5. Freeze the encoder and train a 4-class classifier head with focal loss.
6. Run deterministic sliding-window inference and remove ghost-labeled points.

### ANIMA Module Surfaces
- Library package under `src/anima_def_ghostfwl/`
- CLI for train / infer / evaluate
- FastAPI service for denoising requests
- ROS2 node for waveform-to-point-cloud filtering
- Export / release bundle with benchmark report

## 7. Implementation Phases

### Phase 1 — Foundation + Paper Fidelity ⬜
- [ ] Replace stale `SHINIGAMI` namespace with `DEF-GHOSTFWL`
- [ ] Encode dataset roots, split manifests, and preprocessing constants
- [ ] Implement waveform IO and preprocessing tests

### Phase 2 — Reproduce Core Model ⬜
- [ ] Implement FWL-MAE pretraining
- [ ] Implement frozen-encoder classifier
- [ ] Match paper architecture and hyperparameters

### Phase 3 — Inference + Evaluation ⬜
- [ ] Implement sliding-window inference and ghost removal
- [ ] Reproduce Recall and Ghost Removal Rate
- [ ] Add SLAM and object-detection benchmark harnesses

### Phase 4 — ANIMA Deployment ⬜
- [ ] Add FastAPI + Docker
- [ ] Add ROS2 node
- [ ] Add export, release gates, and observability

## 8. Datasets
| Dataset | Size | URL | Phase Needed |
|---|---|---|---|
| Ghost-FWL supervised set | 24,412 annotated frames | https://keio-csg.github.io/Ghost-FWL/ | Phase 1-4 |
| Ghost-FWL mobile pretrain set | 8,933 unlabeled frames | https://keio-csg.github.io/Ghost-FWL/ | Phase 2 |
| SLAM downstream capture | 231 frames / 23.4 m trajectory | local downstream fixture to assemble | Phase 3 |

## 9. Dependencies on Other Wave Projects
| Needs output from | What it provides |
|---|---|
| None required | This module is self-contained once assets are available |

## 10. Success Criteria
- Paper preprocessing is reproduced exactly.
- Ghost denoising Recall reaches at least `0.73`, target `0.751`.
- Ghost Removal Rate reaches at least `0.90`, target `0.918`.
- SLAM ATE and RTE are each at or below `0.30 m`, target `0.245`.
- Ghost FP rate for object detection is at or below `2.0%`, target `1.34%`.
- The module serves the same inference path through CLI, API, and ROS2.

## 11. Risk Assessment
- The dataset direct download URL is missing from the vendored docs, so implementation may stall on asset acquisition.
- The reference repo and paper contain axis-order ambiguities between `(H,W,T)`, `(X,Y,Z)`, and model tensor order; tests must lock this down early.
- The current scaffold has naming drift (`SHINIGAMI`) that could leak into package names, paths, or deployment artifacts if not fixed first.
- CUDA reproduction is likely required before any MLX adaptation is credible.

## 12. Planning Artifacts
- [ASSETS.md](ASSETS.md)
- [PIPELINE_MAP.md](PIPELINE_MAP.md)
- [PRD Suite](prds/README.md)
- [Task Index](tasks/INDEX.md)

## 13. Shenzhen Demo
- **Demo-ready target**: paper-faithful inference on held-out Ghost-FWL scenes plus a ROS2 or API demo surface
- **Demo plan**:
  1. finish PRD-01 to PRD-03
  2. secure one checkpoint and one representative glass-heavy test scene
  3. show before/after ghost removal with point cloud overlay
