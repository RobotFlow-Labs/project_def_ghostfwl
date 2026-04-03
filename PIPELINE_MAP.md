# DEF-GHOSTFWL — Pipeline Map

## Paper Pipeline → Code / PRD Mapping

| Paper Component | Paper Section | Planned File(s) | PRD |
|---|---|---|---|
| raw FWL ingestion from voxel `.b2` | §3.1, repo `docs/README_dataset.md` | `src/anima_def_ghostfwl/data/io.py` | PRD-01 |
| scene / split registry | §3.2, appendix Table 5 | `src/anima_def_ghostfwl/data/splits.py` | PRD-01 |
| peak-level annotation semantics | §3.3, appendix C.2-C.4 | `src/anima_def_ghostfwl/data/labels.py` | PRD-01 |
| preprocessing: crop top/bottom/front, downsample, random crop | §5.1, appendix D.1/D.2 | `src/anima_def_ghostfwl/data/preprocess.py` | PRD-01 |
| FWL-MAE patch embedding + masking | §4.1, appendix Table 7 | `src/anima_def_ghostfwl/models/fwl_mae_pretrain.py` | PRD-02 |
| peak position / amplitude / width heads | §4.1, Eq. (1), appendix D.1 | `src/anima_def_ghostfwl/models/peak_heads.py` | PRD-02 |
| frozen-encoder ghost classifier | §4.2, Eq. (6) | `src/anima_def_ghostfwl/models/fwl_classifier.py` | PRD-02 |
| pretrain / finetune trainers | appendix D.1/D.2 | `scripts/train_pretrain.py`, `scripts/train_finetune.py` | PRD-02 |
| sliding-window inference + merge | appendix D.2 | `src/anima_def_ghostfwl/inference/sliding_window.py` | PRD-03 |
| thresholded class assignment | appendix D.2 | `src/anima_def_ghostfwl/inference/postprocess.py` | PRD-03 |
| ghost point removal | §4.2, §5.1 | `src/anima_def_ghostfwl/inference/remove_ghosts.py` | PRD-03 |
| ghost recall / removal metrics | §5.1, appendix E.1 | `src/anima_def_ghostfwl/eval/metrics.py` | PRD-04 |
| SLAM benchmark harness | §5.2.1 | `src/anima_def_ghostfwl/eval/slam_benchmark.py` | PRD-04 |
| object-detection ghost-FP benchmark | §5.2.2 | `src/anima_def_ghostfwl/eval/detection_benchmark.py` | PRD-04 |
| FastAPI service | ANIMA deployment target | `src/anima_def_ghostfwl/api/app.py` | PRD-05 |
| Docker / compose runtime | ANIMA deployment target | `Dockerfile`, `docker-compose.yml` | PRD-05 |
| ROS2 ghost-filter node | ANIMA robotics integration | `src/anima_def_ghostfwl/ros2/node.py` | PRD-06 |
| model export and release artifacts | ANIMA production | `src/anima_def_ghostfwl/export/exporter.py` | PRD-07 |

## Data Flow

`raw voxel .b2 (400,512,700)`  
→ `crop top/bottom/front bins`  
→ `downsample T and crop spatial ROI`  
→ `reorder to Tensor[B,1,256,128,128]`  
→ `FWL-MAE encoder / frozen classifier`  
→ `class logits per waveform bin`  
→ `ghost thresholding + peak selection`  
→ `remove ghost peaks / points`  
→ `clean point cloud for SLAM, detection, ROS2, API`

## Planned Directory Map

| Directory | Responsibility |
|---|---|
| `src/anima_def_ghostfwl/data/` | file IO, preprocessing, split manifests, dataset adapters |
| `src/anima_def_ghostfwl/models/` | MAE encoder, classifier head, losses |
| `src/anima_def_ghostfwl/inference/` | checkpoint loading, tiled inference, denoising |
| `src/anima_def_ghostfwl/eval/` | recall, removal rate, SLAM and detection evaluation |
| `src/anima_def_ghostfwl/api/` | FastAPI serving contract |
| `src/anima_def_ghostfwl/ros2/` | ROS2 node, topic bridge, launch config |
| `src/anima_def_ghostfwl/export/` | artifact export and production packaging |

## Repo Drift That Must Be Resolved

| Current Scaffold | Target State | Why |
|---|---|---|
| `src/anima_shinigami/` | `src/anima_def_ghostfwl/` | align module identity with repo docs |
| `project.name = "anima-shinigami"` | `project.name = "anima-def-ghostfwl"` | package metadata correctness |
| `NEXT_STEPS.md` / `PRD.md` use `SHINIGAMI` | replace with `DEF-GHOSTFWL` | avoid cross-project contamination |

## Non-Goals For This Planning Pass

1. No code implementation yet.
2. No dataset download automation until the public archive URL is confirmed.
3. No attempt to port training to MLX before CUDA reproduction exists.
