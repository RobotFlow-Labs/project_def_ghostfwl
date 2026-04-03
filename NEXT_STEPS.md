# DEF-GHOSTFWL — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.
This project covers exactly ONE paper: Ghost-FWL: LiDAR Ghost Object Detection.

## 1. Working Rules
- Work only inside `project_def_ghostfwl/`
- This wave has 17 parallel projects, 17 papers, 17 agents
- Prefix every commit with `[DEF-GHOSTFWL]`
- Stage only `project_def_ghostfwl/` files
- VERIFY THE PAPER BEFORE BUILDING ANYTHING

## 2. The Paper
- **Title**: Ghost-FWL: LiDAR Ghost Object Detection
- **ArXiv**: 2603.28224
- **Link**: https://arxiv.org/abs/2603.28224
- **Repo**: https://github.com/keio-csg/Ghost-FWL
- **Compute**: MLX-OK
- **Verification status**: ArXiv ID ✅ | Repo ✅ | Paper read ✅

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: All PRDs complete — awaiting dataset assets for training
- **MVP Readiness**: 75%
- **Accomplished**:
  - PRD-01: Foundation & config (data/labels, data/io, data/preprocess)
  - PRD-02: Core model (FWL-MAE, classifier, losses, patch embed)
  - PRD-03: Inference pipeline (checkpoint, sliding window, postprocess, CLI)
  - PRD-04: Evaluation harness (metrics, splits, report, benchmark CLI)
  - PRD-05: API & Docker (FastAPI endpoints, schemas, service, Dockerfile.serve)
  - PRD-06: ROS2 integration (node, messages, bridge, launch file)
  - PRD-07: Production (export pipeline, model card, release gates, observability)
  - Full ANIMA infra: anima_module.yaml, serve.py, docker-compose.serve.yml, .env.serve
  - 70+ tests passing across all modules
- **TODO**:
  1. Acquire Ghost-FWL dataset archive (24,412 supervised + 8,933 pretrain frames)
  2. Acquire or train pretrained checkpoints (FWL-MAE encoder + classifier)
  3. Run training pipeline with real data
  4. Validate paper metrics (Recall 0.751, GRR 0.918)
  5. Export: pth → safetensors → ONNX → TRT FP16 → TRT FP32
  6. Push to HuggingFace: ilessio-aiflowlab/project_def_ghostfwl
- **Blockers**: Public dataset archive and pretrained checkpoints are still unresolved

## 4. Datasets
### Required for this paper
| Dataset | Size | URL | Format | Phase Needed |
|---------|------|-----|--------|-------------|
| Ghost-FWL supervised set | 24,412 annotated frames | https://keio-csg.github.io/Ghost-FWL/ | `.b2` + `.npy` | Training |
| Ghost-FWL mobile pretrain set | 8,933 unlabeled frames | https://keio-csg.github.io/Ghost-FWL/ | `.b2` + `.npy` | Pretraining |

### Check shared volume first
/mnt/forge-data/datasets

### Download
`bash scripts/download_data.sh`

## 5. Hardware
- ZED 2i stereo camera: Available
- Unitree L2 3D LiDAR: Available
- xArm 6 cobot: Pending purchase
- Mac Studio M-series: MLX dev
- 8x RTX 6000 Pro Blackwell: GCloud

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex | PRD-01 foundation, PRD-02 model scaffolds, PRD-03 inference |
| 2026-04-03 | Opus 4.6 | PRD-04 evaluation, PRD-05 API/Docker, PRD-06 ROS2, PRD-07 production, full ANIMA infra |
