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
- **Phase**: TRAINING on GPU 6 (CUDA_VISIBLE_DEVICES=6)
- **MVP Readiness**: 85%
- **Training**:
  - Model: GhostDetector3D (3D U-Net, 1.2M params, depthwise-separable convs)
  - Data: 7,481 KITTI voxel tensors cached at /mnt/forge-data/shared_infra/datasets/kitti_voxel_cache/
  - Config: bs=4, bf16, AdamW lr=1e-3, cosine warmup, 50 epochs
  - VRAM: 65% on L4 (14.9GB / 23GB)
  - Steps/sec: ~1.7, epoch ~17 min, total ~14 hrs
  - Step 600: val_loss=0.0234 (down from 0.1599 at step 200)
  - PID: 2483138, Log: /mnt/artifacts-datai/logs/project_def_ghostfwl/train_20260403_1634.log
- **Accomplished**:
  - PRD-01 through PRD-07: All complete (23/23 tasks)
  - GPU voxelization pipeline: 7481 KITTI scans cached in 197s (37.9 files/sec)
  - Shared voxel cache for OccAny/ProjFusion/LiDAR modules
  - Full ANIMA infra: anima_module.yaml, serve.py, Docker, ROS2
  - 83 unit tests passing
- **TODO (after training completes)**:
  1. Export: pth → safetensors → ONNX → TRT FP16 → TRT FP32
  2. Push to HuggingFace: ilessio-aiflowlab/project_def_ghostfwl
  3. Update TRAINING_REPORT.md with final metrics
  4. Git commit + push

## 4. Shared Infra Created
- Voxel cache: `/mnt/forge-data/shared_infra/datasets/kitti_voxel_cache/` (7,481 files, ~14GB)
  - Format: (2, 256, 256, 32) float32 .pt tensors
  - Grid: [-51.2, 51.2]² x [-5, 3] at 0.4m/0.4m/0.25m resolution
  - Channels: occupancy + mean reflectance
  - Updated MAP.md with entry

## 5. Hardware
- Training GPU: NVIDIA L4 #6 (23GB VRAM)
- VRAM usage: 14.9GB (65%)
- torch 2.11.0+cu128
- torch.compile enabled

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex | PRD-01 through PRD-03 |
| 2026-04-03 | Opus 4.6 | PRD-04 through PRD-07, full ANIMA infra |
| 2026-04-03 | Opus 4.6 | CUDA pipeline: KITTI voxelization + GhostDetector3D training launched |
