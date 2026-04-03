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
- **Phase**: PRD-04 evaluation and dataset integration
- **MVP Readiness**: 45%
- **Accomplished**: Paper read, planning suite generated, reference repo vendored, PRD-01 foundation completed, PRD-02 model/loss/CLI scaffold verified, PRD-03 inference/ghost-removal scaffold verified, 29 tests passing on Python 3.11
- **TODO**:
  1. Resolve missing Ghost-FWL dataset archive / checkpoint assets
  2. Add real dataset-backed training loops under `src/anima_def_ghostfwl/training/`
  3. Implement PRD-04 evaluation harness for recall / GRR / SLAM / detection metrics
  4. Add API / Docker runtime behavior beyond the current scaffold surfaces
  5. Add ROS2 packaging and release gating once real checkpoints exist
- **Blockers**: Public dataset archive and pretrained checkpoints are still unresolved

## 4. Datasets
### Required for this paper
| Dataset | Size | URL | Format | Phase Needed |
|---------|------|-----|--------|-------------|
| Ghost-FWL supervised set | 24,412 annotated frames | https://keio-csg.github.io/Ghost-FWL/ | `.b2` + `.npy` | PRD-01 to PRD-04 |
| Ghost-FWL mobile pretrain set | 8,933 unlabeled frames | https://keio-csg.github.io/Ghost-FWL/ | `.b2` + `.npy` | PRD-02 |

### Check shared volume first
/Volumes/AIFlowDev/RobotFlowLabs/datasets

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
| 2026-04-03 | Codex | Read autopilot playbook, normalized PRD/docs, started PRD-01 foundation build |
| 2026-04-03 | Codex | Completed PRD-01 foundation, implemented PRD-02 model/loss scaffolds, added paper-default train CLIs, verified 22 tests on Python 3.11 |
| 2026-04-03 | Codex | Implemented PRD-03 checkpoint loading, tiled inference, ghost-removal CLI, and expanded verification to 29 passing tests |
