# DEF-GHOSTFWL — Design & Implementation Checklist

## Paper: Ghost-FWL: LiDAR Ghost Object Detection
## ArXiv: 2603.28224
## Repo: https://github.com/keio-csg/Ghost-FWL

---

## Phase 1: Scaffold + Verification
- [x] Project structure created
- [x] Paper PDF downloaded to papers/
- [x] Paper read and annotated
- [x] Reference repo cloned
- [ ] Reference demo runs successfully
- [ ] Datasets identified and accessibility confirmed
- [ ] CLAUDE.md filled with paper-specific details
- [ ] PRD.md filled with architecture and plan

## Phase 2: Reproduce
- [x] Core model implemented in src/anima_def_ghostfwl/
- [x] Paper-default training CLI scaffold (`scripts/train_pretrain.py`, `scripts/train_finetune.py`)
- [ ] Evaluation pipeline (scripts/eval.py)
- [ ] Metrics match paper (within ±5%)
- [ ] Dual-compute verified (MLX + CUDA)

## Phase 3: Adapt to Hardware
- [ ] ZED 2i data pipeline (if applicable)
- [ ] Unitree L2 LiDAR pipeline (if applicable)
- [ ] xArm 6 integration (if manipulation module)
- [ ] Real sensor inference test
- [ ] MLX inference port validated

## Phase 4: ANIMA Integration
- [ ] ROS2 bridge node
- [ ] Docker container builds and runs
- [ ] API endpoints defined
- [ ] Integration test with stack: Defense

## Shenzhen Demo Readiness
- [ ] Demo script works end-to-end
- [ ] Demo data prepared
- [ ] Demo runs in < 30 seconds
- [ ] Demo visuals are compelling
