# PRD-04: Evaluation

> Module: DEF-GHOSTFWL | Priority: P1
> Depends on: PRD-01, PRD-02, PRD-03
> Status: ⬜ Not started

## Objective
The repo can reproduce the paper’s denoising metrics and run downstream SLAM / object-detection comparisons on denoised outputs.

## Context (from paper)
The paper measures ghost detection at peak level with recall, at point level with Ghost Removal Rate, and then evaluates downstream effects on SLAM and object detection.
Paper reference: §5.1, §5.2.1, §5.2.2, appendix E.1.

## Acceptance Criteria
- [ ] Evaluation code computes Recall and Ghost Removal Rate exactly as described.
- [ ] Split manifests reproduce train/val/test scene policy from appendix D.2.
- [ ] Threshold sweep covers `0.5`, `0.6`, `0.7`.
- [ ] Benchmark report compares outputs against paper targets: Recall `0.751`, Removal `0.918`, ATE `0.245`, RTE `0.245`, Ghost FP `1.34%`.
- [ ] Test: `uv run pytest tests/test_metrics.py tests/test_splits.py tests/test_benchmark_report.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_ghostfwl/eval/metrics.py` | recall and removal metrics | §5.1, appendix E.1 | ~140 |
| `src/anima_def_ghostfwl/eval/splits.py` | canonical train/val/test scenes | appendix D.2 | ~60 |
| `src/anima_def_ghostfwl/eval/slam_benchmark.py` | ATE/RTE evaluation hooks | §5.2.1 | ~160 |
| `src/anima_def_ghostfwl/eval/detection_benchmark.py` | Ghost FP evaluation hooks | §5.2.2 | ~160 |
| `src/anima_def_ghostfwl/eval/report.py` | markdown / JSON reports | §5 | ~120 |
| `scripts/evaluate.py` | unified benchmark CLI | §5 | ~70 |
| `tests/test_metrics.py` | metric tests | — | ~110 |
| `tests/test_splits.py` | split tests | — | ~40 |
| `tests/test_benchmark_report.py` | report tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `predicted_peak_labels`: `Tensor[T, H, W]`
- `ground_truth_peak_labels`: `Tensor[T, H, W]`
- `predicted_points`: `ndarray[N, 3]`
- `gt_ghost_points`: `ndarray[M, 3]`

### Outputs
- `recall`: `float`
- `ghost_removal_rate`: `float`
- `slam_metrics`: `{"ate_mean": float, "rte_mean": float}`
- `ghost_fp_rate`: `float`

### Algorithm
```python
# Paper §5.1 / Appendix E.1
def ghost_removal_rate(pred_points, gt_points, radius=0.001):
    removed = sum(no_neighbor_within_radius(gt, pred_points, radius) for gt in gt_points)
    return removed / len(gt_points)
```

## Dependencies
```toml
scipy = ">=1.13"
pandas = ">=2.2"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| test scenes 002/007/009 | 1,427 frames | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/ghost_dataset/scene{002,007,009}/` | pending |
| SLAM sequence | 231 frames / 23.4 m | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/downstream/slam/` | to be assembled locally |
| detection benchmark set | pedestrian + glass scenes | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/downstream/detection/` | to be assembled locally |

## Test Plan
```bash
uv run pytest tests/test_metrics.py tests/test_splits.py tests/test_benchmark_report.py -v
uv run python scripts/evaluate.py --help
```

## References
- Paper: §5.1 "Ghost Denoising Evaluation"
- Paper: §5.2.1 "Evaluation on SLAM"
- Paper: §5.2.2 "Evaluation on Object Detection"
- Appendix: E.1
- Depends on: PRD-01, PRD-02, PRD-03
- Feeds into: PRD-07
