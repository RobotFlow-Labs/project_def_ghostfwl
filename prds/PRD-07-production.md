# PRD-07: Production

> Module: DEF-GHOSTFWL | Priority: P2
> Depends on: PRD-04, PRD-05, PRD-06
> Status: ⬜ Not started

## Objective
The Ghost-FWL module is exportable, observable, and ready for controlled production deployment with explicit metric gates against the paper.

## Context (from paper)
The paper’s value is downstream robustness. Productionizing this module therefore means guarding denoising quality, not just serving a model.
Paper reference: §5 overall, especially SLAM and object-detection gains.

## Acceptance Criteria
- [ ] Export pipeline saves checkpoints, config, metadata, and benchmark report together.
- [ ] Release validation refuses promotion when paper-derived gates are not met.
- [ ] Runtime includes structured logging, degradation handling, and checkpoint fingerprinting.
- [ ] Optional model-card / Hugging Face packaging path exists.
- [ ] Test: `uv run pytest tests/test_export.py tests/test_release_gate.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_ghostfwl/export/exporter.py` | package model + metadata | §5 | ~120 |
| `src/anima_def_ghostfwl/export/model_card.py` | artifact summary | §5 | ~80 |
| `src/anima_def_ghostfwl/ops/release_gate.py` | benchmark gate | §5.1-§5.2 | ~120 |
| `src/anima_def_ghostfwl/ops/observability.py` | structured logs and counters | — | ~100 |
| `tests/test_export.py` | export tests | — | ~80 |
| `tests/test_release_gate.py` | release-gate tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- trained checkpoint
- frozen config snapshot
- evaluation report

### Outputs
- portable artifact bundle
- model card
- pass/fail production decision

### Algorithm
```python
def release_gate(report):
    assert report["recall"] >= 0.73
    assert report["ghost_removal_rate"] >= 0.90
    assert report["ghost_fp_rate"] <= 0.02
    return True
```

## Dependencies
```toml
huggingface-hub = ">=0.31"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| benchmark report JSON | 1 file | `artifacts/reports/ghostfwl_benchmark.json` | generated locally |
| export bundle | 1 directory | `artifacts/export/` | generated locally |

## Test Plan
```bash
uv run pytest tests/test_export.py tests/test_release_gate.py -v
```

## References
- Paper: §5 "Experiments and Results"
- Depends on: PRD-04, PRD-05, PRD-06
