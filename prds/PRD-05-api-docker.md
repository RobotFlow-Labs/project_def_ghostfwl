# PRD-05: API & Docker

> Module: DEF-GHOSTFWL | Priority: P1
> Depends on: PRD-01, PRD-02, PRD-03
> Status: ⬜ Not started

## Objective
The Ghost-FWL denoiser is available as a containerized service with health checks, typed API contracts, and reproducible runtime packaging.

## Context (from paper)
The paper itself does not define a service boundary, but the ANIMA module needs a stable inference surface once the paper pipeline is reproduced.
Paper reference: §4.2 inference semantics must remain unchanged inside the service boundary.

## Acceptance Criteria
- [ ] FastAPI endpoint accepts voxel-path or uploaded tensor payloads and returns denoised outputs plus class summaries.
- [ ] Runtime loads a finetuned checkpoint and exposes `/healthz`, `/readyz`, and `/predict`.
- [ ] Docker image builds with `uv sync` and runs a smoke test.
- [ ] Compose file mounts dataset/checkpoint volumes read-only by default.
- [ ] Test: `uv run pytest tests/test_api.py tests/test_service_container.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_ghostfwl/api/schemas.py` | request / response models | §4.2 | ~100 |
| `src/anima_def_ghostfwl/api/app.py` | FastAPI app | §4.2 | ~160 |
| `src/anima_def_ghostfwl/api/service.py` | inference service wiring | §4.2 | ~120 |
| `Dockerfile` | container runtime | — | ~40 |
| `docker-compose.yml` | local orchestration | — | ~40 |
| `.env.example` | runtime env vars | — | ~30 |
| `tests/test_api.py` | contract tests | — | ~100 |
| `tests/test_service_container.py` | container smoke tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `voxel_path` or serialized `Tensor[400,512,700]`
- `checkpoint_path`
- optional `threshold=0.5`

### Outputs
- `ghost_mask_summary`
- `denoised_point_cloud_path`
- `class_volume_path`

### Algorithm
```python
@app.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    voxel = load_or_decode(req)
    result = denoise_service.run(voxel, threshold=req.threshold or 0.5)
    return serialize_result(result)
```

## Dependencies
```toml
fastapi = ">=0.115"
uvicorn = ">=0.34"
orjson = ">=3.10"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| finetuned checkpoint | 1 file | `/Volumes/AIFlowDev/RobotFlowLabs/models/ghost_fwl/fwl_mae_classifier.ckpt` | pending |
| input sample volume | 1 frame | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/samples/` | local fixture |

## Test Plan
```bash
uv run pytest tests/test_api.py tests/test_service_container.py -v
docker build -t def-ghostfwl .
```

## References
- Paper: §4.2 "Ghost Detection and Removal"
- Depends on: PRD-01, PRD-02, PRD-03
- Feeds into: PRD-06, PRD-07
