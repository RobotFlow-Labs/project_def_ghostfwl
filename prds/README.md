# DEF-GHOSTFWL PRD Suite

This directory contains the executable PRD set for reproducing and productizing Ghost-FWL inside ANIMA.

## Order

1. `PRD-01-foundation.md`
2. `PRD-02-core-model.md`
3. `PRD-03-inference.md`
4. `PRD-04-evaluation.md`
5. `PRD-05-api-docker.md`
6. `PRD-06-ros2.md`
7. `PRD-07-production.md`

## Rules

1. Build against the paper first, not against convenience.
2. Preserve paper preprocessing, shapes, split policy, and metrics before adapting for ANIMA hardware.
3. Treat the current `SHINIGAMI` namespace as scaffold debt that must be cleaned in PRD-01.
4. Use `uv`, `pytest`, `ruff`, and TOML-first configuration throughout.
