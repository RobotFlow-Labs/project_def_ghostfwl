# DEF-GHOSTFWL Task Index

## Build Order

1. PRD-0101 Namespace and metadata cleanup
2. PRD-0102 Settings and split manifest
3. PRD-0103 Dataset IO and preprocessing
4. PRD-0104 Foundation tests
5. PRD-0201 Patch embedding and masking core
6. PRD-0202 FWL-MAE pretrain model
7. PRD-0203 Ghost classifier and losses
8. PRD-0204 Training CLIs and loops
9. PRD-0301 Checkpoint loader and predictor
10. PRD-0302 Sliding-window inference
11. PRD-0303 Ghost removal outputs and CLI
12. PRD-0401 Metrics and split fixtures
13. PRD-0402 Benchmark runner
14. PRD-0403 Downstream report synthesis
15. PRD-0501 FastAPI schemas and service
16. PRD-0502 Docker and compose runtime
17. PRD-0503 API tests and smoke path
18. PRD-0601 ROS2 contracts
19. PRD-0602 ROS2 node and launch
20. PRD-0603 ANIMA bridge wiring
21. PRD-0701 Export bundle and model card
22. PRD-0702 Release gates and observability
23. PRD-0703 Production validation tests

## Dependency Notes

- Do not begin PRD-02 before PRD-01 normalizes the namespace away from `SHINIGAMI`.
- Do not benchmark before PRD-03 inference outputs are stable.
- Do not ship API, ROS2, or production artifacts until the paper metrics are measurable in PRD-04.
