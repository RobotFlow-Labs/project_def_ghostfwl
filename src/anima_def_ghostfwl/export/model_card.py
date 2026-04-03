"""Model card generation for HuggingFace uploads."""

from __future__ import annotations

from pathlib import Path

from anima_def_ghostfwl.version import __version__


def generate_model_card(
    *,
    metrics: dict[str, float] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a HuggingFace model card in markdown."""
    m = metrics or {}

    card = f"""---
license: apache-2.0
tags:
  - lidar
  - ghost-detection
  - point-cloud
  - denoising
  - anima
library_name: pytorch
pipeline_tag: other
---

# Ghost-FWL: LiDAR Ghost Object Detection

**ANIMA Module:** DEF-GHOSTFWL v{__version__}
**Paper:** [Ghost-FWL](https://arxiv.org/abs/2603.28224)

## Model Description

Ghost-FWL detects and removes ghost points caused by multi-path LiDAR
reflections through glass and reflective surfaces. It uses a transformer-based
classifier with FWL-MAE self-supervised pretraining on full-waveform LiDAR
histograms.

## Metrics

| Metric | Value | Paper Target |
|--------|-------|-------------|
| Recall | {m.get('recall', 'N/A')} | 0.751 |
| Ghost Removal Rate | {m.get('ghost_removal_rate', 'N/A')} | 0.918 |
| Ghost FP Rate | {m.get('ghost_fp_rate', 'N/A')} | 1.34% |
| SLAM ATE | {m.get('slam_ate', 'N/A')} | 0.245 m |
| SLAM RTE | {m.get('slam_rte', 'N/A')} | 0.245 m |

## Usage

```python
from anima_def_ghostfwl.inference import load_predictor

predictor = load_predictor("best.pth", device="cuda")
labels = predictor.predict_labels(volume)
```

## Citation

```bibtex
@article{{ghostfwl2025,
  title={{Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal}},
  author={{Ikeda et al.}},
  journal={{arXiv preprint arXiv:2603.28224}},
  year={{2025}}
}}
```

## License

Apache 2.0 — Part of the ANIMA Intelligence Compiler Suite by Robot Flow Labs.
"""

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(card)

    return card
