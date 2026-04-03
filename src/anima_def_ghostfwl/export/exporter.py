"""Model export pipeline: pth -> safetensors -> ONNX -> TensorRT."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from anima_def_ghostfwl.models.fwl_mae_pretrain import FWLMAEConfig
from anima_def_ghostfwl.version import __version__


@dataclass
class ExportManifest:
    """Tracks which formats were successfully exported."""

    module: str = "def-ghostfwl"
    version: str = __version__
    formats: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> ExportManifest:
        data = json.loads(path.read_text())
        return cls(**data)


def export_safetensors(
    model: nn.Module,
    output_dir: Path,
    *,
    filename: str = "model.safetensors",
) -> Path:
    """Export model weights in safetensors format."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors package required: uv pip install safetensors")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    save_file(model.state_dict(), str(path))
    return path


def export_onnx(
    model: nn.Module,
    output_dir: Path,
    *,
    config: FWLMAEConfig | None = None,
    filename: str = "model.onnx",
    opset_version: int = 17,
) -> Path:
    """Export model to ONNX format."""
    config = config or FWLMAEConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    dummy_input = torch.randn(1, 1, *config.voxel_size)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        opset_version=opset_version,
        input_names=["voxel"],
        output_names=["logits"],
        dynamic_axes={"voxel": {0: "batch"}, "logits": {0: "batch"}},
    )
    return path


def export_checkpoint(
    model: nn.Module,
    output_dir: Path,
    *,
    config: FWLMAEConfig | None = None,
    metrics: dict[str, float] | None = None,
    filename: str = "best.pth",
) -> Path:
    """Save a complete checkpoint with config and metrics."""
    config = config or FWLMAEConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics or {},
            "version": __version__,
        },
        path,
    )
    return path


def run_export_pipeline(
    model: nn.Module,
    output_dir: Path,
    *,
    config: FWLMAEConfig | None = None,
    metrics: dict[str, float] | None = None,
) -> ExportManifest:
    """Run the full export pipeline and return a manifest."""
    config = config or FWLMAEConfig()
    manifest = ExportManifest(config=asdict(config), metrics=metrics or {})

    # pth
    pth_path = export_checkpoint(model, output_dir, config=config, metrics=metrics)
    manifest.formats["pth"] = str(pth_path)

    # safetensors
    try:
        st_path = export_safetensors(model, output_dir)
        manifest.formats["safetensors"] = str(st_path)
    except ImportError:
        manifest.formats["safetensors"] = "SKIPPED: safetensors not installed"

    # ONNX
    try:
        onnx_path = export_onnx(model, output_dir, config=config)
        manifest.formats["onnx"] = str(onnx_path)
    except Exception as e:
        manifest.formats["onnx"] = f"FAILED: {e}"

    # TRT placeholders — require tensorrt runtime
    manifest.formats["trt_fp16"] = "PENDING: requires tensorrt"
    manifest.formats["trt_fp32"] = "PENDING: requires tensorrt"

    manifest.save(output_dir / "export_manifest.json")
    return manifest
