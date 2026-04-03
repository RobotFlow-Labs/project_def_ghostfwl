#!/usr/bin/env python3
"""Export pipeline: best.pth → safetensors → ONNX → TRT FP16 + TRT FP32.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/export_kitti.py \
        --checkpoint /mnt/artifacts-datai/checkpoints/project_def_ghostfwl/best.pth \
        --output-dir /mnt/artifacts-datai/exports/project_def_ghostfwl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from anima_def_ghostfwl.models.ghost_detector_3d import GhostDetector3D


def export_all(checkpoint_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"[EXPORT] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle torch.compile state dict keys
    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        # Remove _orig_mod. prefix from torch.compile
        clean_key = k.replace("_orig_mod.", "")
        cleaned[clean_key] = v

    model_cfg = ckpt.get("config", {}).get("model", {})
    model = GhostDetector3D(
        in_channels=model_cfg.get("in_channels", 2),
        num_classes=model_cfg.get("num_classes", 3),
        base_ch=model_cfg.get("base_channels", 32),
    )
    model.load_state_dict(cleaned, strict=True)
    model = model.to(device).eval()
    print(f"[EXPORT] Model loaded: {model.param_count:,} params")

    # 1. Save pth (clean)
    pth_path = out / "ghost_detector_3d.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg}, pth_path)
    print(f"[EXPORT] pth: {pth_path} ({pth_path.stat().st_size / 1e6:.1f}MB)")

    # 2. Safetensors
    try:
        from safetensors.torch import save_file
        st_path = out / "ghost_detector_3d.safetensors"
        save_file(model.state_dict(), str(st_path))
        print(f"[EXPORT] safetensors: {st_path} ({st_path.stat().st_size / 1e6:.1f}MB)")
    except ImportError:
        print("[EXPORT] safetensors: SKIPPED (not installed)")

    # 3. ONNX
    onnx_path = out / "ghost_detector_3d.onnx"
    dummy = torch.randn(1, 2, 256, 256, 32, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["voxel"],
        output_names=["logits"],
        dynamic_axes={"voxel": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[EXPORT] ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f}MB)")

    # 4. TRT FP16 + FP32
    trt_script = "/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py"
    if Path(trt_script).exists():
        print("[EXPORT] Running TRT export...")
        try:
            subprocess.run(
                [sys.executable, trt_script, "--onnx", str(onnx_path), "--output-dir", str(out)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"[EXPORT] TRT FP16 + FP32: {out}")
        except subprocess.CalledProcessError as e:
            print(f"[EXPORT] TRT FAILED: {e.stderr}")
    else:
        print(f"[EXPORT] TRT toolkit not found at {trt_script}")

    # Summary
    print("\n[EXPORT] Summary:")
    for f in sorted(out.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="/mnt/artifacts-datai/checkpoints/project_def_ghostfwl/best.pth",
    )
    p.add_argument(
        "--output-dir",
        default="/mnt/artifacts-datai/exports/project_def_ghostfwl",
    )
    args = p.parse_args()
    export_all(args.checkpoint, args.output_dir)
