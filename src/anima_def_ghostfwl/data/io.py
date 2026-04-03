"""I/O utilities for Ghost-FWL blosc2 voxels and peak metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import blosc2
import numpy as np


@dataclass
class FrameFiles:
    """Paths for a single Ghost-FWL frame's assets."""

    frame_id: str
    voxel_path: Path
    annotation_path: Path | None = None
    peak_path: Path | None = None


def save_blosc2_array(path: str | Path, array: np.ndarray) -> Path:
    """Save a numpy array as a blosc2 file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blosc2.save_array(array, str(target), mode="w")
    return target


def load_blosc2_array(path: str | Path) -> np.ndarray:
    """Load a blosc2-compressed numpy array."""
    return blosc2.load_array(str(path))


def load_peak_npy(path: str | Path) -> np.ndarray:
    """Load a peak metadata file (object arrays with nested lists)."""
    return np.load(str(path), allow_pickle=True)


def _extract_frame_id(filename: str) -> str:
    """Extract frame ID by removing known suffixes."""
    for suffix in ("_annotation_voxel", "_voxel", "_peak"):
        if suffix in filename:
            return filename.split(suffix)[0]
    return Path(filename).stem


def discover_frame_files(
    voxel_dirs: list[Path],
    annotation_dirs: list[Path] | None = None,
    peak_dirs: list[Path] | None = None,
) -> list[FrameFiles]:
    """Match voxel files with their annotation and peak counterparts."""
    annotation_dirs = annotation_dirs or []
    peak_dirs = peak_dirs or []

    annotation_map: dict[str, Path] = {}
    for d in annotation_dirs:
        for f in sorted(d.glob("*_annotation_voxel.b2")):
            fid = _extract_frame_id(f.name)
            annotation_map[fid] = f

    peak_map: dict[str, Path] = {}
    for d in peak_dirs:
        for f in sorted(d.glob("*_peak.npy")):
            fid = _extract_frame_id(f.name)
            peak_map[fid] = f

    frames: list[FrameFiles] = []
    for d in voxel_dirs:
        for f in sorted(d.glob("*_voxel.b2")):
            fid = _extract_frame_id(f.name)
            frames.append(
                FrameFiles(
                    frame_id=fid,
                    voxel_path=f,
                    annotation_path=annotation_map.get(fid),
                    peak_path=peak_map.get(fid),
                )
            )
    return frames
