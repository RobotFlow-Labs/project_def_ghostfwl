from pathlib import Path

import numpy as np

from anima_def_ghostfwl.data.io import (
    discover_frame_files,
    load_blosc2_array,
    load_peak_npy,
    save_blosc2_array,
)


def test_blosc2_roundtrip(tmp_path: Path) -> None:
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "sample_voxel.b2"

    save_blosc2_array(path, array)
    loaded = load_blosc2_array(path)

    assert np.array_equal(loaded, array)


def test_peak_npy_loads_object_arrays(tmp_path: Path) -> None:
    payload = np.array([[0, 0, [[10, 0.7, 3.0]]]], dtype=object)
    path = tmp_path / "frame_peak.npy"
    np.save(path, payload, allow_pickle=True)

    loaded = load_peak_npy(path)
    assert loaded.shape == (1, 3)
    assert loaded[0][2][0][0] == 10


def test_discover_frame_files_matches_related_assets(tmp_path: Path) -> None:
    voxel_root = tmp_path / "voxels"
    ann_root = tmp_path / "annotations"
    peak_root = tmp_path / "peaks"
    voxel_root.mkdir()
    ann_root.mkdir()
    peak_root.mkdir()

    frame_id = "20260101000000_t000"
    save_blosc2_array(voxel_root / f"{frame_id}_voxel.b2", np.zeros((2, 2, 2), dtype=np.float32))
    save_blosc2_array(
        ann_root / f"{frame_id}_annotation_voxel.b2",
        np.zeros((2, 2, 2), dtype=np.int16),
    )
    np.save(
        peak_root / f"{frame_id}_peak.npy",
        np.array([[0, 0, []]], dtype=object),
        allow_pickle=True,
    )

    frames = discover_frame_files([voxel_root], [ann_root], [peak_root])

    assert len(frames) == 1
    assert frames[0].frame_id == frame_id
    assert frames[0].annotation_path is not None
    assert frames[0].peak_path is not None
