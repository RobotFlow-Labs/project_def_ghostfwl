import numpy as np
import torch

from anima_def_ghostfwl.data.preprocess import (
    FWLPreprocessor,
    crop_histogram_front,
    crop_xy_patch,
    crop_y_axis,
    downsample_histogram_axis,
)


def test_crop_y_axis_matches_paper_removal() -> None:
    voxel = np.zeros((400, 512, 700), dtype=np.float32)
    cropped = crop_y_axis(voxel, top=90, bottom=90)
    assert cropped.shape == (400, 332, 700)


def test_crop_histogram_front_matches_paper_removal() -> None:
    voxel = np.zeros((400, 332, 700), dtype=np.float32)
    cropped = crop_histogram_front(voxel, front=25)
    assert cropped.shape == (400, 332, 675)


def test_downsample_histogram_axis_produces_target_length() -> None:
    voxel = np.arange(400 * 332 * 675, dtype=np.float32).reshape(400, 332, 675)
    downsampled = downsample_histogram_axis(voxel, 256)
    assert downsampled.shape == (400, 332, 256)
    assert downsampled[0, 0, 0] == voxel[0, 0, 0]
    assert downsampled[0, 0, -1] == voxel[0, 0, -1]


def test_crop_xy_patch_supports_deterministic_coordinates() -> None:
    voxel = np.zeros((400, 332, 256), dtype=np.float32)
    cropped = crop_xy_patch(voxel, target_x=128, target_y=128, start_x=10, start_y=20)
    assert cropped.shape == (128, 128, 256)


def test_full_preprocessor_returns_cthw_tensor() -> None:
    rng = np.random.default_rng(42)
    voxel = rng.normal(size=(400, 512, 700)).astype(np.float32)

    tensor = FWLPreprocessor()(voxel, start_x=0, start_y=0)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 256, 128, 128)
    assert tensor.dtype == torch.float32
