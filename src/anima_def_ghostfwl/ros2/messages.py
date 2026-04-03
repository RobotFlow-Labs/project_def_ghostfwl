"""Internal ROS2 payload adapters for Ghost-FWL.

Converts between Ghost-FWL numpy volumes and ROS2 message types.
When rclpy is not available, these helpers still work for testing with
plain numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WaveformVoxelMsg:
    """Internal representation of a waveform voxel message."""

    frame_id: str = ""
    timestamp_ns: int = 0
    shape: tuple[int, int, int] = (0, 0, 0)
    dtype: str = "float32"
    data: bytes = b""


@dataclass
class DenoisedCloudMsg:
    """Internal representation of a denoised point cloud output."""

    frame_id: str = ""
    timestamp_ns: int = 0
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    ghost_count: int = 0
    total_count: int = 0


def encode_voxel(volume: np.ndarray, *, frame_id: str = "", timestamp_ns: int = 0) -> WaveformVoxelMsg:
    """Encode a numpy volume into a WaveformVoxelMsg."""
    return WaveformVoxelMsg(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        shape=volume.shape,
        dtype=str(volume.dtype),
        data=volume.tobytes(),
    )


def decode_voxel(msg: WaveformVoxelMsg) -> np.ndarray:
    """Decode a WaveformVoxelMsg back into a numpy volume."""
    return np.frombuffer(msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)


def points_to_cloud_msg(
    points: np.ndarray,
    *,
    frame_id: str = "",
    timestamp_ns: int = 0,
    ghost_count: int = 0,
    total_count: int = 0,
) -> DenoisedCloudMsg:
    """Wrap point coordinates into a DenoisedCloudMsg."""
    return DenoisedCloudMsg(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        points=points.astype(np.float32),
        ghost_count=ghost_count,
        total_count=total_count,
    )
