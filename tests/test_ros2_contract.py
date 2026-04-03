"""Contract tests for Ghost-FWL ROS2 node (no rclpy required)."""

from __future__ import annotations

import numpy as np

from anima_def_ghostfwl.ros2.bridge import GhostFWLBridge
from anima_def_ghostfwl.ros2.messages import (
    DenoisedCloudMsg,
    WaveformVoxelMsg,
    decode_voxel,
    encode_voxel,
    points_to_cloud_msg,
)
from anima_def_ghostfwl.ros2.node import GhostFilterNode


def test_encode_decode_voxel_roundtrip() -> None:
    volume = np.random.randn(4, 4, 4).astype(np.float32)
    msg = encode_voxel(volume, frame_id="test_001")
    decoded = decode_voxel(msg)
    assert np.array_equal(volume, decoded)
    assert msg.frame_id == "test_001"


def test_points_to_cloud_msg() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cloud = points_to_cloud_msg(points, frame_id="f1", ghost_count=5, total_count=100)
    assert isinstance(cloud, DenoisedCloudMsg)
    assert cloud.ghost_count == 5
    assert cloud.points.shape == (2, 3)


def test_ghost_filter_node_not_ready_without_checkpoint() -> None:
    node = GhostFilterNode()
    assert node.is_ready is False


def test_ghost_filter_node_returns_none_when_not_ready() -> None:
    node = GhostFilterNode()
    msg = WaveformVoxelMsg(
        shape=(4, 4, 4),
        dtype="float32",
        data=np.zeros((4, 4, 4), dtype=np.float32).tobytes(),
    )
    assert node.on_waveform(msg) is None


def test_ghost_filter_node_status() -> None:
    node = GhostFilterNode()
    status = node.get_status()
    assert status["model_loaded"] is False
    assert status["device"] == "cpu"


def test_bridge_not_ready_without_checkpoint() -> None:
    bridge = GhostFWLBridge()
    assert bridge.is_ready is False


def test_bridge_status_includes_frame_counter() -> None:
    bridge = GhostFWLBridge()
    status = bridge.get_bridge_status()
    assert status["bridge_frames_processed"] == 0


def test_bridge_process_returns_none_without_model() -> None:
    bridge = GhostFWLBridge()
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    result = bridge.process_volume(volume)
    assert result is None
