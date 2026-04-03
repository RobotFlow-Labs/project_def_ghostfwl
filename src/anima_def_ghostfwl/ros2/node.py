"""ROS2 Ghost-FWL denoising node.

Subscribes to waveform voxel inputs, runs inference, and publishes
denoised point clouds. Falls back to standalone mode when rclpy is
not available.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from anima_def_ghostfwl.inference.checkpoint import load_predictor
from anima_def_ghostfwl.inference.postprocess import labels_to_point_cloud
from anima_def_ghostfwl.inference.sliding_window import infer_tiled
from anima_def_ghostfwl.ros2.messages import (
    DenoisedCloudMsg,
    WaveformVoxelMsg,
    decode_voxel,
    points_to_cloud_msg,
)

try:
    import rclpy
    from rclpy.node import Node as RclpyNode

    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    RclpyNode = object


class GhostFilterNode(RclpyNode if HAS_RCLPY else object):
    """ROS2-compatible node for real-time ghost point filtering."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        node_name: str = "ghost_filter_node",
    ) -> None:
        if HAS_RCLPY:
            super().__init__(node_name)

        self.threshold = threshold
        self.device = device
        self._predictor = None
        self._last_result: DenoisedCloudMsg | None = None

        ckpt = checkpoint_path or os.environ.get("ANIMA_CHECKPOINT_PATH")
        if ckpt and Path(ckpt).exists():
            self._predictor = load_predictor(ckpt, device=device)

    @property
    def is_ready(self) -> bool:
        return self._predictor is not None

    def on_waveform(self, msg: WaveformVoxelMsg) -> DenoisedCloudMsg | None:
        """Process an incoming waveform voxel message."""
        if not self.is_ready:
            return None

        volume = decode_voxel(msg)
        window_shape = (
            self._predictor.config.voxel_size[1],
            self._predictor.config.voxel_size[2],
            self._predictor.config.voxel_size[0],
        )
        labels = infer_tiled(
            volume, self._predictor, window_shape=window_shape, threshold=self.threshold
        )
        points = labels_to_point_cloud(labels)
        ghost_count = int((labels == 3).sum())

        result = points_to_cloud_msg(
            points,
            frame_id=msg.frame_id,
            timestamp_ns=msg.timestamp_ns,
            ghost_count=ghost_count,
            total_count=int(labels.size),
        )
        self._last_result = result
        return result

    def get_status(self) -> dict:
        """Module-specific status for health reporting."""
        return {
            "model_loaded": self.is_ready,
            "device": self.device,
            "threshold": self.threshold,
            "last_frame": self._last_result.frame_id if self._last_result else None,
        }


def main() -> None:
    """Standalone entry point for the Ghost-FWL ROS2 node."""
    if HAS_RCLPY:
        rclpy.init()
        node = GhostFilterNode(
            checkpoint_path=os.environ.get("ANIMA_CHECKPOINT_PATH"),
            device=os.environ.get("ANIMA_DEVICE", "cpu"),
            threshold=float(os.environ.get("ANIMA_THRESHOLD", "0.5")),
        )
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[GhostFilterNode] rclpy not available — running in standalone test mode")
        node = GhostFilterNode()
        print(f"[GhostFilterNode] Status: {node.get_status()}")


if __name__ == "__main__":
    main()
