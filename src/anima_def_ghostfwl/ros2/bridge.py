"""ANIMA graph bridge for Ghost-FWL ROS2 integration.

Provides adapters to connect the Ghost-FWL node to other ANIMA nodes
in a processing graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from anima_def_ghostfwl.ros2.messages import (
    DenoisedCloudMsg,
    encode_voxel,
)
from anima_def_ghostfwl.ros2.node import GhostFilterNode


@dataclass
class GhostFWLBridge:
    """Bridge connecting Ghost-FWL to ANIMA pipeline graph."""

    node: GhostFilterNode = field(default_factory=GhostFilterNode)
    _frame_counter: int = field(default=0, init=False)

    def process_volume(
        self,
        volume: np.ndarray,
        *,
        frame_id: str | None = None,
    ) -> DenoisedCloudMsg | None:
        """Process a raw numpy volume through the denoising pipeline."""
        self._frame_counter += 1
        fid = frame_id or f"frame_{self._frame_counter:06d}"

        msg = encode_voxel(volume, frame_id=fid)
        return self.node.on_waveform(msg)

    @property
    def is_ready(self) -> bool:
        return self.node.is_ready

    def get_bridge_status(self) -> dict[str, Any]:
        """Combined bridge and node status."""
        return {
            "bridge_frames_processed": self._frame_counter,
            **self.node.get_status(),
        }
