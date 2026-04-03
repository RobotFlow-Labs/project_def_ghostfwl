# PRD-06: ROS2 Integration

> Module: DEF-GHOSTFWL | Priority: P1
> Depends on: PRD-03, PRD-05
> Status: ⬜ Not started

## Objective
ANIMA can run Ghost-FWL as a ROS2 node that subscribes to waveform input, emits denoised point clouds, and integrates into robotics pipelines without changing paper inference semantics.

## Context (from paper)
Ghost-FWL is motivated by robotics and autonomous driving scenarios where sparse mobile LiDAR data must be cleaned before SLAM or detection.
Paper reference: introduction, §3.2, §5.2.1, §5.2.2.

## Acceptance Criteria
- [ ] ROS2 node subscribes to waveform or prebuilt voxel messages and publishes denoised point clouds plus optional ghost masks.
- [ ] Launch file configures checkpoint path, threshold, and backend.
- [ ] Integration bridge can hand denoised outputs to downstream ANIMA nodes.
- [ ] Test: `uv run pytest tests/test_ros2_contract.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_def_ghostfwl/ros2/messages.py` | internal ROS payload adapters | §5.2 | ~80 |
| `src/anima_def_ghostfwl/ros2/node.py` | ROS2 runtime node | §5.2.1 | ~180 |
| `src/anima_def_ghostfwl/ros2/bridge.py` | ANIMA graph bridge | §5.2 | ~100 |
| `launch/def_ghostfwl.launch.py` | launch definition | — | ~60 |
| `tests/test_ros2_contract.py` | node contract tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `WaveformVoxelMsg` or file-backed volume equivalent
- checkpoint + threshold config

### Outputs
- `sensor_msgs/PointCloud2`-compatible denoised cloud
- debug mask / class overlays

### Algorithm
```python
class GhostFilterNode(Node):
    def on_waveform(self, msg):
        voxel = decode_waveform(msg)
        result = self.service.run(voxel, threshold=self.threshold)
        self.pub_cloud.publish(to_pointcloud2(result.points))
```

## Dependencies
```toml
rclpy = "*"
sensor-msgs-py = "*"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| ROS2 sample bag | small test capture | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/rosbags/` | local recording |
| finetuned checkpoint | 1 file | `/Volumes/AIFlowDev/RobotFlowLabs/models/ghost_fwl/fwl_mae_classifier.ckpt` | pending |

## Test Plan
```bash
uv run pytest tests/test_ros2_contract.py -v
```

## References
- Paper: §3.2 "Sensing Scenario and Scene"
- Paper: §5.2.1 "Evaluation on SLAM"
- Paper: §5.2.2 "Evaluation on Object Detection"
- Depends on: PRD-03, PRD-05
- Feeds into: PRD-07
