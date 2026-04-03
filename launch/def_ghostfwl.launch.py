"""ROS2 launch file for Ghost-FWL denoising node."""

from __future__ import annotations

import os

try:
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node

    def generate_launch_description() -> LaunchDescription:
        return LaunchDescription(
            [
                DeclareLaunchArgument(
                    "checkpoint_path",
                    default_value=os.environ.get("ANIMA_CHECKPOINT_PATH", ""),
                    description="Path to the trained Ghost-FWL checkpoint",
                ),
                DeclareLaunchArgument(
                    "device",
                    default_value=os.environ.get("ANIMA_DEVICE", "cpu"),
                    description="Inference device (cpu, cuda, auto)",
                ),
                DeclareLaunchArgument(
                    "threshold",
                    default_value="0.5",
                    description="Ghost classification confidence threshold",
                ),
                Node(
                    package="anima_def_ghostfwl",
                    executable="ghost_filter_node",
                    name="ghost_filter",
                    output="screen",
                    parameters=[
                        {
                            "checkpoint_path": "",
                            "device": "cpu",
                            "threshold": 0.5,
                        }
                    ],
                ),
            ]
        )

except ImportError:
    def generate_launch_description():
        raise ImportError("ROS2 launch dependencies not available")
