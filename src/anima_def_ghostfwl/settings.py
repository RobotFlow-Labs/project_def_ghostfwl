"""Typed settings for DEF-GHOSTFWL."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GhostFWLSettings(BaseSettings):
    """Module settings that stay aligned with the paper defaults."""

    model_config = SettingsConfigDict(
        env_prefix="ANIMA_DEF_GHOSTFWL_",
        env_file=".env",
        extra="ignore",
    )

    project_name: str = "anima-def-ghostfwl"
    codename: str = "DEF-GHOSTFWL"
    functional_name: str = "DEF-ghostfwl"
    wave: int = 7
    paper_arxiv: str = "2603.28224"
    backend: Literal["auto", "mlx", "cuda", "cpu"] = "auto"
    precision: Literal["fp16", "bf16", "fp32"] = "fp32"

    data_root: Path = Field(default=Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets"))
    models_root: Path = Field(default=Path("/Volumes/AIFlowDev/RobotFlowLabs/models"))
    repos_root: Path = Field(default=Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/repos"))
    artifacts_root: Path = Field(
        default=Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/artifacts")
    )

    dataset_root: Path = Field(
        default=Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/ghost_dataset")
    )
    pretrain_root: Path = Field(
        default=Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/mae_dataset")
    )
    checkpoint_root: Path = Field(default=Path("/Volumes/AIFlowDev/RobotFlowLabs/models/ghost_fwl"))

    raw_shape_xyz: tuple[int, int, int] = (400, 512, 700)
    model_shape_hwt: tuple[int, int, int] = (128, 128, 256)
    crop_top_bins: int = 90
    crop_bottom_bins: int = 90
    crop_front_bins: int = 25
    pretrain_mask_ratio: float = 0.70

    train_scenes: tuple[str, ...] = ("001", "003", "004", "005", "006", "008", "010")
    test_scenes: tuple[str, ...] = ("002", "007", "009")

    @field_validator("raw_shape_xyz")
    @classmethod
    def validate_raw_shape(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if value != (400, 512, 700):
            raise ValueError("Ghost-FWL raw shape must remain (400, 512, 700)")
        return value

    @field_validator("model_shape_hwt")
    @classmethod
    def validate_model_shape(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if value != (128, 128, 256):
            raise ValueError("Model shape must remain (128, 128, 256)")
        return value

    @property
    def train_val_scenes(self) -> tuple[str, ...]:
        """Scenes used for train and validation in the paper split."""

        return self.train_scenes


def get_settings(**overrides: object) -> GhostFWLSettings:
    """Build settings with optional overrides for tests and scripts."""

    return GhostFWLSettings(**overrides)
