"""Request and response models for the Ghost-FWL API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Denoising request payload."""

    voxel_path: str | None = Field(
        default=None,
        description="Path to a .npy or .b2 voxel volume on the server filesystem",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for ghost classification",
    )
    return_mask: bool = Field(
        default=False,
        description="Whether to include the ghost mask in the response",
    )


class ClassSummary(BaseModel):
    """Per-class voxel counts from the prediction."""

    noise: int = 0
    object: int = 0
    glass: int = 0
    ghost: int = 0


class PredictResponse(BaseModel):
    """Denoising response payload."""

    denoised_points_count: int = Field(description="Number of surviving points")
    ghost_points_removed: int = Field(description="Number of ghost points removed")
    class_summary: ClassSummary = Field(default_factory=ClassSummary)
    output_path: str | None = Field(
        default=None,
        description="Path to the saved denoised output",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    module: str = "def-ghostfwl"
    version: str = ""
    gpu_available: bool = False


class ReadyResponse(BaseModel):
    """Readiness probe response."""

    ready: bool = False
    module: str = "def-ghostfwl"
    version: str = ""
    weights_loaded: bool = False
