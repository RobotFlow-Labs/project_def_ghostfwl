"""FastAPI application for Ghost-FWL denoising service."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from anima_def_ghostfwl.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from anima_def_ghostfwl.api.service import DenoiseService
from anima_def_ghostfwl.version import __version__

_service = DenoiseService(
    checkpoint_path=Path(os.environ.get("ANIMA_CHECKPOINT_PATH", "")),
    device=os.environ.get("ANIMA_DEVICE", "cpu"),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    checkpoint = os.environ.get("ANIMA_CHECKPOINT_PATH")
    if checkpoint and Path(checkpoint).exists():
        _service.load()
    yield


app = FastAPI(
    title="Ghost-FWL Denoising API",
    description="LiDAR ghost object detection and removal",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        module="def-ghostfwl",
        version=__version__,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    resp = ReadyResponse(
        ready=_service.is_ready,
        module="def-ghostfwl",
        version=__version__,
        weights_loaded=_service.is_ready,
    )
    if not resp.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return resp


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    if not _service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.voxel_path:
        raise HTTPException(status_code=400, detail="voxel_path is required")

    voxel_file = Path(request.voxel_path)
    if not voxel_file.exists():
        raise HTTPException(status_code=404, detail=f"Voxel file not found: {voxel_file}")

    volume = np.load(str(voxel_file))
    return _service.run(volume, threshold=request.threshold)
