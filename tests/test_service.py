"""Tests for the DenoiseService wiring."""

from __future__ import annotations

import pytest

from anima_def_ghostfwl.api.service import DenoiseService


def test_service_not_ready_initially() -> None:
    svc = DenoiseService()
    assert svc.is_ready is False


def test_service_load_raises_without_path() -> None:
    svc = DenoiseService()
    with pytest.raises(ValueError, match="No checkpoint"):
        svc.load()


def test_service_run_raises_when_not_ready() -> None:
    import numpy as np

    svc = DenoiseService()
    with pytest.raises(RuntimeError, match="not ready"):
        svc.run(np.zeros((2, 2, 2)))
