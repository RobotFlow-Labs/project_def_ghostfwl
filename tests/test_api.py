"""Contract tests for Ghost-FWL FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import anima_def_ghostfwl.api.app as app_mod
from anima_def_ghostfwl.api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["module"] == "def-ghostfwl"


def test_ready_503_when_no_model(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_predict_503_when_no_model(client: TestClient) -> None:
    resp = client.post("/predict", json={"voxel_path": "/tmp/fake.npy"})
    assert resp.status_code == 503


def test_predict_400_when_no_path(client: TestClient) -> None:
    original = app_mod._service._predictor
    app_mod._service._predictor = "fake"
    try:
        resp = client.post("/predict", json={})
        assert resp.status_code == 400
    finally:
        app_mod._service._predictor = original


def test_predict_404_when_missing_file(client: TestClient) -> None:
    original = app_mod._service._predictor
    app_mod._service._predictor = "fake"
    try:
        resp = client.post("/predict", json={"voxel_path": "/tmp/nonexistent_vol.npy"})
        assert resp.status_code == 404
    finally:
        app_mod._service._predictor = original
