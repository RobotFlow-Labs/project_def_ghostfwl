"""Tests for ANIMA serve node."""

from __future__ import annotations

import numpy as np
import pytest

from anima_def_ghostfwl.serve import GhostFWLServeNode


def test_serve_node_initial_status() -> None:
    node = GhostFWLServeNode()
    status = node.get_status()
    assert status["model_loaded"] is False
    assert status["frames_processed"] == 0


def test_serve_node_process_raises_without_setup() -> None:
    node = GhostFWLServeNode()
    with pytest.raises(RuntimeError, match="not loaded"):
        node.process(np.zeros((4, 4, 4)))


def test_serve_node_setup_raises_without_checkpoint() -> None:
    node = GhostFWLServeNode()
    with pytest.raises(FileNotFoundError):
        node.setup_inference()
