"""Structured logging and observability for Ghost-FWL runtime."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path


@dataclass
class InferenceEvent:
    """Structured log entry for a single inference call."""

    timestamp: float = field(default_factory=time.time)
    module: str = "def-ghostfwl"
    frame_id: str = ""
    latency_ms: float = 0.0
    ghost_count: int = 0
    total_points: int = 0
    threshold: float = 0.5
    device: str = "cpu"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class CheckpointFingerprint:
    """Unique identifier for a loaded checkpoint."""

    path: str = ""
    sha256: str = ""
    version: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> CheckpointFingerprint:
        p = Path(path)
        if not p.exists():
            return cls(path=str(p), sha256="MISSING", version="unknown")

        h = sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)

        return cls(path=str(p), sha256=h.hexdigest()[:16], version="unknown")


def get_logger(name: str = "anima.def_ghostfwl") -> logging.Logger:
    """Get a structured logger for Ghost-FWL operations."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class DegradationMonitor:
    """Tracks inference quality and alerts on degradation."""

    def __init__(self, *, window_size: int = 100) -> None:
        self._ghost_rates: list[float] = []
        self._window_size = window_size
        self._logger = get_logger("anima.def_ghostfwl.degradation")

    def record(self, ghost_count: int, total_count: int) -> None:
        if total_count == 0:
            return
        rate = ghost_count / total_count
        self._ghost_rates.append(rate)
        if len(self._ghost_rates) > self._window_size:
            self._ghost_rates.pop(0)

    @property
    def mean_ghost_rate(self) -> float:
        if not self._ghost_rates:
            return 0.0
        return sum(self._ghost_rates) / len(self._ghost_rates)

    def check_degradation(self, *, max_ghost_rate: float = 0.5) -> bool:
        """Returns True if ghost rate exceeds threshold."""
        if self.mean_ghost_rate > max_ghost_rate:
            self._logger.warning(
                "Ghost rate degradation: %.3f > %.3f threshold",
                self.mean_ghost_rate,
                max_ghost_rate,
            )
            return True
        return False
