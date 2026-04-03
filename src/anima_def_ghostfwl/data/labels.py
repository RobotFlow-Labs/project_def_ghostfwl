"""Paper-faithful label definitions from Ghost-FWL §3.3 and constants.py."""

from __future__ import annotations

LABEL_NAME_TO_ID: dict[str, int] = {
    "noise": 0,
    "object": 1,
    "glass": 2,
    "ghost": 3,
}

LABEL_ID_TO_NAME: dict[int, str] = {v: k for k, v in LABEL_NAME_TO_ID.items()}

NUM_CLASSES: int = len(LABEL_NAME_TO_ID)
