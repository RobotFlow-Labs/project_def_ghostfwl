"""Canonical train/val/test scene splits from appendix D.2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneSplit:
    """Immutable scene assignment from the paper."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


PAPER_SPLIT = SceneSplit(
    train=("001", "003", "004", "005", "006", "008", "010"),
    val=("001", "003", "004", "005", "006", "008", "010"),
    test=("002", "007", "009"),
)


def get_scene_split(*, custom_test: tuple[str, ...] | None = None) -> SceneSplit:
    """Return the paper split, optionally overriding test scenes."""
    if custom_test is None:
        return PAPER_SPLIT
    all_scenes = {"001", "002", "003", "004", "005", "006", "007", "008", "009", "010"}
    remaining = sorted(all_scenes - set(custom_test))
    return SceneSplit(
        train=tuple(remaining),
        val=tuple(remaining),
        test=custom_test,
    )
