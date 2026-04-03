"""Tests for Ghost-FWL scene split definitions."""

from anima_def_ghostfwl.eval.splits import PAPER_SPLIT, get_scene_split


def test_paper_split_scene_counts() -> None:
    assert len(PAPER_SPLIT.train) == 7
    assert len(PAPER_SPLIT.test) == 3


def test_paper_split_test_scenes() -> None:
    assert PAPER_SPLIT.test == ("002", "007", "009")


def test_paper_split_train_scenes() -> None:
    assert PAPER_SPLIT.train == ("001", "003", "004", "005", "006", "008", "010")


def test_custom_split_overrides_test() -> None:
    split = get_scene_split(custom_test=("001", "002"))
    assert split.test == ("001", "002")
    assert "001" not in split.train
    assert "002" not in split.train


def test_default_split_matches_paper() -> None:
    split = get_scene_split()
    assert split == PAPER_SPLIT
