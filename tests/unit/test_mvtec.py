
from pathlib import Path

import pytest

from anomaly.data.mvtec import list_samples


def _touch_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture
def fake_category(tmp_path):
    cat = tmp_path / "widget"
    cat.mkdir()

    _touch_png(cat / "train/good/000.png")
    _touch_png(cat / "train/good/001.png")
    _touch_png(cat / "test/good/000.png")
    _touch_png(cat / "test/crack/000.png")
    _touch_png(cat / "ground_truth/crack/000_mask.png")

    return cat


def test_counts_and_labels(fake_category):
    samples = list_samples(fake_category)

    assert len(samples) == 4

    assert [s.label for s in samples if s.split == "train"] == [0, 0]

    assert [s.label for s in samples if s.split == "test"] == [1, 0]

    crack = [s for s in samples if s.defect_type == "crack"]

    assert len(crack) == 1
    assert crack[0].label == 1
    assert crack[0].mask_path.name == "000_mask.png"

def test_order_is_deterministic(fake_category):
    samples = list_samples(fake_category)

    paths = [
        s.path.relative_to(fake_category).as_posix()
        for s in samples
    ]

    assert paths == [
        "train/good/000.png",
        "train/good/001.png",
        "test/crack/000.png",
        "test/good/000.png",
    ]
def test_missing_mask_fails_loudly(fake_category):
    mask_path = fake_category / "ground_truth/crack/000_mask.png"

    mask_path.unlink()

    with pytest.raises(FileNotFoundError):
        list_samples(fake_category)
