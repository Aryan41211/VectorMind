"""Unit tests for vectormind.data.splitter."""

from __future__ import annotations

from pathlib import Path

import pytest

from vectormind.data.splitter import create_splits


@pytest.fixture
def data_config() -> dict:
    """Minimal config for splitter tests."""
    return {
        "dataset": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "random_seed": 42,
        },
    }


@pytest.fixture
def sample_data() -> tuple[list[Path], list[str]]:
    """20 images, each with 5 captions (100 pairs total)."""
    paths: list[Path] = []
    captions: list[str] = []
    for i in range(20):
        for j in range(5):
            paths.append(Path(f"img_{i:04d}.jpg"))
            captions.append(f"Caption {j} for image {i}")
    return paths, captions


def test_split_ratios_approximately_correct(
    data_config: dict, sample_data: tuple[list[Path], list[str]]
) -> None:
    """Splits should approximately match configured ratios."""
    paths, captions = sample_data
    train, val, test = create_splits(data_config, paths, captions)

    total_images = 20
    assert abs(len(train) / 5 - total_images * 0.8) <= 1
    assert abs(len(val) / 5 - total_images * 0.1) <= 1
    assert abs(len(test) / 5 - total_images * 0.1) <= 1


def test_zero_leakage(
    data_config: dict, sample_data: tuple[list[Path], list[str]]
) -> None:
    """No image should appear in more than one split."""
    paths, captions = sample_data
    train, val, test = create_splits(data_config, paths, captions)

    train_images = {p for p, _ in train}
    val_images = {p for p, _ in val}
    test_images = {p for p, _ in test}

    assert train_images.isdisjoint(val_images)
    assert train_images.isdisjoint(test_images)
    assert val_images.isdisjoint(test_images)


def test_all_pairs_preserved(
    data_config: dict, sample_data: tuple[list[Path], list[str]]
) -> None:
    """Total pairs across splits should equal original count."""
    paths, captions = sample_data
    train, val, test = create_splits(data_config, paths, captions)

    assert len(train) + len(val) + len(test) == len(paths)


def test_deterministic_with_same_seed(
    data_config: dict, sample_data: tuple[list[Path], list[str]]
) -> None:
    """Same seed should produce the same split."""
    paths, captions = sample_data
    t1, v1, te1 = create_splits(data_config, paths, captions)
    t2, v2, te2 = create_splits(data_config, paths, captions)

    assert [(p, c) for p, c in t1] == [(p, c) for p, c in t2]
    assert [(p, c) for p, c in v1] == [(p, c) for p, c in v2]
    assert [(p, c) for p, c in te1] == [(p, c) for p, c in te2]


def test_different_seeds_produce_different_splits(
    sample_data: tuple[list[Path], list[str]]
) -> None:
    """Different seeds should (almost certainly) produce different splits."""
    paths, captions = sample_data
    config_a = {
        "dataset": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "random_seed": 42,
        }
    }
    config_b = {
        "dataset": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "random_seed": 123,
        }
    }

    t1, _, _ = create_splits(config_a, paths, captions)
    t2, _, _ = create_splits(config_b, paths, captions)

    # The sets of images should differ (very high probability).
    imgs1 = {p for p, _ in t1}
    imgs2 = {p for p, _ in t2}
    assert imgs1 != imgs2


def test_raises_on_invalid_ratios() -> None:
    """Split ratios that don't sum to 1.0 should raise ValueError."""
    config = {
        "dataset": {
            "train_split": 0.5,
            "val_split": 0.3,
            "test_split": 0.3,
            "random_seed": 42,
        }
    }
    with pytest.raises(ValueError, match="must sum to 1.0"):
        create_splits(config, [Path("a.jpg")], ["caption"])


def test_raises_on_empty_data() -> None:
    """Empty inputs should raise ValueError."""
    config = {
        "dataset": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "random_seed": 42,
        }
    }
    with pytest.raises(ValueError, match="non-empty"):
        create_splits(config, [], [])


def test_five_captions_per_image_stay_together(
    data_config: dict,
) -> None:
    """All 5 captions for a given image should end up in the same split."""
    paths = [Path(f"img_{i:04d}.jpg") for i in range(10) for _ in range(5)]
    captions = [f"Caption {j} for image {i}" for i in range(10) for j in range(5)]

    train, val, test = create_splits(data_config, paths, captions)

    # Check that no image appears in two splits.
    train_imgs = {p for p, _ in train}
    val_imgs = {p for p, _ in val}
    test_imgs = {p for p, _ in test}

    assert train_imgs.isdisjoint(val_imgs)
    assert train_imgs.isdisjoint(test_imgs)
    assert val_imgs.isdisjoint(test_imgs)

    # Check that each image's 5 captions all end up in the same split.
    for img_path in train_imgs:
        assert sum(1 for p, _ in train if p == img_path) == 5
    for img_path in val_imgs:
        assert sum(1 for p, _ in val if p == img_path) == 5
    for img_path in test_imgs:
        assert sum(1 for p, _ in test if p == img_path) == 5
