"""Unit tests for vectormind.data.flickr_split.

The official Flickr30k train/val/test split (Gong et al. / Karpathy
convention, 29,783 / 1,000 / 1,000 images) is defined by the Flickr30k
Entities project's train.txt / val.txt / test.txt files, keyed by the
original Flickr image id. These tests cover loading those lists and
assigning cached images to a split by their Flickr id.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vectormind.data.flickr_split import (
    create_official_splits,
    load_official_split_lists,
)


@pytest.fixture
def split_files(tmp_path: Path) -> dict[str, Path]:
    """Write small official-style split lists into a temp directory."""
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    test = tmp_path / "test.txt"

    train.write_text("100\n101\n102\n103\n", encoding="utf-8")
    val.write_text("200\n201\n", encoding="utf-8")
    test.write_text("300\n301\n", encoding="utf-8")

    return {"train": train, "val": val, "test": test}


def test_load_official_split_lists_parses_ids(split_files: dict[str, Path]) -> None:
    """Each txt file should load into a set of Flickr ids."""
    splits = load_official_split_lists(split_files)

    assert splits["train"] == {"100", "101", "102", "103"}
    assert splits["val"] == {"200", "201"}
    assert splits["test"] == {"300", "301"}


def test_load_official_split_lists_ignores_blank_lines(tmp_path: Path) -> None:
    """Blank lines / trailing newlines should not become ids."""
    (tmp_path / "train.txt").write_text("100\n\n101\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("", encoding="utf-8")
    (tmp_path / "test.txt").write_text("", encoding="utf-8")

    splits = load_official_split_lists(
        {"train": tmp_path / "train.txt", "val": tmp_path / "val.txt", "test": tmp_path / "test.txt"}
    )

    assert splits["train"] == {"100", "101"}
    assert splits["val"] == set()
    assert splits["test"] == set()


@pytest.fixture
def corpus() -> tuple[list[Path], list[str], list[str]]:
    """8 images (each 5 captions) with Flickr ids spanning the splits."""
    image_ids = ["100", "101", "102", "103", "200", "201", "300", "301"]
    paths: list[Path] = []
    captions: list[str] = []
    ids: list[str] = []
    for i, img_id in enumerate(image_ids):
        for j in range(5):
            paths.append(Path(f"img_{i:04d}.jpg"))
            captions.append(f"Caption {j} for image {img_id}")
            ids.append(img_id)
    return paths, captions, ids


def test_create_official_splits_preserves_all_pairs(
    split_files: dict[str, Path], corpus: tuple[list[Path], list[str], list[str]]
) -> None:
    """Every (image, caption) pair must land in exactly one split."""
    paths, captions, ids = corpus
    train, val, test = create_official_splits(paths, captions, ids, split_files)

    assert len(train) + len(val) + len(test) == len(paths)


def test_create_official_splits_assigns_correctly(
    split_files: dict[str, Path], corpus: tuple[list[Path], list[str], list[str]]
) -> None:
    """Images must be assigned to the split their Flickr id is in."""
    paths, captions, ids = corpus
    train, val, test = create_official_splits(paths, captions, ids, split_files)

    train_imgs = {p for p, _ in train}
    val_imgs = {p for p, _ in val}
    test_imgs = {p for p, _ in test}

    # Images 0-3 (ids 100-103) are train; 4-5 (200-201) val; 6-7 (300-301) test.
    assert train_imgs == {Path(f"img_{i:04d}.jpg") for i in range(4)}
    assert val_imgs == {Path(f"img_{i:04d}.jpg") for i in (4, 5)}
    assert test_imgs == {Path(f"img_{i:04d}.jpg") for i in (6, 7)}


def test_create_official_splits_no_leakage(
    split_files: dict[str, Path], corpus: tuple[list[Path], list[str], list[str]]
) -> None:
    """No image may appear in more than one split."""
    paths, captions, ids = corpus
    train, val, test = create_official_splits(paths, captions, ids, split_files)

    train_imgs = {p for p, _ in train}
    val_imgs = {p for p, _ in val}
    test_imgs = {p for p, _ in test}

    assert train_imgs.isdisjoint(val_imgs)
    assert train_imgs.isdisjoint(test_imgs)
    assert val_imgs.isdisjoint(test_imgs)


def test_create_official_splits_keeps_captions_together(
    split_files: dict[str, Path], corpus: tuple[list[Path], list[str], list[str]]
) -> None:
    """All 5 captions of one image stay in the same split."""
    paths, captions, ids = corpus
    train, val, test = create_official_splits(paths, captions, ids, split_files)

    for split in (train, val, test):
        counts: dict[Path, int] = {}
        for p, _ in split:
            counts[p] = counts.get(p, 0) + 1
        assert set(counts.values()) == {5}


def test_create_official_splits_missing_id_raises(
    split_files: dict[str, Path],
) -> None:
    """An image whose id is in no official split must raise."""
    paths = [Path("a.jpg")] * 5
    captions = [f"cap {i}" for i in range(5)]
    ids = ["999999"] * 5  # not in any split list

    with pytest.raises(ValueError, match="no official split"):
        create_official_splits(paths, captions, ids, split_files)


def test_create_official_splits_mismatched_lengths_raise(
    split_files: dict[str, Path],
) -> None:
    """image_paths / captions / image_ids must all be the same length."""
    with pytest.raises(ValueError, match="must all"):
        create_official_splits(
            [Path("a.jpg")] * 5,
            ["c"] * 5,
            ["1"] * 4,  # wrong length
            split_files,
        )
