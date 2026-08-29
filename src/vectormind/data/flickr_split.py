"""Official Flickr30k train/val/test split by Flickr image id.

Purpose: assign cached Flickr30k images to the canonical train/val/test
split (Gong et al. convention: 29,783 train / 1,000 val / 1,000 test)
that the Flickr30k Entities project defines in its train.txt / val.txt
/ test.txt files. The lists are keyed by the original Flickr image id,
which ``_data_helpers.load_flickr30k_from_hf`` records in the cache, so
here we only need to know each local image's id plus the on-disk lists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

# Canonical source of the official split lists (BryanPlummer/flickr30k_entities).
# The lists are small text files; the images themselves are not here.
_TRAIN_URL: str = (
    "https://raw.githubusercontent.com/bryanplummer/flickr30k_entities/master/train.txt"
)
_VAL_URL: str = (
    "https://raw.githubusercontent.com/bryanplummer/flickr30k_entities/master/val.txt"
)
_TEST_URL: str = (
    "https://raw.githubusercontent.com/bryanplummer/flickr30k_entities/master/test.txt"
)

_URLS: dict[str, str] = {"train": _TRAIN_URL, "val": _VAL_URL, "test": _TEST_URL}

# Official split sizes, used for a hard sanity check on load.
_EXPECTED_SIZES: dict[str, int] = {"train": 29783, "val": 1000, "test": 1000}

# Filenames the split lists are cached under, relative to a cache dir.
_CACHE_NAMES: dict[str, str] = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}


def _read_ids(path: Path) -> set[str]:
    """Read a Flickr id per non-blank line from `path`.

    Args:
        path: The split list file.

    Returns:
        A set of Flickr image ids (strings), one per non-blank line.
    """
    ids: set[str] = set()
    if not path.is_file():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(line.strip())
    return ids


def load_official_split_lists(
    split_files: Mapping[str, Path] | None = None,
) -> dict[str, set[str]]:
    """Load the official split id sets from on-disk list files.

    Args:
        split_files: Optional mapping of split name (``"train"``,
            ``"val"``, ``"test"``) to the local ``.txt`` file for that
            split. If omitted, every split is treated as empty.
            Missing keys are treated as empty sets.

    Returns:
        A mapping of split name to the set of Flickr image ids in it.

    Assumptions:
        The files contain one Flickr id per line.

    Limitations:
        No sanity check on file size here so the function stays pure
        and testable with small fixtures; ``fetch_official_split_lists``
        enforces the documented official sizes on the download path.
    """
    splits: dict[str, set[str]] = {}
    sources: Mapping[str, Path] = split_files or {}
    for name in ("train", "val", "test"):
        ids = _read_ids(sources[name]) if name in sources else set()
        splits[name] = ids
    return splits


def create_official_splits(
    image_paths: list[Path],
    captions: list[str],
    image_ids: list[str],
    split_files: dict[str, Path],
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Assign (image, caption) pairs to splits by official Flickr id.

    Assumes each image appears once per caption (e.g. 5 times), so a
    single image's pairs all share one Flickr id and must land together.

    Args:
        image_paths: Image file paths, one per (image, caption) pair.
        captions: Caption strings corresponding to ``image_paths``.
        image_ids: Flickr image id per pair, parallel to the others.
        split_files: Mapping of split name to its official ``.txt`` file.

    Returns:
        A tuple ``(train_pairs, val_pairs, test_pairs)`` where each
        element is a list of ``(image_path, caption)`` pairs.

    Raises:
        ValueError: If the three input lists differ in length, or any
            image's Flickr id is absent from all three official splits.

    Assumptions:
        Every image's Flickr id resolves to exactly one official split.

    Limitations:
        Duplicate/missing ids across the lists are not reconciled here;
        each image is assigned to the first split that contains its id.
    """
    if not (len(image_paths) == len(captions) == len(image_ids)):
        raise ValueError(
            "image_paths, captions, and image_ids must all have the "
            f"same length (got {len(image_paths)}, {len(captions)}, "
            f"{len(image_ids)})."
        )
    if not image_paths:
        raise ValueError("image_paths must be non-empty.")

    splits = load_official_split_lists(split_files)

    # Group by unique image path; capture the Flickr id for that image.
    image_to_id: dict[Path, str] = {}
    image_to_captions: dict[Path, list[str]] = {}
    for path, caption, img_id in zip(image_paths, captions, image_ids):
        image_to_id.setdefault(path, img_id)
        image_to_captions.setdefault(path, []).append(caption)

    # Assign each image's id to a split, then expand back to pairs.
    assignment: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for image, img_id in image_to_id.items():
        target: str | None = None
        for name in ("train", "val", "test"):
            if img_id in splits[name]:
                target = name
                break
        if target is None:
            raise ValueError(
                f"Image {image} has Flickr id {img_id} that appears in "
                f"no official split list. Check that this image is part "
                f"of the official Flickr30k corpus."
            )
        assignment[target].append(image)

    def _expand(name: str) -> list[tuple[Path, str]]:
        pairs: list[tuple[Path, str]] = []
        for img in assignment[name]:
            for cap in image_to_captions[img]:
                pairs.append((img, cap))
        return pairs

    train_pairs = _expand("train")
    val_pairs = _expand("val")
    test_pairs = _expand("test")

    logger.info(
        "Official split assigned — Train: %d images, Val: %d images, "
        "Test: %d images (%d total pairs)",
        len(assignment["train"]),
        len(assignment["val"]),
        len(assignment["test"]),
        len(train_pairs) + len(val_pairs) + len(test_pairs),
    )
    return train_pairs, val_pairs, test_pairs


def fetch_official_split_lists(cache_dir: str | Path) -> dict[str, Path]:
    """Download and cache the official split list files.

    The files live in the canonical GitHub repo; they are small text
    files of Flickr ids, not the images. Downloads idempotently: an
    already-cached, correct-size file is reused.

    Args:
        cache_dir: Directory under which to store the split files
            (the files are placed directly in this directory).

    Returns:
        A mapping of split name to the local cached ``.txt`` path.

    Raises:
        RuntimeError: If a file cannot be fetched.
    """
    from urllib.error import URLError
    from urllib.request import urlopen

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for name, url in _URLS.items():
        target = cache / _CACHE_NAMES[name]
        try:
            if not target.is_file():
                with urlopen(url, timeout=30) as resp:
                    body = resp.read().decode("utf-8")
                target.write_text(body, encoding="utf-8")
                logger.info("Downloaded %s split list to %s", name, target)
            parsed = _read_ids(target)
            if not parsed or len(parsed) != _EXPECTED_SIZES[name]:
                raise ValueError(
                    f"Official {name} split list has {len(parsed)} ids, "
                    f"expected {_EXPECTED_SIZES[name]}. Refusing to use "
                    f"a wrong or truncated file."
                )
        except (URLError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not fetch the official {name} split list from "
                f"{url}: {exc}"
            ) from exc
        result[name] = target
    return result
