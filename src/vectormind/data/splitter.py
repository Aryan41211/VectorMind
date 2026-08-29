"""Train/val/test splitting with zero image leakage.

Purpose: split Flickr30k image-caption pairs into train/val/test sets
by IMAGE (not by caption) so that all 5 captions for a given image
stay in the same split — preventing information leakage between splits
(ROADMAP.md Phase 1 acceptance criteria).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vectormind.data.flickr_split import create_official_splits

logger = logging.getLogger(__name__)


def create_splits(
    config: dict[str, Any],
    image_paths: list[Path],
    captions: list[str],
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Split data into train/val/test by image, with zero leakage.

    Each unique image (and all its 5 captions) is assigned to exactly
    one split. Splitting is deterministic given the same random seed.

    Args:
        config: The full data config dict (from ``configs/data.yaml``).
            Must contain ``dataset.train_split``, ``dataset.val_split``,
            ``dataset.test_split``, and ``dataset.random_seed``.
        image_paths: List of image file paths (length N, where each
            image appears 5 times — once per caption).
        captions: List of caption strings (length N, one per path).

    Returns:
        A tuple ``(train_pairs, val_pairs, test_pairs)`` where each
        element is a list of ``(image_path, caption)`` tuples.

    Raises:
        ValueError: If split ratios do not sum to 1.0 (within
            floating-point tolerance), or if inputs are empty.

    Assumptions:
        - ``image_paths`` and ``captions`` have the same length.
        - Each image appears exactly 5 times consecutively (or at
          least the same number of times) — the function groups by
          unique image path, not by position.
        - The dataset is already loaded into memory (not lazy-loaded
          from disk at this point).

    Limitations:
        Uses a simple random split, not stratified. For Flickr30k's
        relatively uniform distribution this is fine; if a future
        dataset has severe class imbalance, a stratified split would
        be needed.
    """
    train_ratio = config["dataset"]["train_split"]
    val_ratio = config["dataset"]["val_split"]
    test_ratio = config["dataset"]["test_split"]
    seed = config["dataset"]["random_seed"]

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    if len(image_paths) == 0:
        raise ValueError("image_paths must be non-empty.")

    # Group by unique image — all 5 captions stay together.
    image_to_captions: dict[Path, list[str]] = {}
    for path, caption in zip(image_paths, captions):
        if path not in image_to_captions:
            image_to_captions[path] = []
        image_to_captions[path].append(caption)

    unique_images = list(image_to_captions.keys())
    num_images = len(unique_images)

    logger.info(
        "Splitting %d unique images (%d total pairs) with seed=%d",
        num_images,
        len(image_paths),
        seed,
    )

    # Deterministic shuffle using the configured seed.
    rng = __import__("random").Random(seed)
    shuffled = unique_images.copy()
    rng.shuffle(shuffled)

    # Compute split boundaries.
    n_train = int(num_images * train_ratio)
    n_val = int(num_images * val_ratio)
    # n_test = num_images - n_train - n_val (takes the remainder)

    train_images = shuffled[:n_train]
    val_images = shuffled[n_train : n_train + n_val]
    test_images = shuffled[n_train + n_val :]

    # Expand images back to (image_path, caption) pairs.
    def _expand(image_list: list[Path]) -> list[tuple[Path, str]]:
        pairs: list[tuple[Path, str]] = []
        for img in image_list:
            for cap in image_to_captions[img]:
                pairs.append((img, cap))
        return pairs

    train_pairs = _expand(train_images)
    val_pairs = _expand(val_images)
    test_pairs = _expand(test_images)

    # Verify zero leakage: no image in more than one split.
    train_image_set = set(train_images)
    val_image_set = set(val_images)
    test_image_set = set(test_images)

    assert train_image_set.isdisjoint(val_image_set), "Leakage: train/val overlap"
    assert train_image_set.isdisjoint(test_image_set), "Leakage: train/test overlap"
    assert val_image_set.isdisjoint(test_image_set), "Leakage: val/test overlap"

    logger.info(
        "Split complete — Train: %d images (%d pairs), "
        "Val: %d images (%d pairs), "
        "Test: %d images (%d pairs)",
        len(train_images),
        len(train_pairs),
        len(val_images),
        len(val_pairs),
        len(test_images),
        len(test_pairs),
    )

    return train_pairs, val_pairs, test_pairs


def create_splits_from_config(
    config: dict[str, Any],
    image_paths: list[Path],
    captions: list[str],
    image_ids: list[str] | None = None,
    split_files: dict[str, Path] | None = None,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Split into train/val/test according to ``config.dataset.split_mode``.

    Routes between the two split algorithms without each caller having
    to branch on the config itself:

    - ``"official"`` (default): the canonical Flickr30k 29,783 / 1,000 /
      1,000 split by Flickr image id (see :mod:`flickr_split`).
    - ``"random"``: the reproducible seeded ratio split in
      :func:`create_splits`.

    Args:
        config: The full data config dict; ``dataset.split_mode`` selects
            the algorithm and defaults to ``"official"`` when absent.
        image_paths: Image file paths, one per (image, caption) pair.
        captions: Caption strings corresponding to ``image_paths``.
        image_ids: Flickr image ids per pair, required in official mode.
        split_files: Mapping of split name to its official ``.txt`` file,
            used only in official mode.

    Returns:
        A tuple ``(train_pairs, val_pairs, test_pairs)``.

    Raises:
        ValueError: In official mode if ``image_ids`` is not provided, or
            an image's id matches no official split.

    Assumptions:
        ``config["dataset"]`` is present.
    """
    mode = config.get("dataset", {}).get("split_mode", "official")
    if mode != "official":
        return create_splits(config, image_paths, captions)

    if image_ids is None:
        raise ValueError(
            "split_mode == 'official' requires image_ids to be passed so "
            "images can be assigned to a split by Flickr id."
        )
    return create_official_splits(image_paths, captions, image_ids, split_files)
