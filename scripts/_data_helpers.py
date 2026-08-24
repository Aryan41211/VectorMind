"""Shared data loading utilities for scripts.

Purpose: provide a single source of truth for the Flickr30k HuggingFace
loading logic used by acceptance test scripts (smoke_test_model.py,
test_train_loop.py, verify_dataset.py).

Uses ``lmms-lab/flickr30k`` (Parquet format, compatible with
datasets>=5.0). The original ``nlphuji/flickr30k`` uses a deprecated
loading script that is no longer supported.

This is an entry-point helper module, NOT part of the core library.
It is only imported by scripts/ and never by src/vectormind/.

Usage:
    from scripts._data_helpers import load_flickr30k_from_hf
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vectormind.data.dataloader import create_dataloaders
from vectormind.data.splitter import create_splits
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms, get_train_transforms

logger = logging.getLogger(__name__)

# Dataset configuration
_HF_DATASET_ID: str = "lmms-lab/flickr30k"
_HF_SPLIT: str = "test"  # Contains all 31,783 images


def load_flickr30k_from_hf(cache_dir: str) -> tuple[list[str], list[str]]:
    """Load Flickr30k from HuggingFace Datasets.

    Downloads (on first call) and caches images under ``cache_dir/images/``.
    Each example is expanded into 5 (image_path, caption) pairs — one per
    caption.

    Uses ``lmms-lab/flickr30k`` which stores the full dataset in Parquet
    format and is compatible with datasets>=5.0.

    Args:
        cache_dir: Root directory for cached dataset files.

    Returns:
        A tuple (image_paths, captions) with each caption paired to
        its image path. Each image appears 5 times (once per caption).

    Raises:
        ImportError: If ``datasets`` is not installed (only when cache
            is missing and download is needed).
    """
    images_dir = Path(cache_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    captions: list[str] = []

    # Check if images and captions are already cached
    existing_images = sorted(images_dir.glob("*.jpg"))
    captions_file = Path(cache_dir) / "captions.json"

    if len(existing_images) >= 31700 and captions_file.exists():
        # Full cache — load from disk
        import json

        logger.info(
            "Found %d cached images and captions.json — loading from cache",
            len(existing_images),
        )
        with open(captions_file, encoding="utf-8") as f:
            captions_data = json.load(f)

        for entry in captions_data:
            img_path = entry["image_path"]
            for cap in entry["captions"]:
                image_paths.append(img_path)
                captions.append(cap)

        logger.info(
            "Loaded %d pairs (%d unique images) from cache",
            len(image_paths),
            len(existing_images),
        )
        return image_paths, captions

    # Download from HuggingFace (only reached if cache is missing)
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required for downloading Flickr30k. "
            "Install it with: pip install datasets"
        )
        raise

    # Download from HuggingFace
    logger.info(
        "Loading Flickr30k from HuggingFace (%s) — this may take a while...",
        _HF_DATASET_ID,
    )
    ds = load_dataset(_HF_DATASET_ID, split=_HF_SPLIT, streaming=True)

    for example in ds:
        image = example["image"]
        caption_list = example["caption"]

        idx = len(image_paths) // 5
        img_path = images_dir / f"{idx:06d}.jpg"

        if not img_path.exists():
            image.save(img_path)

        for cap in caption_list:
            image_paths.append(str(img_path))
            captions.append(cap)

    logger.info(
        "Loaded %d pairs (%d unique images) from Flickr30k",
        len(image_paths),
        len(image_paths) // 5,
    )
    return image_paths, captions


def build_eval_pairs(
    data_config: dict[str, object],
) -> dict[str, list[tuple[Path, str]]]:
    """Return the (image_path, caption) pairs behind the eval splits.

    The dataloaders hand back tensors, which is all a metric needs and
    not enough for a qualitative report — that needs the caption text
    and the file the image came from, in the same order the loader
    yields them.

    Args:
        data_config: Parsed ``configs/data.yaml``.

    Returns:
        Mapping of split name (``"val"``, ``"test"``) to its ordered
        list of ``(image_path, caption)`` pairs.

    Assumptions:
        Splitting is deterministic given the config's random seed, so
        these pairs line up row-for-row with what
        :func:`build_eval_loaders` produces from the same config.
    """
    cfg = _dataset_copy(data_config, batch_size=None)
    image_paths, captions = load_flickr30k_from_hf(cfg["dataset"]["local_cache_dir"])
    _, val_pairs, test_pairs = create_splits(
        config=cfg,
        image_paths=[Path(p) for p in image_paths],
        captions=captions,
    )
    return {"val": val_pairs, "test": test_pairs}


def _dataset_copy(
    data_config: dict[str, object], batch_size: int | None
) -> dict[str, Any]:
    """Shallow-copy a data config, optionally overriding the batch size.

    Args:
        data_config: Parsed ``configs/data.yaml``.
        batch_size: Replacement batch size, or None to keep the config's.

    Returns:
        A copy safe to mutate; the caller's config is left untouched.
    """
    cfg: dict[str, Any] = dict(data_config)
    cfg["dataset"] = dict(cfg["dataset"])
    if batch_size is not None:
        cfg["dataset"]["batch_size"] = batch_size
    return cfg


def build_eval_loaders(
    data_config: dict[str, Any],
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Build the val and test dataloaders described by ``data_config``.

    Why it is here: every script that evaluates a checkpoint needs the
    same loaders built the same way, and four scripts each owning a copy
    is what let four scripts report different numbers for one checkpoint
    (docs/KNOWN_ISSUES.md §9).

    Args:
        data_config: Parsed ``configs/data.yaml``.
        batch_size: Optional override for the evaluation batch size,
            for machines with less VRAM than the training run had.

    Returns:
        Mapping of split name (``"val"``, ``"test"``) to its DataLoader.

    Assumptions:
        The Flickr30k image cache already exists under
        ``dataset.local_cache_dir``; otherwise this downloads it.

    Limitations:
        The train loader is built and discarded — ``create_splits`` and
        ``create_dataloaders`` produce all three together, and the split
        boundaries depend on the train fraction, so it cannot be
        skipped without changing which images land in val and test.
    """
    cfg = _dataset_copy(data_config, batch_size)

    image_paths, captions = load_flickr30k_from_hf(cfg["dataset"]["local_cache_dir"])
    train_pairs, val_pairs, test_pairs = create_splits(
        config=cfg,
        image_paths=[Path(p) for p in image_paths],
        captions=captions,
    )

    tokenizer = CaptionTokenizer(
        tokenizer_name=cfg["dataset"]["tokenizer_name"],
        max_length=cfg["dataset"]["max_text_length"],
    )
    _, val_loader, test_loader = create_dataloaders(
        config=cfg,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=get_train_transforms(cfg),
        eval_transform=get_eval_transforms(cfg),
        tokenizer=tokenizer,
    )
    return {"val": val_loader, "test": test_loader}
