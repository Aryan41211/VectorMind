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
from pathlib import Path

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
