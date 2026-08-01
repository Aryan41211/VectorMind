"""Shared data loading utilities for scripts.

Purpose: provide a single source of truth for the Flickr30k HuggingFace
loading logic used by acceptance test scripts (smoke_test_model.py,
test_train_loop.py).

This is an entry-point helper module, NOT part of the core library.
It is only imported by scripts/ and never by src/vectormind/.

Usage:
    from scripts._data_helpers import load_flickr30k_from_hf
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_flickr30k_from_hf(cache_dir: str) -> tuple[list[str], list[str]]:
    """Load Flickr30k from HuggingFace Datasets.

    Downloads (on first call) and caches images under ``cache_dir/images/``.
    Each example is expanded into 5 (image_path, caption) pairs — one per
    caption.

    Args:
        cache_dir: Root directory for cached dataset files.

    Returns:
        A tuple (image_paths, captions) with each caption paired to
        its image path. Each image appears 5 times (once per caption).

    Raises:
        ImportError: If ``datasets`` is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required for downloading Flickr30k. "
            "Install it with: pip install datasets"
        )
        raise

    logger.info("Loading Flickr30k from HuggingFace (this may take a while)...")
    ds = load_dataset("nlphuji/flickr30k", cache_dir=cache_dir, trust_remote_code=True)

    image_paths: list[str] = []
    captions: list[str] = []

    for split_name in ["test", "train", "validation"]:
        if split_name in ds:
            for example in ds[split_name]:
                image = example["image"]
                caption_list = example["caption"]

                idx = len(image_paths) // 5
                img_path = Path(cache_dir) / "images" / f"{idx:06d}.jpg"
                img_path.parent.mkdir(parents=True, exist_ok=True)

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
