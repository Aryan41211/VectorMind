"""Shared data loading utilities for scripts.

Purpose: provide a single source of truth for the Flickr30k HuggingFace
loading logic used by acceptance and dataset scripts
(smoke_test_model.py, verify_dataset.py, train.py).

Uses ``lmms-lab/flickr30k`` (Parquet format, compatible with
datasets>=5.0). The original ``nlphuji/flickr30k`` uses a deprecated
loading script that is no longer supported.

This is an entry-point helper module, NOT part of the core library.
It is only imported by scripts/ and never by src/vectormind/.

Usage:
    from scripts._data_helpers import load_flickr30k_from_hf
    from scripts._data_helpers import load_flickr30k_with_ids
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

# Corpus constants (DATASETS.md). These are named so no magic numbers
# appear inline and so the cache-completeness gate is explicit.
_EXPECTED_IMAGES: int = 31783
_CAPTIONS_PER_IMAGE: int = 5
_EXPECTED_PAIRS: int = _EXPECTED_IMAGES * _CAPTIONS_PER_IMAGE
_CAPTIONS_FILE: str = "captions.json"


def _corpus_complete(images_dir: Path, captions_file: Path) -> bool:
    """Return True if the on-disk cache holds the full corpus.

    A cache is only usable when it has every image *and* the caption
    index file. Previously this checked ``>= 31700`` images, silently
    accepting a cache that could be missing up to 83 images; the gate
    is now that the *complete* corpus is present so nothing is dropped
    from training or evaluation.

    Args:
        images_dir: Directory containing the cached ``*.jpg`` files.
        captions_file: Path to the captions index (``captions.json``).

    Returns:
        True if the cache is complete, False otherwise.
    """
    if not captions_file.is_file():
        return False
    image_count = len(sorted(images_dir.glob("*.jpg")))
    if image_count < _EXPECTED_IMAGES:
        logger.warning(
            "Cache has %d images, expected %d — will fetch the rest",
            image_count,
            _EXPECTED_IMAGES,
        )
        return False
    return True


def _has_image_ids(captions_data: list[dict[str, Any]]) -> bool:
    """Return True if every cache entry carries a Flickr image id.

    Args:
        captions_data: Parsed ``captions.json`` entries.

    Returns:
        True if entries are non-empty and all contain an ``img_id``.
    """
    return bool(captions_data) and all("img_id" in entry for entry in captions_data)


def _migrate_missing_image_ids(cache_dir: Path) -> None:
    """Backfill Flickr image ids into a cache that predates id capture.

    Older caches stored only ``image_path`` and ``captions``. The Flickr
    image id for image slot ``i`` equals row ``i``'s ``filename`` in the
    dataset (images are cached as ``000000.jpg`` ... in row order), so we
    stream just that column from the local HF cache to assign ids. This
    runs once; afterwards the cache carries ids and this is a no-op.

    Args:
        cache_dir: Root directory of the cached dataset.
    """
    captions_file = cache_dir / _CAPTIONS_FILE
    import json

    with open(captions_file, encoding="utf-8") as f:
        captions_data = json.load(f)

    if _has_image_ids(captions_data):
        return

    logger.info("Backfilling Flickr image ids into an older cache...")
    from datasets import load_dataset

    ds = load_dataset(_HF_DATASET_ID, split=_HF_SPLIT, streaming=True)
    updated: list[dict[str, Any]] = []
    for idx, example in enumerate(ds):
        filename = example.get("filename") or f"{idx:06d}.jpg"
        flickr_id = Path(filename).stem
        if idx < len(captions_data):
            entry = dict(captions_data[idx])
            entry["img_id"] = flickr_id
            entry["filename"] = filename
            updated.append(entry)
        else:
            break

    if len(updated) != len(captions_data):
        raise RuntimeError(
            f"ID backfill produced {len(updated)} entries, expected "
            f"{len(captions_data)}. Refusing to overwrite the cache."
        )
    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    logger.info("Backfilled image ids for %d images", len(updated))


def load_flickr30k_with_ids(
    cache_dir: str, img_id_key: str = "img_id"
) -> tuple[list[str], list[str], list[str]]:
    """Load Flickr30k from HuggingFace, returning Flickr image ids too.

    Downloads (on first call) and caches images under ``cache_dir/images/``.
    Each example is expanded into 5 (image_path, caption) pairs — one per
    caption — and the original Flickr image id is carried alongside so the
    caller can apply the official train/val/test split by id.

    If the cache exists but predates id capture, ids are backfilled from
    the local HF cache before returning.

    Args:
        cache_dir: Root directory for cached dataset files.
        img_id_key: Cache key under which each entry stores its Flickr id.

    Returns:
        A tuple (image_paths, captions, image_ids) each of length
        ``_EXPECTED_PAIRS`` (every image 5 times, once per caption).

    Raises:
        ImportError: If ``datasets`` is not installed (only when cache
            is missing and download is needed).
    """
    images_dir = Path(cache_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_file = Path(cache_dir) / _CAPTIONS_FILE

    image_paths: list[str] = []
    captions: list[str] = []
    image_ids: list[str] = []

    if _corpus_complete(images_dir, captions_file):
        import json

        _migrate_missing_image_ids(Path(cache_dir))
        with open(captions_file, encoding="utf-8") as f:
            captions_data = json.load(f)

        for entry in captions_data:
            img_path = entry["image_path"]
            flickr_id = entry[img_id_key]
            for cap in entry["captions"]:
                image_paths.append(img_path)
                captions.append(cap)
                image_ids.append(flickr_id)

        logger.info(
            "Loaded %d pairs (%d unique images) from cache",
            len(image_paths),
            len(captions_data),
        )
        return image_paths, captions, image_ids

    # Download from HuggingFace (only reached if the cache is missing).
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required for downloading Flickr30k. "
            "Install it with: pip install datasets"
        )
        raise

    logger.info(
        "Loading Flickr30k from HuggingFace (%s) — this may take a while...",
        _HF_DATASET_ID,
    )
    ds = load_dataset(_HF_DATASET_ID, split=_HF_SPLIT, streaming=True)
    cached_entries: list[dict[str, Any]] = []

    for idx, example in enumerate(ds):
        image = example["image"]
        caption_list = example["caption"]
        filename = example.get("filename") or f"{idx:06d}.jpg"
        flickr_id = Path(filename).stem

        img_path = images_dir / f"{idx:06d}.jpg"
        if not img_path.exists():
            image.save(img_path)

        for cap in caption_list:
            image_paths.append(str(img_path))
            captions.append(cap)
            image_ids.append(flickr_id)

        cached_entries.append(
            {
                "image_path": str(img_path),
                "img_id": flickr_id,
                "filename": filename,
                "captions": caption_list,
            }
        )

    # Persist the enriched index once the download completes.
    with open(captions_file, "w", encoding="utf-8") as f:
        import json

        json.dump(cached_entries, f, ensure_ascii=False, indent=2)

    logger.info(
        "Loaded %d pairs (%d unique images) from Flickr30k",
        len(image_paths),
        len(image_paths) // _CAPTIONS_PER_IMAGE,
    )
    return image_paths, captions, image_ids


def load_flickr30k_from_hf(cache_dir: str) -> tuple[list[str], list[str]]:
    """Load Flickr30k from HuggingFace, returning image paths and captions.

    Thin backward-compatible wrapper over :func:`load_flickr30k_with_ids`
    that discards the image ids. Existing callers keep working unchanged.

    Args:
        cache_dir: Root directory for cached dataset files.

    Returns:
        A tuple (image_paths, captions) with each caption paired to its
        image path. Each image appears 5 times (once per caption).
    """
    image_paths, captions, _ = load_flickr30k_with_ids(cache_dir)
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
    image_paths, captions, _ = load_flickr30k_with_ids(
        cfg["dataset"]["local_cache_dir"]
    )
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

    image_paths, captions, image_ids = load_flickr30k_with_ids(
        cfg["dataset"]["local_cache_dir"]
    )
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
