"""Verify Flickr30k dataset integrity for Phase 3.5.

Purpose: download (if needed) and validate the Flickr30k dataset from
HuggingFace, checking image count, caption count, captions per image,
corrupted files, and missing files. Generates a verification report.

This is an entry-point script, NOT imported by src/vectormind/.

Usage:
    python scripts/verify_dataset.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PIL import Image

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import load_flickr30k_from_hf
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Expected counts from DATASETS.md
EXPECTED_IMAGES: int = 31783
EXPECTED_CAPTIONS_PER_IMAGE: int = 5
EXPECTED_TOTAL_PAIRS: int = EXPECTED_IMAGES * EXPECTED_CAPTIONS_PER_IMAGE
# Tolerance for minor dataset version differences
IMAGE_COUNT_TOLERANCE: int = 100


def verify_dataset() -> bool:
    """Download and verify Flickr30k dataset integrity.

    Returns:
        True if all checks pass, False otherwise.
    """
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("Flickr30k Dataset Verification (Phase 3.5 Pre-check)")
    logger.info("=" * 60)

    all_passed = True
    start_time = time.time()

    # ---- Step 1: Load config ----
    logger.info("Step 1: Loading data configuration...")
    data_config = load_config("configs/data.yaml")
    require_keys(data_config, ["dataset", "transforms"])
    cache_dir = data_config["dataset"]["local_cache_dir"]
    logger.info("  Cache directory: %s", cache_dir)

    # ---- Step 2: Download / load dataset ----
    logger.info("Step 2: Loading Flickr30k from HuggingFace Datasets...")
    download_start = time.time()
    try:
        image_paths, captions = load_flickr30k_from_hf(cache_dir)
    except Exception as e:
        logger.error("  Failed to load dataset: %s", e)
        return False
    download_elapsed = time.time() - download_start
    logger.info("  Download/load completed in %.1f seconds", download_elapsed)

    # ---- Step 3: Count verification ----
    logger.info("Step 3: Verifying counts...")
    total_pairs = len(image_paths)
    unique_images = len(set(image_paths))
    captions_per_image: dict[str, int] = {}
    for path in image_paths:
        captions_per_image[path] = captions_per_image.get(path, 0) + 1

    logger.info("  Total pairs:      %d", total_pairs)
    logger.info("  Unique images:    %d", unique_images)
    logger.info(
        "  Captions/image:   %d (expected %d)",
        captions_per_image[next(iter(captions_per_image))],
        EXPECTED_CAPTIONS_PER_IMAGE,
    )

    # Check unique image count
    if abs(unique_images - EXPECTED_IMAGES) > IMAGE_COUNT_TOLERANCE:
        logger.error(
            "  Image count mismatch: got %d, expected ~%d (±%d)",
            unique_images,
            EXPECTED_IMAGES,
            IMAGE_COUNT_TOLERANCE,
        )
        all_passed = False
    else:
        logger.info(
            "  Image count OK: %d (expected ~%d)", unique_images, EXPECTED_IMAGES
        )

    # Check total pairs
    expected_pairs = unique_images * EXPECTED_CAPTIONS_PER_IMAGE
    if total_pairs != expected_pairs:
        logger.error(
            "  Total pair count mismatch: got %d, expected %d (%d images × %d captions)",
            total_pairs,
            expected_pairs,
            unique_images,
            EXPECTED_CAPTIONS_PER_IMAGE,
        )
        all_passed = False
    else:
        logger.info(
            "  Total pairs OK: %d = %d × %d",
            total_pairs,
            unique_images,
            EXPECTED_CAPTIONS_PER_IMAGE,
        )

    # Check captions per image is uniform
    cap_counts = set(captions_per_image.values())
    if cap_counts != {EXPECTED_CAPTIONS_PER_IMAGE}:
        logger.error(
            "  Non-uniform captions per image: found counts %s (expected all %d)",
            cap_counts,
            EXPECTED_CAPTIONS_PER_IMAGE,
        )
        all_passed = False
    else:
        logger.info(
            "  All images have exactly %d captions", EXPECTED_CAPTIONS_PER_IMAGE
        )

    # ---- Step 4: File existence check ----
    logger.info("Step 4: Checking file existence (sampling 500 images)...")
    sample_paths = list(set(image_paths))[:500]
    missing_count = 0
    for img_path in sample_paths:
        if not Path(img_path).exists():
            missing_count += 1
            logger.warning("  Missing file: %s", img_path)

    if missing_count > 0:
        logger.error(
            "  %d / %d sampled files are missing", missing_count, len(sample_paths)
        )
        all_passed = False
    else:
        logger.info("  All %d sampled files exist", len(sample_paths))

    # ---- Step 5: Image corruption check ----
    logger.info("Step 5: Checking image integrity (sampling 200 images)...")
    corrupt_count = 0
    corrupt_files: list[str] = []
    check_sample = list(set(image_paths))[:200]
    for img_path in check_sample:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            corrupt_count += 1
            corrupt_files.append(img_path)
            logger.warning("  Corrupt image: %s (%s)", img_path, e)

    if corrupt_count > 0:
        logger.error(
            "  %d / %d sampled images are corrupt", corrupt_count, len(check_sample)
        )
        for f in corrupt_files[:10]:
            logger.error("    %s", f)
        all_passed = False
    else:
        logger.info("  All %d sampled images are valid", len(check_sample))

    # ---- Step 6: Caption content check ----
    logger.info("Step 6: Checking caption content (sampling 50 pairs)...")
    empty_captions = 0
    short_captions = 0
    sample_indices = range(min(50, len(captions)))
    for idx in sample_indices:
        cap = captions[idx]
        if not cap or not cap.strip():
            empty_captions += 1
            logger.warning("  Empty caption at index %d", idx)
        elif len(cap.strip().split()) < 2:
            short_captions += 1
            logger.warning("  Very short caption at index %d: '%s'", idx, cap)

    if empty_captions > 0:
        logger.error("  %d empty captions found in sample", empty_captions)
        all_passed = False
    else:
        logger.info(
            "  All %d sampled captions are non-empty", len(list(sample_indices))
        )

    if short_captions > 0:
        logger.warning(
            "  %d very short captions found in sample (may be valid)", short_captions
        )

    # ---- Step 7: Split sanity check ----
    logger.info("Step 7: Verifying train/val/test splitting...")
    from vectormind.data.splitter import create_splits

    train_pairs, val_pairs, test_pairs = create_splits(
        data_config, image_paths, captions
    )
    train_images = len(set(p[0] for p in train_pairs))
    val_images = len(set(p[0] for p in val_pairs))
    test_images = len(set(p[0] for p in test_pairs))

    logger.info("  Train: %d images (%d pairs)", train_images, len(train_pairs))
    logger.info("  Val:   %d images (%d pairs)", val_images, len(val_pairs))
    logger.info("  Test:  %d images (%d pairs)", test_images, len(test_pairs))

    # Verify no leakage
    train_img_set = set(p[0] for p in train_pairs)
    val_img_set = set(p[0] for p in val_pairs)
    test_img_set = set(p[0] for p in test_pairs)

    assert train_img_set.isdisjoint(val_img_set), "LEAKAGE: train/val overlap"
    assert train_img_set.isdisjoint(test_img_set), "LEAKAGE: train/test overlap"
    assert val_img_set.isdisjoint(test_img_set), "LEAKAGE: val/test overlap"
    logger.info("  Zero image leakage confirmed across splits")

    total_split_images = train_images + val_images + test_images
    if total_split_images != unique_images:
        logger.error(
            "  Split image total mismatch: %d != %d",
            total_split_images,
            unique_images,
        )
        all_passed = False
    else:
        logger.info("  All %d images accounted for in splits", unique_images)

    # ---- Summary ----
    elapsed = time.time() - start_time
    logger.info("-" * 60)
    if all_passed:
        logger.info("DATASET VERIFICATION PASSED")
        logger.info("")
        logger.info("Summary:")
        logger.info("  - %d unique images, %d total pairs", unique_images, total_pairs)
        logger.info("  - %d captions per image (uniform)", EXPECTED_CAPTIONS_PER_IMAGE)
        logger.info("  - No missing files (in 500-sample check)")
        logger.info("  - No corrupt images (in 200-sample check)")
        logger.info("  - No empty captions (in 50-sample check)")
        logger.info("  - Zero leakage across train/val/test splits")
        logger.info("  - Verification completed in %.1f seconds", elapsed)
    else:
        logger.error("DATASET VERIFICATION FAILED — see errors above.")
        logger.error("Do NOT proceed to Phase 3.5 training until this passes.")

    logger.info("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
