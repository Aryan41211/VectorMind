"""Generate the overfit validation subset from real Flickr30k data.

Purpose: create a deterministic 100-image (500-pair) subset of
Flickr30k for the Phase 3.5 overfit sanity check. Saves subset
metadata to data/processed/overfit_subset.json.

This is an entry-point script, NOT imported by src/vectormind/.

Usage:
    python scripts/create_overfit_subset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import load_flickr30k_from_hf

from vectormind.data.overfit_subset import (
    DEFAULT_SEED,
    DEFAULT_SUBSET_SIZE,
    create_overfit_subset,
    save_subset_metadata,
)
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Output path for subset metadata
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "overfit_subset.json"


def main() -> None:
    """Generate and save the overfit validation subset."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("Overfit Subset Generator (Phase 3.5)")
    logger.info("=" * 60)

    # Load config
    data_config = load_config("configs/data.yaml")
    require_keys(data_config, ["dataset"])
    cache_dir = data_config["dataset"]["local_cache_dir"]

    # Load full dataset
    logger.info("Loading Flickr30k from cache...")
    image_paths, captions = load_flickr30k_from_hf(cache_dir)
    total_images = len(set(image_paths))
    total_pairs = len(image_paths)
    logger.info("Loaded %d pairs (%d unique images)", total_pairs, total_images)

    # Create subset
    logger.info(
        "Creating overfit subset: %d images, seed=%d",
        DEFAULT_SUBSET_SIZE,
        DEFAULT_SEED,
    )
    subset_pairs = create_overfit_subset(
        image_paths=image_paths,
        captions=captions,
        subset_size=DEFAULT_SUBSET_SIZE,
        seed=DEFAULT_SEED,
    )

    # Save metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_subset_metadata(
        pairs=subset_pairs,
        output_path=OUTPUT_FILE,
        subset_size=DEFAULT_SUBSET_SIZE,
        seed=DEFAULT_SEED,
        total_images=total_images,
        total_pairs=total_pairs,
    )

    # Summary
    logger.info("-" * 60)
    logger.info("Overfit subset created successfully")
    logger.info("  Images: %d", DEFAULT_SUBSET_SIZE)
    logger.info("  Pairs:  %d", len(subset_pairs))
    logger.info("  Seed:   %d", DEFAULT_SEED)
    logger.info("  Output: %s", OUTPUT_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
