"""Sanity check script for the Phase 1 data pipeline.

Purpose: load Flickr30k via the data pipeline, iterate 10 batches,
decode captions, log them alongside image tensor stats, and save a
sample grid — catching pairing/tokenizer bugs before real training.

Usage:
    python scripts/inspect_data_pipeline.py

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import load_flickr30k_from_hf

from vectormind.data.dataloader import create_dataloaders
from vectormind.data.splitter import create_splits
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms, get_train_transforms
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

NUM_INSPECT_BATCHES = 10


def main() -> None:
    """Run the data pipeline inspection."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("VectorMind Data Pipeline Inspection (Phase 1 Sanity Check)")
    logger.info("=" * 60)

    # Load config.
    config = load_config("configs/data.yaml")
    require_keys(config, ["dataset", "transforms"])

    # Load dataset from HuggingFace.
    cache_dir = config["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)

    # Create splits.
    train_pairs, val_pairs, test_pairs = create_splits(config, image_paths, captions)

    # Initialize tokenizer.
    tokenizer = CaptionTokenizer(
        tokenizer_name=config["dataset"]["tokenizer_name"],
        max_length=config["dataset"]["max_text_length"],
    )

    # Create transforms.
    train_transform = get_train_transforms(config)
    eval_transform = get_eval_transforms(config)

    # Create dataloaders.
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    # Inspect batches.
    logger.info("-" * 60)
    logger.info("Inspecting %d batches from train_loader...", NUM_INSPECT_BATCHES)
    logger.info("-" * 60)

    all_passed = True
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= NUM_INSPECT_BATCHES:
            break

        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        caption_texts = batch["caption_text"]

        # Check shapes.
        expected_img_shape = (
            config["dataset"]["batch_size"],
            3,
            config["dataset"]["image_size"],
            config["dataset"]["image_size"],
        )
        if images.shape != expected_img_shape:
            logger.error(
                "Batch %d: image shape mismatch — expected %s, got %s",
                batch_idx,
                expected_img_shape,
                images.shape,
            )
            all_passed = False
        else:
            logger.info("Batch %d: image shape OK %s", batch_idx, tuple(images.shape))

        # Check for NaN/Inf.
        if images.isnan().any() or images.isinf().any():
            logger.error("Batch %d: images contain NaN or Inf!", batch_idx)
            all_passed = False
        if input_ids.isnan().any() or input_ids.isinf().any():
            logger.error("Batch %d: input_ids contain NaN or Inf!", batch_idx)
            all_passed = False

        # Decode and log first 3 captions.
        decoded = tokenizer.decode(input_ids[:3])
        for i, (orig, dec) in enumerate(zip(caption_texts[:3], decoded, strict=False)):
            logger.info(
                "  Batch %d, sample %d: ORIGINAL=%r | DECODED=%r",
                batch_idx,
                i,
                orig[:80],
                dec[:80],
            )

        # Check attention mask: sum should be > 0 for all samples.
        mask_sums = attention_mask.sum(dim=1)
        if (mask_sums == 0).any():
            logger.error("Batch %d: some attention masks are all-zero!", batch_idx)
            all_passed = False

    logger.info("-" * 60)
    if all_passed:
        logger.info(
            "ALL %d BATCHES PASSED sanity checks.",
            min(NUM_INSPECT_BATCHES, len(train_loader)),
        )
    else:
        logger.error("SOME BATCHES FAILED — see errors above.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Data pipeline inspection complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
