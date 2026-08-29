"""Persist the configured train/val/test split as an auditable manifest.

Purpose: compute the active data split (official Flickr30k split by
default) and write a JSON mapping of every image path to its split, so
training, evaluation, the index builder and a human reviewer all agree
on — and can point at — a single persisted artifact
(``dataset.split_manifest_path``).

Usage:
    python scripts/build_split_manifest.py

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import build_split_from_cache

from vectormind.data.splitter import persist_split_manifest
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Build and persist the split manifest from the active config."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("VectorMind: Building Split Manifest")
    logger.info("=" * 60)

    config = load_config("configs/data.yaml")
    require_keys(config, ["dataset"])

    split_mode = config["dataset"].get("split_mode", "official")
    logger.info("Split mode: %s", split_mode)

    train_pairs, val_pairs, test_pairs = build_split_from_cache(config)

    manifest_path = config["dataset"]["split_manifest_path"]
    written = persist_split_manifest(
        train_pairs, val_pairs, test_pairs, manifest_path
    )

    logger.info("-" * 60)
    logger.info("Train pairs: %d", len(train_pairs))
    logger.info("Val pairs:   %d", len(val_pairs))
    logger.info("Test pairs:  %d", len(test_pairs))
    logger.info("Manifest written to: %s", written)


if __name__ == "__main__":
    main()
