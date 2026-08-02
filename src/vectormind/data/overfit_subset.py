"""Deterministic overfit validation subset builder.

Purpose: create a small, fixed subset of Flickr30k for the Phase 3.5
overfit sanity check. Selects by IMAGE (keeping all 5 captions per
selected image) with a fixed seed for reproducibility.

Design decisions:
- Select by IMAGE, not by caption — all 5 captions for a selected image
  stay together. This ensures the model sees all caption variants for
  each image during overfit training, which is the correct behavior
  for contrastive learning.
- Fixed seed (42) — the same subset is produced every time, making
  experiments reproducible.
- Persist subset as a JSON metadata file — the training script loads
  this file to know which images/captions to use, without re-running
  the selection logic.
- 100 images (500 pairs) — large enough to give the model something
  non-trivial to memorize, small enough to overfit in a reasonable
  number of steps on 6GB VRAM.

Input:
  - Full Flickr30k image_paths + captions from load_flickr30k_from_hf()
  - subset_size: number of unique images to select (default 100)
  - seed: random seed for deterministic selection (default 42)

Output:
  - subset_pairs: list of (image_path, caption) tuples
  - subset metadata JSON file
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Default subset configuration
DEFAULT_SUBSET_SIZE: int = 100
DEFAULT_SEED: int = 42
CAPTIONS_PER_IMAGE: int = 5


def create_overfit_subset(
    image_paths: list[str],
    captions: list[str],
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, str]]:
    """Create a deterministic overfit validation subset.

    Selects ``subset_size`` unique images (with all 5 captions each)
    using a fixed random seed. The selection is deterministic: same
    inputs always produce the same subset.

    Args:
        image_paths: List of image file paths (length N, each image
            appears 5 times — once per caption).
        captions: List of caption strings (length N, one per path).
        subset_size: Number of unique images to select. Default 100.
        seed: Random seed for deterministic selection. Default 42.

    Returns:
        A list of (image_path, caption) tuples for the selected subset.
        Length = subset_size * CAPTIONS_PER_IMAGE.

    Raises:
        ValueError: If subset_size exceeds the number of unique images.
        ValueError: If image_paths and captions have different lengths.
        ValueError: If image_paths is empty.

    Assumptions:
        - image_paths and captions have the same length.
        - Each image appears exactly 5 times (once per caption).
        - All image paths exist and are readable.

    Limitations:
        The subset is a random sample, not stratified by content.
        For a 100-image subset from 31k images, content diversity
        is expected to be sufficient for an overfit sanity check.
    """
    if len(image_paths) != len(captions):
        raise ValueError(
            f"image_paths length ({len(image_paths)}) must equal "
            f"captions length ({len(captions)})."
        )
    if len(image_paths) == 0:
        raise ValueError("image_paths must be non-empty.")

    # Group by unique image
    image_to_captions: dict[str, list[str]] = {}
    for path, caption in zip(image_paths, captions):
        if path not in image_to_captions:
            image_to_captions[path] = []
        image_to_captions[path].append(caption)

    unique_images = list(image_to_captions.keys())
    num_unique = len(unique_images)

    if subset_size > num_unique:
        raise ValueError(
            f"subset_size ({subset_size}) exceeds number of unique "
            f"images ({num_unique})."
        )

    logger.info(
        "Creating overfit subset: selecting %d images from %d unique "
        "images (seed=%d)",
        subset_size,
        num_unique,
        seed,
    )

    # Deterministic selection
    rng = random.Random(seed)
    selected_images = rng.sample(unique_images, subset_size)

    # Expand back to (image_path, caption) pairs
    subset_pairs: list[tuple[str, str]] = []
    for img_path in selected_images:
        for cap in image_to_captions[img_path]:
            subset_pairs.append((img_path, cap))

    logger.info(
        "Overfit subset created: %d images, %d pairs",
        subset_size,
        len(subset_pairs),
    )

    return subset_pairs


def save_subset_metadata(
    pairs: list[tuple[str, str]],
    output_path: str | Path,
    subset_size: int,
    seed: int,
    total_images: int,
    total_pairs: int,
) -> None:
    """Save subset metadata to a JSON file.

    The metadata file records:
    - The subset configuration (size, seed)
    - The source dataset statistics
    - The list of selected (image_path, caption) pairs

    Args:
        pairs: The subset pairs to save.
        output_path: Path to the output JSON file.
        subset_size: Number of unique images in the subset.
        seed: Random seed used for selection.
        total_images: Total unique images in the source dataset.
        total_pairs: Total pairs in the source dataset.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": {
            "subset_size": subset_size,
            "seed": seed,
            "captions_per_image": CAPTIONS_PER_IMAGE,
        },
        "source": {
            "total_images": total_images,
            "total_pairs": total_pairs,
            "dataset_id": "lmms-lab/flickr30k",
        },
        "subset": {
            "num_images": subset_size,
            "num_pairs": len(pairs),
            "pairs": [{"image_path": p, "caption": c} for p, c in pairs],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    file_size_kb = output_path.stat().st_size / 1024
    logger.info(
        "Subset metadata saved: %s (%.1f KB, %d pairs)",
        output_path,
        file_size_kb,
        len(pairs),
    )


def load_subset_metadata(
    metadata_path: str | Path,
) -> list[tuple[str, str]]:
    """Load subset pairs from a metadata JSON file.

    Args:
        metadata_path: Path to the subset metadata JSON file.

    Returns:
        A list of (image_path, caption) tuples.

    Raises:
        FileNotFoundError: If the metadata file doesn't exist.
        KeyError: If required keys are missing.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Subset metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pairs_data = metadata["subset"]["pairs"]
    pairs = [(entry["image_path"], entry["caption"]) for entry in pairs_data]

    logger.info(
        "Loaded subset metadata: %d images, %d pairs from %s",
        metadata["subset"]["num_images"],
        len(pairs),
        metadata_path,
    )

    return pairs
