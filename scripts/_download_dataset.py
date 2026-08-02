"""Download remaining Flickr30k images and captions."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

_HF_DATASET_ID = "lmms-lab/flickr30k"
_HF_SPLIT = "test"
CACHE_DIR = "data/raw/flickr30k"


def download_remaining():
    """Download any missing images and all captions."""
    setup_logging(level=logging.INFO)
    images_dir = Path(CACHE_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(images_dir.glob("*.jpg"))
    existing_count = len(existing)
    logger.info("Existing images: %d / 31783", existing_count)

    # Download images + captions via streaming
    from datasets import load_dataset

    logger.info("Loading dataset via streaming...")
    ds = load_dataset(_HF_DATASET_ID, split=_HF_SPLIT, streaming=True)

    idx = 0
    new_images = 0
    captions = []
    start = time.time()

    for example in ds:
        img_path = images_dir / f"{idx:06d}.jpg"
        if not img_path.exists():
            example["image"].save(img_path)
            new_images += 1
            if new_images % 100 == 0:
                logger.info("  Downloaded %d new images (total: %d)", new_images, idx + 1)

        for cap in example["caption"]:
            captions.append(cap)

        idx += 1
        if idx % 5000 == 0:
            elapsed = time.time() - start
            logger.info("  Progress: %d images, %.1fs elapsed", idx, elapsed)

    elapsed = time.time() - start
    total = len(list(images_dir.glob("*.jpg")))
    logger.info(
        "Download complete: %d total images, %d new, %d captions, %.1fs",
        total, new_images, len(captions), elapsed,
    )

    # Verify
    if total >= 31700 and len(captions) == total * 5:
        logger.info("VERIFICATION PASSED")
    else:
        logger.warning(
            "Counts: images=%d, captions=%d (expected ~31783 images, ~158915 captions)",
            total, len(captions),
        )


if __name__ == "__main__":
    download_remaining()
