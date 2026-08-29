"""Smoke test: run one real batch through VectorMindModel end-to-end.

Purpose: validate the complete Phase 2 dual-encoder architecture on a
real batch from the Phase 1 data pipeline (NOT synthetic data),
verifying shapes, numerical stability, L2 normalization, and
temperature initialization. This is the Phase 2 acceptance gate
(ROADMAP.md Phase 2 acceptance criteria).

Usage:
    python scripts/smoke_test_model.py

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import build_split_from_cache

from vectormind.data.dataloader import create_dataloaders
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Expected shared embedding dimension (ARCHITECTURE.md §4, configs/model.yaml)
EXPECTED_SHARED_DIM: int = 256


def _validate_embeddings(
    name: str,
    embeddings: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> bool:
    """Validate embedding tensor shape, NaN, Inf, and L2 normalization.

    Args:
        name: Human-readable name for logging (e.g. "image", "text").
        embeddings: The embedding tensor to validate.
        expected_shape: Expected shape of the tensor.

    Returns:
        True if all checks pass, False otherwise.
    """
    passed = True

    # Shape check
    if embeddings.shape != expected_shape:
        logger.error(
            "  %s shape mismatch: expected %s, got %s",
            name,
            expected_shape,
            embeddings.shape,
        )
        passed = False
    else:
        logger.info("  %s shape OK: %s", name, tuple(embeddings.shape))

    # NaN check
    if embeddings.isnan().any():
        logger.error("  %s contains NaN values!", name)
        passed = False
    else:
        logger.info("  %s NaN check passed", name)

    # Inf check
    if embeddings.isinf().any():
        logger.error("  %s contains Inf values!", name)
        passed = False
    else:
        logger.info("  %s Inf check passed", name)

    # L2 normalization check: each row should have unit norm
    norms = embeddings.norm(p=2, dim=-1)
    all_unit = torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    if not all_unit:
        logger.error(
            "  %s L2 normalization failed: norm range [%.6f, %.6f]",
            name,
            norms.min().item(),
            norms.max().item(),
        )
        passed = False
    else:
        logger.info(
            "  %s L2 normalization OK (norm range [%.6f, %.6f])",
            name,
            norms.min().item(),
            norms.max().item(),
        )

    return passed


def main() -> None:
    """Run the model smoke test."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("VectorMind Model Smoke Test (Phase 2 Acceptance Gate)")
    logger.info("=" * 60)

    # ---- Step 1: Load configs ----
    logger.info("Step 1: Loading configs...")
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    require_keys(data_config, ["dataset", "transforms"])
    require_keys(model_config, ["image_encoder", "text_encoder", "embedding"])

    # ---- Step 2: Load real data ----
    logger.info("Step 2: Loading real Flickr30k data...")

    # Create splits and tokenizer (reuse Phase 1 code exactly)
    train_pairs, val_pairs, test_pairs = build_split_from_cache(data_config)
    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    eval_transform = get_eval_transforms(data_config)

    # Use a single validation batch for the smoke test
    _, val_loader, _ = create_dataloaders(
        config=data_config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    # Grab one real batch
    logger.info("Step 3: Fetching one real batch from validation loader...")
    batch = next(iter(val_loader))
    images = batch["image"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    B = images.shape[0]
    logger.info("  Batch size: %d", B)
    logger.info("  Image shape: %s", tuple(images.shape))
    logger.info("  Input IDs shape: %s", tuple(input_ids.shape))
    logger.info("  Attention mask shape: %s", tuple(attention_mask.shape))

    # Decode and log first caption as sanity check
    decoded = tokenizer.decode(input_ids[:1])
    logger.info("  Sample caption (decoded): %r", decoded[0][:100])

    # ---- Step 4: Initialize model ----
    logger.info("Step 4: Initializing VectorMindModel...")
    model = VectorMindModel(model_config)
    model.eval()

    # ---- Step 5: Forward pass ----
    logger.info("Step 5: Running forward pass...")
    with torch.no_grad():
        result = model(images, input_ids, attention_mask)

    image_embeddings = result["image_embeddings"]
    text_embeddings = result["text_embeddings"]
    temperature = result["temperature"]

    # ---- Step 6: Validate outputs ----
    logger.info("Step 6: Validating outputs...")
    all_passed = True

    # Validate image embeddings
    logger.info("Validating image embeddings...")
    if not _validate_embeddings(
        "image_embeddings", image_embeddings, (B, EXPECTED_SHARED_DIM)
    ):
        all_passed = False

    # Validate text embeddings
    logger.info("Validating text embeddings...")
    if not _validate_embeddings(
        "text_embeddings", text_embeddings, (B, EXPECTED_SHARED_DIM)
    ):
        all_passed = False

    # Validate temperature
    logger.info("Validating temperature...")
    temp_val = temperature.item()
    if temp_val <= 0:
        logger.error("  Temperature is non-positive: %.6f", temp_val)
        all_passed = False
    else:
        logger.info("  Temperature value: %.6f (expected ~14.286)", temp_val)

    # Cross-modal similarity sanity check
    logger.info("Cross-modal similarity check...")
    similarities = torch.mm(image_embeddings, text_embeddings.t())
    diag_mean = similarities.diag().mean().item()
    off_diag_mean = (similarities.sum().item() - similarities.diag().sum().item()) / (
        B * (B - 1)
    )
    logger.info("  Diagonal (matched pairs) mean similarity: %.6f", diag_mean)
    logger.info("  Off-diagonal (unmatched) mean similarity: %.6f", off_diag_mean)
    if diag_mean <= off_diag_mean:
        logger.warning(
            "  WARNING: matched pairs are not more similar than "
            "unmatched pairs — this is expected before training, but "
            "worth monitoring."
        )

    # ---- Step 7: Summary ----
    logger.info("-" * 60)
    if all_passed:
        logger.info("ALL SMOKE TEST CHECKS PASSED")
        logger.info("Phase 2 acceptance criteria satisfied.")
    else:
        logger.error("SMOKE TEST FAILED — see errors above.")
        sys.exit(1)

    # Log model stats
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)
    logger.info("Temperature: %.6f", temp_val)
    logger.info("Shared embedding dim: %d", EXPECTED_SHARED_DIM)

    logger.info("=" * 60)
    logger.info("Model smoke test complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
