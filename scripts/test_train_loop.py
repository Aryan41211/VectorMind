"""Acceptance test: validate training infrastructure on real data.

Purpose: run train_one_step() for 5-10 steps on a REAL batch from the
Phase 1 data pipeline, verifying that the training machinery works
end-to-end without crashing, produces finite loss, updates model
weights, and can save/load checkpoints.

This is the Phase 3 acceptance gate (ROADMAP.md Phase 3 acceptance
criteria). This is NOT a training run — it's a smoke test to confirm
the infrastructure is functional before Phase 3.5/4.

Usage:
    python scripts/test_train_loop.py

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import torch

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import load_flickr30k_from_hf
from vectormind.data.dataloader import create_dataloaders
from vectormind.data.splitter import create_splits
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.checkpoint import load_checkpoint, save_checkpoint
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.train_loop import (
    create_optimizer,
    create_scaler,
    train_one_step,
)
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Number of training steps for this smoke test
NUM_STEPS: int = 8


def main() -> None:
    """Run the training loop acceptance test."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("VectorMind Training Loop Acceptance Test (Phase 3 Gate)")
    logger.info("=" * 60)

    all_passed = True

    # ---- Step 1: Load configs and data ----
    logger.info("Step 1: Loading configs and real Flickr30k data...")
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    require_keys(data_config, ["dataset", "transforms"])
    require_keys(model_config, ["image_encoder", "text_encoder", "embedding"])

    cache_dir = data_config["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)

    train_pairs, val_pairs, test_pairs = create_splits(
        data_config, image_paths, captions
    )
    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    eval_transform = get_eval_transforms(data_config)

    _, val_loader, _ = create_dataloaders(
        config=data_config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    # ---- Step 2: Initialize model, optimizer, scaler, queue ----
    logger.info("Step 2: Initializing model, optimizer, scaler, memory queue...")
    model = VectorMindModel(model_config)
    model.train()

    optimizer = create_optimizer(model, lr=1e-3)
    scaler = create_scaler()
    memory_queue = MemoryQueue(
        queue_size=128, embed_dim=model_config["embedding"]["shared_dim"]
    )

    # Save initial weights for comparison
    initial_weights = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # ---- Step 3: Run training steps ----
    logger.info("Step 3: Running %d training steps...", NUM_STEPS)
    losses: list[float] = []

    for step in range(NUM_STEPS):
        # Get a batch
        batch = next(iter(val_loader))

        # Train one step
        metrics = train_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            memory_queue=memory_queue,
        )

        # Optimizer step (accumulation_steps=1, so step every batch)
        optimizer.step()
        scaler.update()
        optimizer.zero_grad()

        # Enqueue embeddings for memory queue
        with torch.no_grad():
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            text_embeds = model.encode_text(input_ids, attention_mask)
            memory_queue.enqueue(text_embeds)

        losses.append(metrics["loss"])
        logger.info(
            "  Step %d/%d: loss=%.4f, temp=%.4f, img_norm=%.4f, txt_norm=%.4f",
            step + 1,
            NUM_STEPS,
            metrics["loss"],
            metrics["temperature"],
            metrics["image_embed_norm"],
            metrics["text_embed_norm"],
        )

    # ---- Step 4: Verify no crash, finite loss ----
    logger.info("Step 4: Verifying results...")
    if not all(torch.isfinite(torch.tensor(loss)) for loss in losses):
        logger.error("  Some losses are not finite!")
        all_passed = False
    else:
        logger.info("  All %d losses are finite", NUM_STEPS)

    if not all(loss > 0 for loss in losses):
        logger.error("  Some losses are not positive!")
        all_passed = False
    else:
        logger.info("  All %d losses are positive", NUM_STEPS)

    # ---- Step 5: Verify weight updates ----
    logger.info("Step 5: Verifying weight updates...")
    weights_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param.detach(), initial_weights[name]):
            weights_changed = True
            break

    if not weights_changed:
        logger.error("  No model weights were updated!")
        all_passed = False
    else:
        logger.info("  Model weights were updated (optimizer is working)")

    # ---- Step 6: Checkpoint save/load test ----
    logger.info("Step 6: Checkpoint save/load test...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_checkpoint.pt"

        # Save
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            memory_queue=memory_queue,
            epoch=0,
            step=NUM_STEPS,
        )
        logger.info("  Checkpoint saved: %s", ckpt_path)

        # Load into fresh model
        model2 = VectorMindModel(model_config)
        optimizer2 = create_optimizer(model2, lr=1e-3)
        scaler2 = create_scaler()
        queue2 = MemoryQueue(
            queue_size=128, embed_dim=model_config["embedding"]["shared_dim"]
        )

        loaded_epoch, loaded_step = load_checkpoint(
            ckpt_path, model2, optimizer2, scaler2, queue2
        )

        # Verify restored state
        if loaded_epoch != 0 or loaded_step != NUM_STEPS:
            logger.error(
                "  Checkpoint metadata mismatch: epoch=%d (expected 0), "
                "step=%d (expected %d)",
                loaded_epoch,
                loaded_step,
                NUM_STEPS,
            )
            all_passed = False
        else:
            logger.info(
                "  Checkpoint metadata correct: epoch=%d, step=%d",
                loaded_epoch,
                loaded_step,
            )

        # Verify weights match
        weights_match = all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(model.parameters(), model2.parameters())
        )
        if not weights_match:
            logger.error("  Checkpoint weights do not match!")
            all_passed = False
        else:
            logger.info("  Checkpoint weights match exactly")

        # Verify queue state
        queue_match = (
            torch.allclose(memory_queue.queue, queue2.queue)
            and memory_queue.pointer == queue2.pointer
        )
        if not queue_match:
            logger.error("  Checkpoint queue state does not match!")
            all_passed = False
        else:
            logger.info("  Checkpoint queue state matches")

    # ---- Step 7: Summary ----
    logger.info("-" * 60)
    if all_passed:
        logger.info("ALL ACCEPTANCE TEST CHECKS PASSED")
        logger.info("Phase 3 acceptance criteria satisfied.")
        logger.info("")
        logger.info("Summary:")
        logger.info("  - %d training steps completed without crash", NUM_STEPS)
        logger.info("  - All losses finite and positive")
        logger.info("  - Model weights updated by optimizer")
        logger.info("  - Checkpoint save/load works correctly")
        logger.info("  - Memory queue state preserved across save/load")
    else:
        logger.error("ACCEPTANCE TEST FAILED — see errors above.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Training loop acceptance test complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
