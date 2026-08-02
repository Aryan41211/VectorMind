"""Phase 3.5 overfit sanity check: prove the pipeline can learn.

Purpose: train the VectorMind dual-encoder on a tiny 100-image subset
of Flickr30k for many epochs until it memorizes the training pairs.
This is the single most important risk-reduction step in the project —
it catches embedding collapse, broken loss, data bugs, and silent
failures before any real compute is spent on a full training run.

Success criteria (ROADMAP.md Phase 3.5):
- Near-perfect Recall@1 on the training set itself
- Embedding variance stays healthy (not collapsing to a single point)
- Loss decreases to near zero

This script reuses the COMPLETE Phase 3 infrastructure:
- train_one_step() for forward/backward
- create_optimizer() and create_scaler()
- save_checkpoint() / load_checkpoint()
- TrainingLogger for TensorBoard

Memory queue is DISABLED — for a 100-image overfit check, extra
negatives from a queue would actively work against memorization.

Usage:
    python scripts/overfit_sanity_check.py

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import load_flickr30k_from_hf
from vectormind.data.dataloader import create_dataloaders
from vectormind.data.overfit_subset import load_subset_metadata
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.checkpoint import save_checkpoint
from vectormind.training.logger import TrainingLogger
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.train_loop import (
    create_optimizer,
    create_scaler,
    train_one_step,
)
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Checkpoint directory for overfit experiments
CHECKPOINT_DIR = Path("checkpoints/overfit")
LOG_DIR = Path("logs/overfit")


def compute_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute Recall@K for image-to-text retrieval.

    For each image, check whether its matching caption is within the
    top-K results when ranking all captions by cosine similarity.

    Args:
        image_embeds: L2-normalized image embeddings [N, D].
        text_embeds: L2-normalized text embeddings [N, D].
        k: Number of top results to consider.

    Returns:
        Recall@K as a float between 0 and 1.
    """
    # Similarity matrix: [N_images, N_texts]
    similarity = image_embeds @ text_embeds.T

    # For each image (row), the matching caption is at the same index
    # (diagonal), since image_paths[i] corresponds to captions[i]
    labels = torch.arange(similarity.shape[0], device=similarity.device)

    # Top-K retrieval
    _, top_k_indices = similarity.topk(k, dim=1)

    # Check if the correct label is in the top-K
    correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
    recall_at_k = correct.float().mean().item()

    return recall_at_k


def compute_embedding_variance(
    embeds: torch.Tensor,
) -> dict[str, float]:
    """Compute embedding space diagnostics.

    Args:
        embeds: L2-normalized embeddings [N, D].

    Returns:
        Dictionary with embedding variance metrics.
    """
    # Per-dimension variance (should be > 0 for healthy embeddings)
    dim_variance = embeds.var(dim=0).mean().item()

    # Pairwise distance statistics
    pairwise_dist = torch.cdist(embeds, embeds, p=2)
    # Exclude diagonal (self-distance = 0)
    mask = ~torch.eye(
        pairwise_dist.shape[0], dtype=torch.bool, device=pairwise_dist.device
    )
    mean_dist = pairwise_dist[mask].mean().item()
    min_dist = pairwise_dist[mask].min().item()

    return {
        "embed_dim_variance": dim_variance,
        "embed_mean_pairwise_dist": mean_dist,
        "embed_min_pairwise_dist": min_dist,
    }


def main() -> None:
    """Run the overfit sanity check."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("VectorMind Phase 3.5 — Overfit Sanity Check")
    logger.info("=" * 60)

    start_time = time.time()

    # ---- Step 1: Load configs ----
    logger.info("Step 1: Loading configurations...")
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    overfit_config = load_config("configs/overfit.yaml")
    require_keys(data_config, ["dataset", "transforms"])
    require_keys(model_config, ["image_encoder", "text_encoder", "embedding"])
    require_keys(overfit_config, ["subset", "training", "optimizer", "memory_queue"])

    subset_cfg = overfit_config["subset"]
    train_cfg = overfit_config["training"]
    optim_cfg = overfit_config["optimizer"]
    mq_cfg = overfit_config["memory_queue"]

    # ---- Step 2: Load overfit subset ----
    logger.info("Step 2: Loading overfit subset...")
    subset_pairs = load_subset_metadata(subset_cfg["metadata_path"])
    logger.info(
        "  Loaded %d pairs from %s", len(subset_pairs), subset_cfg["metadata_path"]
    )

    # ---- Step 3: Build DataLoader from subset ----
    logger.info("Step 3: Building DataLoader from subset...")
    # Override batch_size in data_config for overfit
    overfit_data_config = dict(data_config)
    overfit_data_config["dataset"] = dict(data_config["dataset"])
    overfit_data_config["dataset"]["batch_size"] = subset_cfg["batch_size"]

    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    eval_transform = get_eval_transforms(data_config)

    # Use all subset pairs for both "train" and "val" (overfit = memorize training set)
    train_loader, _, _ = create_dataloaders(
        config=overfit_data_config,
        train_pairs=subset_pairs,
        val_pairs=subset_pairs[:10],  # dummy val (not used)
        test_pairs=subset_pairs[:10],  # dummy test (not used)
        train_transform=eval_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    logger.info(
        "  Train loader: %d batches (batch_size=%d)",
        len(train_loader),
        subset_cfg["batch_size"],
    )

    # ---- Step 4: Initialize model ----
    logger.info("Step 4: Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Model: %d parameters, device=%s", n_params, device)

    # ---- Step 5: Initialize optimizer, scaler, queue ----
    logger.info("Step 5: Initializing optimizer and scaler...")
    optimizer = create_optimizer(
        model, lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"]
    )
    scaler = create_scaler()

    # Memory queue disabled for overfit
    if mq_cfg["enabled"]:
        memory_queue = MemoryQueue(
            queue_size=mq_cfg["queue_size"],
            embed_dim=model_config["embedding"]["shared_dim"],
            device=device,
        )
        logger.info("  Memory queue: ENABLED (size=%d)", mq_cfg["queue_size"])
    else:
        memory_queue = MemoryQueue(
            queue_size=1,
            embed_dim=model_config["embedding"]["shared_dim"],
            device=device,
        )
        logger.info("  Memory queue: DISABLED (overfit mode)")

    # ---- Step 6: Initialize logger ----
    logger.info("Step 6: Initializing TensorBoard logger...")
    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    training_logger = TrainingLogger(log_dir=log_dir)

    # ---- Step 7: Training loop ----
    num_epochs = train_cfg["epochs"]
    log_every = train_cfg["log_every_n_steps"]
    save_every = train_cfg["save_every_n_epochs"]
    global_step = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 7: Starting overfit training for %d epochs...", num_epochs)
    logger.info(
        "  Success criteria: Recall@1 > 0.9, loss -> near 0, embedding variance > 0"
    )

    best_recall = 0.0
    training_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []

        for batch_idx, batch in enumerate(train_loader):
            # Train one step
            metrics = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                memory_queue=memory_queue,
                accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
                device=device,
            )

            # Optimizer step
            optimizer.step()
            scaler.update()
            optimizer.zero_grad()

            # Enqueue text embeddings (even with disabled queue, no-op)
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                text_embeds = model.encode_text(input_ids, attention_mask)
                memory_queue.enqueue(text_embeds)

            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm**0.5

            epoch_losses.append(metrics["loss"])
            epoch_grad_norms.append(grad_norm)

            # Log per-step
            if global_step % log_every == 0:
                step_metrics = {
                    "train/loss": metrics["loss"],
                    "train/loss_i2t": metrics.get("loss_i2t", 0.0),
                    "train/loss_t2i": metrics.get("loss_t2i", 0.0),
                    "train/temperature": metrics["temperature"],
                    "train/image_embed_norm": metrics["image_embed_norm"],
                    "train/text_embed_norm": metrics["text_embed_norm"],
                    "train/image_embed_std": metrics["image_embed_std"],
                    "train/text_embed_std": metrics["text_embed_std"],
                    "train/grad_norm": grad_norm,
                    "train/lr": optim_cfg["lr"],
                    "train/gpu_memory_gb": metrics.get("gpu_memory_gb", 0.0),
                }
                training_logger.log_metrics(global_step, step_metrics)

                logger.info(
                    "  Epoch %d, Step %d/%d: loss=%.4f, temp=%.4f, grad_norm=%.4f",
                    epoch + 1,
                    batch_idx + 1,
                    len(train_loader),
                    metrics["loss"],
                    metrics["temperature"],
                    grad_norm,
                )

            global_step += 1

        # ---- End of epoch: compute Recall@1 on training set ----
        epoch_elapsed = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)

        # Compute Recall@1 on the full training set
        model.eval()
        all_image_embeds: list[torch.Tensor] = []
        all_text_embeds: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                img_emb = model.encode_image(images)
                txt_emb = model.encode_text(input_ids, attention_mask)
                all_image_embeds.append(img_emb)
                all_text_embeds.append(txt_emb)

        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)

        recall_1 = compute_recall_at_k(all_image_embeds, all_text_embeds, k=1)
        recall_5 = compute_recall_at_k(all_image_embeds, all_text_embeds, k=5)
        recall_10 = compute_recall_at_k(all_image_embeds, all_text_embeds, k=10)
        embed_stats = compute_embedding_variance(all_image_embeds)

        model.train()

        # Log epoch metrics
        epoch_metrics = {
            "epoch/avg_loss": avg_loss,
            "epoch/avg_grad_norm": avg_grad_norm,
            "epoch/recall@1": recall_1,
            "epoch/recall@5": recall_5,
            "epoch/recall@10": recall_10,
            "epoch/embed_dim_variance": embed_stats["embed_dim_variance"],
            "epoch/embed_mean_pairwise_dist": embed_stats["embed_mean_pairwise_dist"],
            "epoch/embed_min_pairwise_dist": embed_stats["embed_min_pairwise_dist"],
            "epoch/temperature": model.temperature.item(),
            "epoch/epoch_time_s": epoch_elapsed,
        }
        training_logger.log_epoch(epoch, epoch_metrics)
        training_logger.flush()

        # Track best recall
        if recall_1 > best_recall:
            best_recall = recall_1

        logger.info(
            "  Epoch %d/%d complete (%.1fs): loss=%.4f, R@1=%.4f, R@5=%.4f, "
            "R@10=%.4f, embed_var=%.6f, best_R@1=%.4f",
            epoch + 1,
            num_epochs,
            epoch_elapsed,
            avg_loss,
            recall_1,
            recall_5,
            recall_10,
            embed_stats["embed_dim_variance"],
            best_recall,
        )

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            ckpt_path = CHECKPOINT_DIR / f"overfit_epoch{epoch + 1:03d}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                memory_queue=memory_queue,
                epoch=epoch,
                step=global_step,
                config=overfit_config,
            )

    # ---- Step 8: Final evaluation and summary ----
    total_elapsed = time.time() - training_start
    logger.info("=" * 60)
    logger.info("Phase 3.5 Overfit Sanity Check — RESULTS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  Subset: %d images, %d pairs", 100, len(subset_pairs))
    logger.info("  Epochs: %d", num_epochs)
    logger.info("  Batch size: %d", subset_cfg["batch_size"])
    logger.info("  Learning rate: %.1e", optim_cfg["lr"])
    logger.info("  Memory queue: %s", "enabled" if mq_cfg["enabled"] else "disabled")
    logger.info("")
    logger.info("Results:")
    logger.info("  Final loss:          %.4f", avg_loss)
    logger.info("  Best Recall@1:       %.4f", best_recall)
    logger.info("  Final Recall@1:      %.4f", recall_1)
    logger.info("  Final Recall@5:      %.4f", recall_5)
    logger.info("  Final Recall@10:     %.4f", recall_10)
    logger.info("  Embed dim variance:  %.6f", embed_stats["embed_dim_variance"])
    logger.info("  Mean pairwise dist:  %.4f", embed_stats["embed_mean_pairwise_dist"])
    logger.info("  Min pairwise dist:   %.4f", embed_stats["embed_min_pairwise_dist"])
    logger.info("  Total time:          %.1f seconds", total_elapsed)
    logger.info("  Global steps:        %d", global_step)
    logger.info("")

    # Verdict
    passed = True
    failure_reasons: list[str] = []

    if best_recall < 0.9:
        passed = False
        failure_reasons.append(
            f"Recall@1={best_recall:.4f} < 0.9 (model cannot memorize)"
        )

    if avg_loss > 0.5:
        passed = False
        failure_reasons.append(f"Loss={avg_loss:.4f} > 0.5 (loss not converging)")

    if embed_stats["embed_dim_variance"] < 1e-6:
        passed = False
        failure_reasons.append(
            f"Embedding variance={embed_stats['embed_dim_variance']:.8f} < 1e-6 (collapse detected)"
        )

    if passed:
        logger.info("VERDICT: PASSED")
        logger.info("The pipeline can memorize a tiny subset.")
        logger.info("Phase 3.5 acceptance criteria satisfied.")
        logger.info("Safe to proceed to Phase 4 (full training run).")
    else:
        logger.error("VERDICT: FAILED")
        for reason in failure_reasons:
            logger.error("  - %s", reason)
        logger.error("Do NOT proceed to Phase 4 until these issues are resolved.")

    logger.info("=" * 60)
    training_logger.close()

    # Save final checkpoint
    final_ckpt_path = CHECKPOINT_DIR / "overfit_final.pt"
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        memory_queue=memory_queue,
        epoch=num_epochs,
        step=global_step,
        config=overfit_config,
    )
    logger.info("Final checkpoint saved: %s", final_ckpt_path)


if __name__ == "__main__":
    main()
