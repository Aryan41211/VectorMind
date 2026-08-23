"""Phase 4 full training script for VectorMind.

Purpose: train the dual-encoder model on the full Flickr30k training
split, with validation monitoring, checkpointing, and TensorBoard
logging. This is the main entry point for Phase 4 of the ROADMAP.

This script reuses the COMPLETE Phase 3 infrastructure:
- train_one_step() for forward/backward with AMP
- create_optimizer() / create_scaler()
- save_checkpoint() / load_checkpoint()
- TrainingLogger for TensorBoard
- MemoryQueue for MoCo-style negatives

Usage:
    python scripts/train.py
    python scripts/train.py --resume checkpoints/train/latest.pt

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import argparse
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
from vectormind.data.splitter import create_splits
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms, get_train_transforms
from vectormind.evaluation.evaluator import evaluate_split
from vectormind.models.vectormind_model import (
    DEFAULT_MAX_LOGIT_SCALE,
    VectorMindModel,
)
from vectormind.training.checkpoint import (
    load_checkpoint,
    read_checkpoint_metric,
    save_checkpoint,
)
from vectormind.training.logger import TrainingLogger
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.oom import run_step_with_oom_retry
from vectormind.training.train_loop import (
    create_optimizer,
    create_scaler,
    train_one_step,
)
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Directories


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VectorMind Phase 4: Full Training Run"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--no-queue",
        action="store_true",
        default=False,
        help="Disable memory queue (baseline experiment).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataset.num_workers from configs/data.yaml.",
    )
    return parser.parse_args()


def evaluate(
    model: VectorMindModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    captions_per_image: int = 5,
) -> dict[str, float]:
    """Evaluate the model on a dataset split.

    Thin wrapper over the shared implementation in
    ``vectormind.evaluation.evaluator``. This script used to carry its
    own copy of the recall and diagnostics code; four scripts each
    having one meant a metric fix in any of them silently left the
    others reporting different numbers for the same checkpoint.

    Args:
        model: The trained VectorMindModel.
        dataloader: DataLoader over the split to evaluate.
        device: Device to evaluate on.
        captions_per_image: Captions per image for Recall computation.

    Returns:
        Flat metric mapping: ``recall@1/5/10``, ``t2i_recall@1/5/10``,
        embedding diagnostics, and health fields including
        ``separation`` and ``collapsed``.

    Raises:
        ValueError: If the split size is not divisible by
            ``captions_per_image``.
    """
    return evaluate_split(
        model, dataloader, device, captions_per_image
    ).to_flat_dict()


def main() -> None:
    """Run the Phase 4 full training pipeline."""
    args = parse_args()
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("VectorMind Phase 4 -- Full Training Run")
    logger.info("=" * 60)

    # ---- Performance optimizations ----
    torch.backends.cudnn.benchmark = True
    logger.info("  cuDNN benchmark: enabled")

    # ---- Step 1: Load configurations ----
    logger.info("Step 1: Loading configurations...")
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    training_config = load_config("configs/training.yaml")
    require_keys(data_config, ["dataset", "transforms"])
    require_keys(model_config, ["image_encoder", "text_encoder", "embedding"])
    require_keys(
        training_config,
        ["optimizer", "scheduler", "memory_queue", "epochs"],
    )

    optim_cfg = training_config["optimizer"]
    sched_cfg = training_config["scheduler"]
    mq_cfg = training_config["memory_queue"]
    train_cfg = training_config

    # Apply CLI overrides
    num_epochs = args.epochs or train_cfg["epochs"]
    lr = args.lr or optim_cfg["lr"]

    # DataLoader settings come from configs/data.yaml (CLAUDE.md §6);
    # the RTX 4050 tuning rationale is documented there. --num-workers
    # is the one CLI override, so a machine with different core counts
    # can adjust without editing config.
    data_config_optimized = dict(data_config)
    data_config_optimized["dataset"] = dict(data_config["dataset"])
    if args.num_workers is not None:
        data_config_optimized["dataset"]["num_workers"] = args.num_workers

    # ---- Step 2: Load and split dataset ----
    logger.info("Step 2: Loading Flickr30k dataset...")
    cache_dir = data_config["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)
    logger.info(
        "  Loaded %d pairs (%d unique images)",
        len(image_paths),
        len(set(image_paths)),
    )

    logger.info("Step 2b: Splitting dataset...")
    train_pairs, val_pairs, test_pairs = create_splits(
        config=data_config,
        image_paths=[Path(p) for p in image_paths],
        captions=captions,
    )
    logger.info(
        "  Train: %d pairs, Val: %d pairs, Test: %d pairs",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )

    # ---- Step 3: Build DataLoaders ----
    logger.info("Step 3: Building DataLoaders (optimized)...")
    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    train_transform = get_train_transforms(data_config)
    eval_transform = get_eval_transforms(data_config)

    train_loader, val_loader, _ = create_dataloaders(
        config=data_config_optimized,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )
    logger.info(
        "  DataLoader: batch_size=%d, num_workers=%d, persistent_workers=False, "
        "prefetch_factor=%d, pin_memory=True",
        data_config_optimized["dataset"]["batch_size"],
        data_config_optimized["dataset"]["num_workers"],
        data_config_optimized["dataset"]["prefetch_factor"],
    )

    # ---- Step 4: Initialize model ----
    logger.info("Step 4: Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Model: %d parameters, device=%s", n_params, device)

    # ---- Step 5: Initialize optimizer, scaler, scheduler, queue ----
    logger.info("Step 5: Initializing optimizer and scheduler...")
    optimizer = create_optimizer(model, lr=lr, weight_decay=optim_cfg["weight_decay"])
    scaler = create_scaler()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["T_max"],
        eta_min=sched_cfg["eta_min"],
    )

    # Memory queue. It starts inactive and warms up: without a momentum
    # encoder, early queue entries are stale enough to swamp the gradient
    # at 4096-against-128 (see MemoryQueue's class docstring).
    use_queue = not args.no_queue
    queue_warmup_epochs = int(mq_cfg.get("warmup_epochs", 0))
    if use_queue:
        memory_queue = MemoryQueue(
            queue_size=mq_cfg["queue_size"],
            embed_dim=model_config["embedding"]["shared_dim"],
            device=device,
            active=queue_warmup_epochs <= 0,
        )
        logger.info(
            "  Memory queue: ENABLED (size=%d, warmup=%d epochs)",
            mq_cfg["queue_size"],
            queue_warmup_epochs,
        )
    else:
        # Dummy queue with size=1 — enqueue is a no-op, get returns empty
        memory_queue = MemoryQueue(
            queue_size=1,
            embed_dim=model_config["embedding"]["shared_dim"],
            device=device,
        )
        logger.info("  Memory queue: DISABLED (baseline experiment)")

    logger.info(
        "  Optimizer: AdamW (lr=%.1e, wd=%.4f), Scheduler: cosine (T_max=%d, eta_min=%.1e)",
        lr,
        optim_cfg["weight_decay"],
        sched_cfg["T_max"],
        sched_cfg["eta_min"],
    )

    # ---- Step 6: Resume from checkpoint (if specified) ----
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info("Step 6: Resuming from checkpoint %s...", args.resume)
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scaler, memory_queue
        )
        start_epoch += 1  # Resume from next epoch
        logger.info("  Resumed: epoch=%d, step=%d", start_epoch, global_step)
    else:
        logger.info("Step 6: Starting from scratch.")

    # ---- Step 7: Initialize logger ----
    logger.info("Step 7: Initializing TensorBoard logger...")
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    log_dir = Path(train_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_logger = TrainingLogger(log_dir=log_dir)

    # ---- Step 8: Training loop ----
    log_every = 50  # Reduced from 10 to minimize logging overhead
    eval_every = train_cfg["eval_every_n_epochs"]
    save_every = train_cfg["save_every_n_epochs"]
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

    logger.info("Step 8: Starting training for %d epochs...", num_epochs)
    logger.info(
        "  Effective batch: %d x %d = %d",
        data_config["dataset"]["batch_size"],
        accum_steps,
        data_config["dataset"]["batch_size"] * accum_steps,
    )

    best_val_recall = 0.0
    # Recover the best-so-far from the existing best checkpoint. Starting
    # at 0.0 after a resume means the first completed epoch always wins
    # the comparison and overwrites best_model.pt however bad it is —
    # which replaced a 17.46% R@10 checkpoint with a 10.51% one.
    best_val_recall10 = 0.0
    if args.resume:
        best_val_recall10 = read_checkpoint_metric(
            checkpoint_dir / "best_model.pt", "recall@10"
        )
        logger.info(
            "  Best-so-far restored: val R@10=%.4f", best_val_recall10
        )
    training_start = time.time()
    first_epoch_metrics: dict[str, float] | None = None
    last_epoch_metrics: dict[str, float] | None = None
    loss_ema: float | None = None  # Exponential moving average for spike detection
    epochs_without_improvement = 0
    early_stop_patience = train_cfg.get("early_stopping", {}).get("patience", 5)
    early_stop_enabled = train_cfg.get("early_stopping", {}).get("enabled", True)
    min_delta = train_cfg.get("early_stopping", {}).get("min_delta", 0.001)

    temp_cfg = train_cfg.get("temperature", {})
    clamp_enabled = temp_cfg.get("clamp_enabled", True)
    max_logit_scale = float(temp_cfg.get("max_logit_scale", DEFAULT_MAX_LOGIT_SCALE))
    logger.info(
        "Logit-scale clamp: %s (ceiling=%.1f)",
        "ENABLED" if clamp_enabled else "DISABLED",
        max_logit_scale,
    )

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Activate the queue once the encoder has stabilized. It has been
        # filling throughout warmup, so it switches on already full of
        # recent embeddings rather than starting empty.
        if use_queue and not memory_queue.active and epoch >= queue_warmup_epochs:
            memory_queue.activate()

        model.train()
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []

        for batch_idx, batch in enumerate(train_loader):
            # Forward + backward, retried if a transient allocation
            # fails. This GPU also drives the display, and a run reached
            # epoch 5 of 20 before another process took the headroom and
            # killed it (src/vectormind/training/oom.py). The retry sits
            # before the optimizer step, so replaying it is safe:
            # gradients are overwritten, not accumulated twice.
            metrics = run_step_with_oom_retry(
                lambda batch=batch: train_one_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scaler=scaler,
                    memory_queue=memory_queue,
                    accumulation_steps=accum_steps,
                    device=device,
                ),
                context=f"epoch {epoch + 1} step {batch_idx + 1}",
            )

            # Gradient norm (BEFORE optimizer step to capture actual gradients)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm**0.5

            # Optimizer step at accumulation boundary
            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Clamp the learnable logit scale immediately after the
                # update. Without this the optimizer lowers contrastive
                # loss by inflating the scale rather than separating
                # representations, and the embedding space collapses
                # into a cone (docs/KNOWN_ISSUES.md §1).
                if clamp_enabled:
                    metrics["temperature"] = model.clamp_log_temperature(
                        max_logit_scale
                    )

            # The memory queue is filled inside train_one_step() from the
            # embeddings the loss already computed — no second forward pass.

            epoch_losses.append(metrics["loss"])
            epoch_grad_norms.append(grad_norm)

            # ---- Instability detection ----
            loss_val = metrics["loss"]
            if torch.isnan(torch.tensor(loss_val)) or torch.isinf(
                torch.tensor(loss_val)
            ):
                logger.error(
                    "  INSTABILITY DETECTED: loss is %s at step %d. Stopping training.",
                    "NaN" if torch.isnan(torch.tensor(loss_val)) else "Inf",
                    global_step,
                )
                break

            # Loss spike detection (3x EMA)
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss_val
                if loss_val > 3.0 * loss_ema and global_step > 100:
                    logger.warning(
                        "  LOSS SPIKE: loss=%.4f at step %d (EMA=%.4f, ratio=%.1fx)",
                        loss_val,
                        global_step,
                        loss_val / loss_ema,
                        loss_val / loss_ema,
                    )

            # Log per-step
            if global_step % log_every == 0:
                step_metrics = {
                    "train/loss": metrics["loss"],
                    "train/temperature": metrics["temperature"],
                    "train/image_embed_norm": metrics["image_embed_norm"],
                    "train/text_embed_norm": metrics["text_embed_norm"],
                    "train/image_embed_std": metrics["image_embed_std"],
                    "train/text_embed_std": metrics["text_embed_std"],
                    "train/grad_norm": grad_norm,
                    "train/lr": lr,
                    "train/gpu_memory_gb": metrics.get("gpu_memory_gb", 0.0),
                }
                training_logger.log_metrics(global_step, step_metrics)

                logger.info(
                    "  Epoch %d, Step %d/%d: loss=%.4f, temp=%.4f",
                    epoch + 1,
                    batch_idx + 1,
                    len(train_loader),
                    metrics["loss"],
                    metrics["temperature"],
                )

            global_step += 1

        # ---- End of epoch ----
        epoch_elapsed = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch metrics
        epoch_metrics = {
            "epoch/avg_loss": avg_loss,
            "epoch/avg_grad_norm": avg_grad_norm,
            "epoch/temperature": model.temperature.item(),
            "epoch/lr": current_lr,
            "epoch/epoch_time_s": epoch_elapsed,
            "epoch/memory_queue_size": memory_queue.current_size,
        }

        # ---- Validation every N epochs ----
        if (epoch + 1) % eval_every == 0:
            # Free GPU memory before validation to avoid OOM
            torch.cuda.empty_cache()
            logger.info("  Running validation...")
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                captions_per_image=5,
            )
            for key, val in val_metrics.items():
                epoch_metrics[f"val/{key}"] = val

            logger.info(
                "  Val: R@1=%.4f, R@5=%.4f, R@10=%.4f, embed_var=%.6f",
                val_metrics["recall@1"],
                val_metrics["recall@5"],
                val_metrics["recall@10"],
                val_metrics["image_dim_variance"],
            )
            # Retrieval metrics can hold up while the representation
            # degrades — Phase 4 proved that. Log health next to them so
            # a collapse is visible in the same glance.
            logger.info(
                "  Health: separation=%.4f (matched=%.4f, unmatched=%.4f), "
                "mean_cos=%.4f, ||mean||=%.4f%s",
                val_metrics["separation"],
                val_metrics["matched_similarity"],
                val_metrics["unmatched_similarity"],
                val_metrics["image_mean_cosine"],
                val_metrics["image_mean_norm"],
                "  <-- COLLAPSED" if val_metrics["collapsed"] else "",
            )

            # Track best model based on Recall@10
            if val_metrics["recall@10"] > best_val_recall10 + min_delta:
                best_val_recall10 = val_metrics["recall@10"]
                best_val_recall = val_metrics["recall@1"]
                best_ckpt_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    memory_queue=memory_queue,
                    epoch=epoch,
                    step=global_step,
                    config=training_config,
                    metrics=val_metrics,
                )
                epochs_without_improvement = 0
                logger.info(
                    "  New best model saved (R@10=%.4f, R@1=%.4f)",
                    best_val_recall10,
                    best_val_recall,
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    "  No improvement for %d epochs (best R@10=%.4f)",
                    epochs_without_improvement,
                    best_val_recall10,
                )

            # Early stopping check
            if early_stop_enabled and epochs_without_improvement >= early_stop_patience:
                logger.warning(
                    "  EARLY STOPPING: no improvement for %d epochs. Stopping.",
                    epochs_without_improvement,
                )
                break

            # Capture first epoch metrics for comparison
            if first_epoch_metrics is None:
                first_epoch_metrics = {
                    "loss": avg_loss,
                    "recall@1": val_metrics["recall@1"],
                    "recall@5": val_metrics["recall@5"],
                    "recall@10": val_metrics["recall@10"],
                    "temperature": model.temperature.item(),
                    "image_dim_variance": val_metrics["image_dim_variance"],
                    "text_dim_variance": val_metrics["text_dim_variance"],
                    "lr": current_lr,
                }

        training_logger.log_epoch(epoch, epoch_metrics)
        training_logger.flush()

        logger.info(
            "  Epoch %d/%d complete (%.1fs): loss=%.4f, lr=%.2e, best_val_R@10=%.4f",
            epoch + 1,
            num_epochs,
            epoch_elapsed,
            avg_loss,
            current_lr,
            best_val_recall10,
        )

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                memory_queue=memory_queue,
                epoch=epoch,
                step=global_step,
                config=training_config,
            )

    # ---- Step 9: Final summary ----
    total_elapsed = time.time() - training_start

    # Capture last epoch metrics
    last_epoch_metrics = {
        "loss": avg_loss,
        "recall@1": val_metrics["recall@1"] if (epoch + 1) % eval_every == 0 else 0.0,
        "recall@5": val_metrics["recall@5"] if (epoch + 1) % eval_every == 0 else 0.0,
        "recall@10": val_metrics["recall@10"] if (epoch + 1) % eval_every == 0 else 0.0,
        "temperature": model.temperature.item(),
        "image_dim_variance": (
            val_metrics.get("image_dim_variance", 0.0)
            if (epoch + 1) % eval_every == 0
            else 0.0
        ),
        "text_dim_variance": (
            val_metrics.get("text_dim_variance", 0.0)
            if (epoch + 1) % eval_every == 0
            else 0.0
        ),
        "lr": current_lr,
    }

    logger.info("=" * 70)
    logger.info("Phase 4 Baseline Training Complete")
    logger.info("=" * 70)
    logger.info(
        "  Total time: %.1f seconds (%.1f minutes)", total_elapsed, total_elapsed / 60
    )
    logger.info("  Total epochs: %d", epoch + 1)
    logger.info("  Total steps: %d", global_step)
    logger.info("  Best val Recall@10: %.4f", best_val_recall10)
    logger.info("  Best val Recall@1:  %.4f", best_val_recall)
    logger.info("  Final loss: %.4f", avg_loss)
    logger.info("  Memory queue: %s", "enabled" if use_queue else "disabled")
    logger.info("  Checkpoints saved to: %s", checkpoint_dir)
    logger.info("  TensorBoard logs at: %s", log_dir)

    # ---- First vs Last Epoch Comparison ----
    if first_epoch_metrics and last_epoch_metrics:
        logger.info("")
        logger.info("--- First vs Last Epoch Comparison ---")
        logger.info(
            "  Loss:           %.4f -> %.4f (%.1f%% %s)",
            first_epoch_metrics["loss"],
            last_epoch_metrics["loss"],
            abs(last_epoch_metrics["loss"] - first_epoch_metrics["loss"])
            / max(first_epoch_metrics["loss"], 1e-8)
            * 100,
            (
                "decrease"
                if last_epoch_metrics["loss"] < first_epoch_metrics["loss"]
                else "increase"
            ),
        )
        logger.info(
            "  Recall@1:       %.4f -> %.4f",
            first_epoch_metrics["recall@1"],
            last_epoch_metrics["recall@1"],
        )
        logger.info(
            "  Recall@5:       %.4f -> %.4f",
            first_epoch_metrics["recall@5"],
            last_epoch_metrics["recall@5"],
        )
        logger.info(
            "  Recall@10:      %.4f -> %.4f",
            first_epoch_metrics["recall@10"],
            last_epoch_metrics["recall@10"],
        )
        logger.info(
            "  Temperature:    %.4f -> %.4f",
            first_epoch_metrics["temperature"],
            last_epoch_metrics["temperature"],
        )
        logger.info(
            "  Image emb var:  %.6f -> %.6f",
            first_epoch_metrics["image_dim_variance"],
            last_epoch_metrics["image_dim_variance"],
        )
        logger.info(
            "  Text emb var:   %.6f -> %.6f",
            first_epoch_metrics["text_dim_variance"],
            last_epoch_metrics["text_dim_variance"],
        )
        logger.info(
            "  Learning rate:  %.2e -> %.2e",
            first_epoch_metrics["lr"],
            last_epoch_metrics["lr"],
        )

        # Learning progress assessment
        recall_improved = (
            last_epoch_metrics["recall@10"] > first_epoch_metrics["recall@10"]
        )
        loss_decreased = last_epoch_metrics["loss"] < first_epoch_metrics["loss"]
        logger.info("")
        logger.info("--- Learning Progress Assessment ---")
        if loss_decreased and recall_improved:
            logger.info(
                "  STATUS: Model is LEARNING — loss decreased and Recall@10 improved."
            )
        elif loss_decreased:
            logger.info(
                "  STATUS: Loss decreasing but Recall@10 not yet improving — needs more epochs."
            )
        elif recall_improved:
            logger.info(
                "  STATUS: Recall@10 improving despite loss not decreasing — possible regime change."
            )
        else:
            logger.info(
                "  STATUS: NO LEARNING DETECTED — loss and Recall@10 both stagnant."
            )

    logger.info("=" * 70)

    # Save final checkpoint
    final_ckpt_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        memory_queue=memory_queue,
        epoch=num_epochs - 1,
        step=global_step,
        config=training_config,
    )
    logger.info("Final checkpoint saved: %s", final_ckpt_path)

    training_logger.close()


if __name__ == "__main__":
    main()
