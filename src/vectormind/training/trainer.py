"""The single shared training loop for all VectorMind runs.

Purpose: run the multi-epoch loop (AMP, gradient accumulation, memory
queue warmup, scheduler stepping, validation cadence, best/periodic/final
checkpointing, early stopping) in exactly one place.

This module replaces the four Phase 4 scripts' four copies of the loop
(scripts/train.py, resume_training.py, benchmark_epoch.py,
hyperparameter_experiment.py). Each of those drifted from the others —
different schedules, different summary logic, and one that silently
logged the *initial* learning rate to TensorBoard for the entire run
while the real LR decayed. The loop now belongs to the library;
``scripts/train.py`` is a thin CLI over it.

Design decisions (locked in ARCHITECTURE.md §7):
- All loop knobs come from ``train_config`` — never hardcoded here.
- The logger and evaluator are injected so the loop never depends on
  TensorBoard or evaluation internals.
- ``train/lr`` is read from the scheduler each step. A static LR looked
  flat in TensorBoard while training was actually decaying.
- The memory queue is a caller-provided object; the loop only decides
  when to activate it (warmup) and reads its size for metrics.

Input:
  All state is passed in keyword-only. See ``train``.

Output:
  Summary dict: total steps, best validation recall, first/last epoch
  metrics, and the number of epochs actually run (which is less than
  ``num_epochs`` when early stopping or instability stops the run).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import torch

from vectormind.evaluation.evaluator import evaluate_split
from vectormind.models.vectormind_model import (
    DEFAULT_MAX_LOGIT_SCALE,
    VectorMindModel,
)
from vectormind.training.checkpoint import read_checkpoint_metric, save_checkpoint
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.oom import run_step_with_oom_retry
from vectormind.training.train_loop import train_one_step

logger = logging.getLogger(__name__)

EvaluateFn = Callable[
    [VectorMindModel, torch.utils.data.DataLoader, torch.device, int],
    dict[str, float],
]


class MetricLogger(Protocol):
    """The subset of TrainingLogger the loop needs.

    Declared as a protocol so the loop is testable with a recording
    stand-in and never pulls in TensorBoard at import time.
    """

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log cumulative step metrics at a global step."""
        ...

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log per-epoch aggregates."""
        ...

    def flush(self) -> None:
        """Persist buffered log writes."""
        ...

    def close(self) -> None:
        """Release logging resources at the end of training."""
        ...


def _default_evaluate(
    model: VectorMindModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    captions_per_image: int,
) -> dict[str, float]:
    """Evaluate via the shared evaluation module.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader over the split to evaluate.
        device: Device to evaluate on.
        captions_per_image: Captions per image for the recall cutoffs.

    Returns:
        Flat metric mapping (``recall@1/5/10``, embedding diagnostics,
        health fields). See
        :meth:`vectormind.evaluation.evaluator.EvaluationResult.to_flat_dict`.
    """
    return evaluate_split(
        model, dataloader, device, captions_per_image
    ).to_flat_dict()


def train(
    *,
    model: VectorMindModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    memory_queue: MemoryQueue,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    training_logger: MetricLogger,
    train_config: dict[str, Any],
    num_epochs: int,
    start_epoch: int = 0,
    global_step: int = 0,
    best_val_recall10: float = 0.0,
    evaluate: EvaluateFn | None = None,
    captions_per_image: int = 5,
    log_every_steps: int = 50,
) -> dict[str, Any]:
    """Run the shared training loop.

    Args:
        model: The dual-encoder model to train.
        optimizer: The optimizer instance (typically AdamW).
        scaler: GradScaler for mixed precision.
        scheduler: LR scheduler stepped once per epoch.
        memory_queue: Memory queue; the loop activates it after warmup.
        train_loader: DataLoader over the training split.
        val_loader: DataLoader over the validation split.
        device: Device to train on.
        training_logger: Object with ``log_metrics``/``log_epoch``/
            ``flush``/``close`` (see MetricLogger).
        train_config: Full training config dict. Read keys:
            ``checkpoint_dir``, ``eval_every_n_epochs``,
            ``save_every_n_epochs``, ``gradient_accumulation_steps``,
            ``memory_queue.enabled``, ``memory_queue.warmup_epochs``,
            ``early_stopping.*``, ``uniformity.weight``,
            ``temperature.clamp_enabled``, ``temperature.max_logit_scale``.
        num_epochs: Total epochs for this run (used as the loop upper
            bound after ``start_epoch``).
        start_epoch: Epoch to resume from; the loop runs
            ``[start_epoch, num_epochs)``. Scheduler state must already
            match (restore it before calling — see ``load_checkpoint``).
        global_step: Global step counter to resume from.
        best_val_recall10: Best validation R@10 seen so far; pass the
            restored value on resume so a stale checkpoint cannot be
            overwritten by a worse one.
        evaluate: Callable(model, dataloader, device, captions_per_image)
            -> flat metric dict. Defaults to the shared evaluator.
        captions_per_image: Captions per image for recall computation.
        log_every_steps: Log cumulative step metrics every N steps.

    Returns:
        Summary dict with keys ``total_steps``, ``epochs_run``,
        ``best_val_recall10``, ``best_val_recall1``,
        ``first_epoch_metrics``, ``last_epoch_metrics``.

    Raises:
        ValueError: If ``train_config`` lacks a required key.

    Assumptions:
        The optimizer, scaler, scheduler, and memory queue state match
        ``start_epoch``/``global_step`` (restore via checkpoint).
    """
    logger.info("Starting shared training loop for %d epochs...", num_epochs)
    _ = read_checkpoint_metric  # imported for callers that recover best recall

    checkpoint_dir = Path(train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    eval_every = int(train_config["eval_every_n_epochs"])
    save_every = int(train_config["save_every_n_epochs"])
    accum_steps = int(train_config.get("gradient_accumulation_steps", 1))

    mq_cfg = train_config.get("memory_queue", {})
    use_queue = bool(mq_cfg.get("enabled", False))
    queue_warmup_epochs = int(mq_cfg.get("warmup_epochs", 0))

    early_cfg = train_config.get("early_stopping", {})
    early_stop_enabled = bool(early_cfg.get("enabled", True))
    early_stop_patience = int(early_cfg.get("patience", 5))
    min_delta = float(early_cfg.get("min_delta", 0.001))

    uniformity_weight = float(
        train_config.get("uniformity", {}).get("weight", 0.0)
    )

    temp_cfg = train_config.get("temperature", {})
    clamp_enabled = bool(temp_cfg.get("clamp_enabled", True))
    max_logit_scale = float(
        temp_cfg.get("max_logit_scale", DEFAULT_MAX_LOGIT_SCALE)
    )

    eval_fn = evaluate or _default_evaluate

    best_val_recall1 = 0.0
    epochs_without_improvement = 0
    loss_ema: float | None = None
    first_epoch_metrics: dict[str, float] | None = None
    last_eval_metrics: dict[str, float] | None = None
    training_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Activate the queue once the encoder has stabilized. It has been
        # filling throughout warmup, so it switches on already full.
        if use_queue and not memory_queue.active and epoch >= queue_warmup_epochs:
            memory_queue.activate()

        model.train()
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []

        # The scheduler advances only at epoch end, so every step in this
        # epoch trains at the same LR. Cache it once — this is also the
        # value TensorBoard must show, not the static initial LR (#7).
        current_lr = float(scheduler.get_last_lr()[0])

        for batch_idx, batch in enumerate(train_loader):
            step_fn: Callable[[], dict[str, float]] = partial(
                train_one_step,
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                memory_queue=memory_queue,
                accumulation_steps=accum_steps,
                device=device,
                uniformity_weight=uniformity_weight,
            )
            metrics = run_step_with_oom_retry(
                step_fn,
                context=f"epoch {epoch + 1} step {batch_idx + 1}",
            )

            # Gradient norm BEFORE the optimizer step captures the actual
            # gradients.
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm**0.5

            # Optimizer step at the accumulation boundary.
            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Clamp the learnable logit scale after each update; an
                # unclamped optimizer wins by inflating the scale and the
                # embedding space collapses into a cone
                # (docs/KNOWN_ISSUES.md §1).
                if clamp_enabled:
                    metrics["temperature"] = model.clamp_log_temperature(
                        max_logit_scale
                    )

            epoch_losses.append(metrics["loss"])
            epoch_grad_norms.append(grad_norm)

            # ---- Instability detection ----
            loss_val = metrics["loss"]
            if torch.isnan(torch.tensor(loss_val)) or torch.isinf(
                torch.tensor(loss_val)
            ):
                logger.error(
                    "  INSTABILITY DETECTED: loss is %s at step %d. "
                    "Stopping training.",
                    "NaN" if torch.isnan(torch.tensor(loss_val)) else "Inf",
                    global_step,
                )
                break

            # ---- Loss spike detection (3x EMA) ----
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss_val
                if loss_val > 3.0 * loss_ema and global_step > 100:
                    logger.warning(
                        "  LOSS SPIKE: loss=%.4f at step %d "
                        "(EMA=%.4f, ratio=%.1fx)",
                        loss_val,
                        global_step,
                        loss_val / loss_ema,
                        loss_val / loss_ema,
                    )

            # ---- Log per-step metrics ----
            if global_step % log_every_steps == 0:
                training_logger.log_metrics(
                    global_step,
                    {
                        "train/loss": metrics["loss"],
                        "train/temperature": metrics["temperature"],
                        "train/image_embed_norm": metrics["image_embed_norm"],
                        "train/text_embed_norm": metrics["text_embed_norm"],
                        "train/image_embed_std": metrics["image_embed_std"],
                        "train/text_embed_std": metrics["text_embed_std"],
                        "train/grad_norm": grad_norm,
                        "train/lr": current_lr,
                        "train/gpu_memory_gb": metrics.get("gpu_memory_gb", 0.0),
                    },
                )
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
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_grad_norm = sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1)

        # Step the scheduler once per epoch.
        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])

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
            torch.cuda.empty_cache()
            logger.info("  Running validation...")
            val_metrics = eval_fn(model, val_loader, device, captions_per_image)
            last_eval_metrics = val_metrics
            for key, value in val_metrics.items():
                epoch_metrics[f"val/{key}"] = value

            logger.info(
                "  Val: R@1=%.4f, R@5=%.4f, R@10=%.4f, embed_var=%.6f",
                val_metrics["recall@1"],
                val_metrics["recall@5"],
                val_metrics["recall@10"],
                val_metrics["image_dim_variance"],
            )
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

            # Best model is tracked on Recall@10.
            if val_metrics["recall@10"] > best_val_recall10 + min_delta:
                best_val_recall10 = val_metrics["recall@10"]
                best_val_recall1 = val_metrics["recall@1"]
                save_checkpoint(
                    path=checkpoint_dir / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    memory_queue=memory_queue,
                    epoch=epoch,
                    step=global_step,
                    config=train_config,
                    metrics=val_metrics,
                    scheduler=scheduler,
                )
                epochs_without_improvement = 0
                logger.info(
                    "  New best model saved (R@10=%.4f, R@1=%.4f)",
                    best_val_recall10,
                    best_val_recall1,
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    "  No improvement for %d epochs (best R@10=%.4f)",
                    epochs_without_improvement,
                    best_val_recall10,
                )

            # Early stopping check.
            if early_stop_enabled and epochs_without_improvement >= early_stop_patience:
                logger.warning(
                    "  EARLY STOPPING: no improvement for %d epochs. Stopping.",
                    epochs_without_improvement,
                )
                break

            # Capture first-epoch metrics for the first-vs-last report.
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
            "  Epoch %d/%d complete (%.1fs): loss=%.4f, lr=%.2e, "
            "best_val_R@10=%.4f",
            epoch + 1,
            num_epochs,
            epoch_elapsed,
            avg_loss,
            current_lr,
            best_val_recall10,
        )

        # Save periodic checkpoint.
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                memory_queue=memory_queue,
                epoch=epoch,
                step=global_step,
                config=train_config,
                scheduler=scheduler,
            )

    # ---- Final checkpoint and summary ----
    total_elapsed = time.time() - training_start
    last_epoch_metrics = {
        "loss": avg_loss,
        "recall@1": last_eval_metrics["recall@1"] if last_eval_metrics else 0.0,
        "recall@5": last_eval_metrics["recall@5"] if last_eval_metrics else 0.0,
        "recall@10": last_eval_metrics["recall@10"] if last_eval_metrics else 0.0,
        "temperature": model.temperature.item(),
        "image_dim_variance": (
            last_eval_metrics["image_dim_variance"] if last_eval_metrics else 0.0
        ),
        "text_dim_variance": (
            last_eval_metrics["text_dim_variance"] if last_eval_metrics else 0.0
        ),
        "lr": current_lr,
    }

    logger.info("=" * 70)
    logger.info("Training Complete")
    logger.info("=" * 70)
    logger.info("  Total time: %.1f seconds (%.1f minutes)", total_elapsed, total_elapsed / 60)
    logger.info("  Total epochs: %d", epoch + 1)
    logger.info("  Total steps: %d", global_step)
    logger.info("  Best val Recall@10: %.4f", best_val_recall10)
    logger.info("  Best val Recall@1:  %.4f", best_val_recall1)
    logger.info("  Final loss: %.4f", avg_loss)
    logger.info("  Memory queue: %s", "enabled" if use_queue else "disabled")
    logger.info("  Checkpoints saved to: %s", checkpoint_dir)

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

        _log_learning_assessment(last_epoch_metrics, first_epoch_metrics)

    logger.info("=" * 70)

    save_checkpoint(
        path=checkpoint_dir / "final_model.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        memory_queue=memory_queue,
        epoch=num_epochs - 1,
        step=global_step,
        config=train_config,
        scheduler=scheduler,
    )
    logger.info("Final checkpoint saved: %s", checkpoint_dir / "final_model.pt")

    training_logger.close()

    return {
        "total_steps": global_step,
        "epochs_run": epoch + 1,
        "best_val_recall10": best_val_recall10,
        "best_val_recall1": best_val_recall1,
        "first_epoch_metrics": first_epoch_metrics,
        "last_epoch_metrics": last_epoch_metrics,
    }


def _log_learning_assessment(
    last_epoch_metrics: dict[str, float],
    first_epoch_metrics: dict[str, float],
) -> None:
    """Log whether the run is learning, based on loss and Recall@10.

    Args:
        last_epoch_metrics: Metrics from the final evaluated epoch.
        first_epoch_metrics: Metrics from the first evaluated epoch.

    Returns:
        None.
    """
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
            "  STATUS: Loss decreasing but Recall@10 not yet improving — "
            "needs more epochs."
        )
    elif recall_improved:
        logger.info(
            "  STATUS: Recall@10 improving despite loss not decreasing — "
            "possible regime change."
        )
    else:
        logger.info(
            "  STATUS: NO LEARNING DETECTED — loss and Recall@10 both stagnant."
        )