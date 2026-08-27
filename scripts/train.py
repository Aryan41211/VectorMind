"""Phase 4 full training entry point for VectorMind.

Purpose: take the CLI flags and config files, assemble the data, model,
optimizer, scheduler, and memory queue, and hand them to the single
shared training loop in ``vectormind.training.trainer``.

This is the main entry point for Phase 4 of the ROADMAP. It replaces the
three separate Phase 4 scripts (``resume_training.py``,
``benchmark_epoch.py``, ``hyperparameter_experiment.py``) — their
handling of AMP, gradient accumulation, the memory queue, scheduler
stepping, checkpointing, and early stopping now lives in exactly one
place instead of four copies of the same loop.

Usage:
    python scripts/train.py
    python scripts/train.py --resume checkpoints/train/latest.pt

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import argparse
import logging
import sys
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
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.checkpoint import (
    load_checkpoint,
    read_checkpoint_metric,
)
from vectormind.training.logger import TrainingLogger
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.train_loop import create_optimizer, create_scaler
from vectormind.training.trainer import train
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


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
    # The config's memory_queue.enabled decides; these two override it
    # in either direction for a one-off run, and cannot both be given.
    queue_group = parser.add_mutually_exclusive_group()
    queue_group.add_argument(
        "--no-queue",
        action="store_true",
        default=False,
        help="Force the memory queue off, whatever the config says.",
    )
    queue_group.add_argument(
        "--queue",
        action="store_true",
        default=False,
        help="Force the memory queue on (it is off by default — see §6).",
    )
    parser.add_argument(
        "--uniformity-weight",
        type=float,
        default=None,
        help="Override uniformity.weight from configs/training.yaml.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataset.num_workers from configs/data.yaml.",
    )
    # Sweep arms must not write into the shipped run's directory. Two
    # arms sharing checkpoints/train would overwrite each other's
    # epoch_0NN.pt files, and an arm that beat the shipped R@10 would
    # replace best_model.pt with weights trained under a different loss.
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override training.checkpoint_dir (isolates an experiment arm).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override training.log_dir (isolates an experiment arm).",
    )
    return parser.parse_args()


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

    # Apply CLI overrides to the configs. The trainer reads everything
    # from configs, so overrides are written back rather than passed
    # separately — one source of truth for the run.
    num_epochs = args.epochs or train_cfg["epochs"]
    lr = args.lr or optim_cfg["lr"]

    if args.no_queue:
        mq_cfg["enabled"] = False
    elif args.queue:
        mq_cfg["enabled"] = True
    if args.uniformity_weight is not None:
        train_cfg.setdefault("uniformity", {})["weight"] = args.uniformity_weight
    if args.checkpoint_dir is not None:
        train_cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        train_cfg["log_dir"] = args.log_dir

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
    # at 4096-against-128 (see MemoryQueue's class docstring). The loop
    # in vectormind.training.trainer activates it after `warmup_epochs`.
    use_queue = bool(mq_cfg.get("enabled", False))
    queue_warmup_epochs = int(mq_cfg.get("warmup_epochs", 0))

    # --no-queue deactivates the queue rather than replacing it with a
    # size-1 stub. The stub could not be resumed from any real
    # checkpoint: load_checkpoint compares queue_size and rejected a
    # 4096-entry checkpoint against a 1-entry queue, so the baseline
    # experiment could only ever start from scratch. Keeping the real
    # queue and leaving it inactive costs 4MB, makes --no-queue usable
    # mid-run, and means "no queue" describes the loss rather than the
    # object.
    memory_queue = MemoryQueue(
        queue_size=mq_cfg["queue_size"],
        embed_dim=model_config["embedding"]["shared_dim"],
        device=device,
        active=use_queue and queue_warmup_epochs <= 0,
    )
    logger.info(
        "  Memory queue: %s (size=%d, warmup=%d epochs)",
        "ENABLED" if use_queue else "DISABLED",
        mq_cfg["queue_size"],
        queue_warmup_epochs,
    )

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
            args.resume,
            model,
            optimizer,
            scaler,
            memory_queue,
            scheduler=scheduler,
        )
        start_epoch += 1  # Resume from next epoch
        logger.info("  Resumed: epoch=%d, step=%d", start_epoch, global_step)
    else:
        logger.info("Step 6: Starting from scratch.")

    # ---- Step 7: Initialize logger ----
    logger.info("Step 7: Initializing TensorBoard logger...")
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    log_dir = Path(train_cfg["log_dir"])
    logger.info("  Checkpoints -> %s, TensorBoard -> %s", checkpoint_dir, log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_logger = TrainingLogger(log_dir=log_dir)

    # ---- Step 8: Shared training loop ----
    # Recover the best-so-far from the existing best checkpoint. Starting
    # at 0.0 after a resume means the first completed epoch always wins
    # the comparison and overwrites best_model.pt however bad it is —
    # which replaced a 17.46% R@10 checkpoint with a 10.51% one.
    best_val_recall10 = 0.0
    if args.resume:
        best_val_recall10 = read_checkpoint_metric(
            checkpoint_dir / "best_model.pt", "recall@10"
        )
        logger.info("  Best-so-far restored: val R@10=%.4f", best_val_recall10)

    summary = train(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        memory_queue=memory_queue,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        training_logger=training_logger,
        train_config=train_cfg,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        global_step=global_step,
        best_val_recall10=best_val_recall10,
    )

    logger.info("=" * 70)
    logger.info("Phase 4 Baseline Training Complete")
    logger.info("=" * 70)
    logger.info("  Total epochs: %d", summary["epochs_run"])
    logger.info("  Total steps: %d", summary["total_steps"])
    logger.info("  Best val Recall@10: %.4f", summary["best_val_recall10"])
    logger.info("  Best val Recall@1:  %.4f", summary["best_val_recall1"])
    logger.info("  Checkpoints saved to: %s", checkpoint_dir)
    logger.info("  TensorBoard logs at: %s", log_dir)


if __name__ == "__main__":
    main()