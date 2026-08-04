"""Phase 4 benchmark: measure one epoch of training on full Flickr30k.

Purpose: empirically measure training throughput, GPU/CPU utilization,
memory usage, and component timings to produce accurate runtime
estimates for 10/20/40 epoch runs.

This script runs EXACTLY ONE epoch and stops. It is not a training
script — it is a measurement instrument.

Usage:
    python scripts/benchmark_epoch.py

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
from vectormind.data.splitter import create_splits
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms, get_train_transforms
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

# Directories
CHECKPOINT_DIR = Path("checkpoints/benchmark")
LOG_DIR = Path("logs/benchmark")


def compute_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
    k: int = 1,
) -> float:
    """Compute image-level Recall@K for image-to-text retrieval.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image.
        k: Number of top results to consider.

    Returns:
        Image-level Recall@K as a float between 0 and 1.
    """
    N_images = image_embeds.shape[0]
    similarity = image_embeds @ text_embeds.T

    correct_indices = []
    for i in range(N_images):
        start = i * captions_per_image
        end = start + captions_per_image
        correct_indices.append(set(range(start, end)))

    _, top_k_indices = similarity.topk(k, dim=1)

    correct_count = 0
    for i in range(N_images):
        top_k_set = set(top_k_indices[i].tolist())
        if top_k_set & correct_indices[i]:
            correct_count += 1

    return correct_count / N_images


@torch.no_grad()
def evaluate(
    model: VectorMindModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    captions_per_image: int = 5,
) -> dict[str, float]:
    """Evaluate the model on a dataset split.

    Args:
        model: The trained VectorMindModel.
        dataloader: DataLoader yielding batches.
        device: Device to evaluate on.
        captions_per_image: Captions per image.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    all_image_embeds: list[torch.Tensor] = []
    all_text_embeds: list[torch.Tensor] = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(input_ids, attention_mask)
        all_image_embeds.append(img_emb)
        all_text_embeds.append(txt_emb)

    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    N_total = image_embeds.shape[0]
    N_unique = N_total // captions_per_image
    image_embeds_unique = image_embeds.view(N_unique, captions_per_image, -1).mean(
        dim=1
    )

    r1 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=1)
    r5 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=5)
    r10 = compute_recall_at_k(
        image_embeds_unique, text_embeds, captions_per_image, k=10
    )

    return {
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
    }


def get_gpu_stats() -> dict[str, float]:
    """Collect GPU memory and utilization statistics.

    Returns:
        Dictionary with GPU memory stats in GB.
    """
    if not torch.cuda.is_available():
        return {
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_max_allocated_gb": 0.0,
        }

    return {
        "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def main() -> None:
    """Run exactly one epoch of training and collect benchmark metrics."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 70)
    logger.info("VectorMind Phase 4 — ONE-EPOCH BENCHMARK")
    logger.info("=" * 70)

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

    # Performance-tuned DataLoader settings for RTX 4050 6GB
    data_config_optimized = dict(data_config)
    data_config_optimized["dataset"] = dict(data_config["dataset"])
    data_config_optimized["dataset"]["batch_size"] = 128
    data_config_optimized["dataset"]["num_workers"] = 8
    data_config_optimized["dataset"]["pin_memory"] = True
    data_config_optimized["dataset"]["persistent_workers"] = True
    data_config_optimized["dataset"]["prefetch_factor"] = 4

    # ---- Step 2: Load and split dataset ----
    logger.info("Step 2: Loading Flickr30k dataset...")
    t0 = time.time()
    cache_dir = data_config["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)
    data_load_time = time.time() - t0
    logger.info(
        "  Loaded %d pairs (%d unique images) in %.1fs",
        len(image_paths),
        len(set(image_paths)),
        data_load_time,
    )

    logger.info("Step 2b: Splitting dataset...")
    t0 = time.time()
    train_pairs, val_pairs, test_pairs = create_splits(
        config=data_config,
        image_paths=[Path(p) for p in image_paths],
        captions=captions,
    )
    split_time = time.time() - t0
    logger.info(
        "  Train: %d pairs, Val: %d pairs, Test: %d pairs (split in %.1fs)",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
        split_time,
    )

    # ---- Step 3: Build DataLoaders ----
    logger.info("Step 3: Building DataLoaders (optimized)...")
    t0 = time.time()
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
    dataloader_time = time.time() - t0
    batch_size = data_config_optimized["dataset"]["batch_size"]
    num_workers = data_config_optimized["dataset"]["num_workers"]
    logger.info(
        "  DataLoader ready: %d train batches, %d val batches (batch_size=%d, "
        "num_workers=%d, persistent_workers=True, prefetch_factor=4) in %.1fs",
        len(train_loader),
        len(val_loader),
        batch_size,
        num_workers,
        dataloader_time,
    )

    # ---- Step 4: Initialize model ----
    logger.info("Step 4: Initializing model...")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)
    model_init_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  Model: %d parameters, device=%s (init in %.1fs)",
        n_params,
        device,
        model_init_time,
    )

    # ---- Step 5: Initialize optimizer, scaler, scheduler, queue ----
    logger.info("Step 5: Initializing optimizer and scheduler...")
    t0 = time.time()
    lr = optim_cfg["lr"]
    optimizer = create_optimizer(model, lr=lr, weight_decay=optim_cfg["weight_decay"])
    scaler = create_scaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["T_max"],
        eta_min=sched_cfg["eta_min"],
    )

    memory_queue = MemoryQueue(
        queue_size=mq_cfg["queue_size"],
        embed_dim=model_config["embedding"]["shared_dim"],
        device=device,
    )
    optim_init_time = time.time() - t0
    logger.info(
        "  Optimizer: AdamW (lr=%.1e, wd=%.4f), Scheduler: cosine (T_max=%d, "
        "eta_min=%.1e), Memory queue: size=%d (init in %.1fs)",
        lr,
        optim_cfg["weight_decay"],
        sched_cfg["T_max"],
        sched_cfg["eta_min"],
        mq_cfg["queue_size"],
        optim_init_time,
    )

    # ---- Step 6: Initialize logger and checkpoint dir ----
    logger.info("Step 6: Initializing TensorBoard logger...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    training_logger = TrainingLogger(log_dir=LOG_DIR)

    # ---- Step 7: Benchmark — ONE EPOCH ----
    log_every = 50
    accum_steps = training_config.get("gradient_accumulation_steps", 1)

    logger.info("Step 7: Starting BENCHMARK — exactly 1 epoch...")
    logger.info(
        "  Effective batch: %d x %d = %d",
        batch_size,
        accum_steps,
        batch_size * accum_steps,
    )
    logger.info("  Total train batches: %d", len(train_loader))

    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    benchmark_start = time.time()
    model.train()

    batch_times: list[float] = []
    data_load_times: list[float] = []
    forward_times: list[float] = []
    optimizer_times: list[float] = []
    queue_times: list[float] = []
    epoch_losses: list[float] = []
    epoch_grad_norms: list[float] = []

    prev_batch_end = time.time()

    for batch_idx, batch in enumerate(train_loader):
        step_start = time.time()

        # Measure data loading time (time between previous batch end and this batch start)
        data_load_times.append(time.time() - prev_batch_end)

        # Forward + backward
        forward_start = time.time()
        metrics = train_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            memory_queue=memory_queue,
            accumulation_steps=accum_steps,
            device=device,
        )
        forward_end = time.time()
        forward_times.append(forward_end - forward_start)

        # Optimizer step at accumulation boundary
        opt_start = time.time()
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        optimizer_times.append(time.time() - opt_start)

        # Enqueue text embeddings into memory queue
        q_start = time.time()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            text_embeds = model.encode_text(input_ids, attention_mask)
            memory_queue.enqueue(text_embeds)
        queue_times.append(time.time() - q_start)

        # Gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm**0.5

        epoch_losses.append(metrics["loss"])
        epoch_grad_norms.append(grad_norm)

        step_time = time.time() - step_start
        batch_times.append(step_time)

        # Log per-step
        if batch_idx % log_every == 0:
            step_metrics = {
                "train/loss": metrics["loss"],
                "train/temperature": metrics["temperature"],
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/gpu_memory_gb": metrics.get("gpu_memory_gb", 0.0),
            }
            training_logger.log_metrics(batch_idx, step_metrics)

            gpu_stats = get_gpu_stats()
            logger.info(
                "  Batch %d/%d: loss=%.4f, temp=%.4f, step=%.3fs, "
                "gpu_alloc=%.2fGB, gpu_max=%.2fGB",
                batch_idx + 1,
                len(train_loader),
                metrics["loss"],
                metrics["temperature"],
                step_time,
                gpu_stats["gpu_allocated_gb"],
                gpu_stats["gpu_max_allocated_gb"],
            )

        prev_batch_end = time.time()

    # ---- End of epoch ----
    epoch_elapsed = time.time() - benchmark_start
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)

    # Step the scheduler
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # ---- Validation timing ----
    logger.info("  Running validation benchmark...")
    torch.cuda.empty_cache()
    val_start = time.time()
    val_metrics = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        captions_per_image=5,
    )
    val_time = time.time() - val_start

    logger.info(
        "  Val: R@1=%.4f, R@5=%.4f, R@10=%.4f (in %.1fs)",
        val_metrics["recall@1"],
        val_metrics["recall@5"],
        val_metrics["recall@10"],
        val_time,
    )

    # ---- Checkpoint timing ----
    logger.info("  Running checkpoint benchmark...")
    ckpt_start = time.time()
    ckpt_path = CHECKPOINT_DIR / "benchmark_epoch_001.pt"
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        memory_queue=memory_queue,
        epoch=0,
        step=len(train_loader),
        config=training_config,
    )
    ckpt_time = time.time() - ckpt_start
    ckpt_size_mb = ckpt_path.stat().st_size / (1024**2)

    # ---- TensorBoard flush timing ----
    tb_start = time.time()
    epoch_metrics = {
        "epoch/avg_loss": avg_loss,
        "epoch/avg_grad_norm": avg_grad_norm,
        "epoch/temperature": model.temperature.item(),
        "epoch/lr": current_lr,
        "epoch/epoch_time_s": epoch_elapsed,
        "epoch/memory_queue_size": memory_queue.current_size,
        "val/recall@1": val_metrics["recall@1"],
        "val/recall@5": val_metrics["recall@5"],
        "val/recall@10": val_metrics["recall@10"],
    }
    training_logger.log_epoch(0, epoch_metrics)
    training_logger.flush()
    tb_time = time.time() - tb_start

    training_logger.close()

    # ---- Collect final GPU stats ----
    gpu_stats = get_gpu_stats()

    # ---- Compute DataLoader efficiency ----
    avg_data_load = (
        sum(data_load_times) / len(data_load_times) if data_load_times else 0
    )
    max_data_load = max(data_load_times) if data_load_times else 0
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_forward = sum(forward_times) / len(forward_times)
    avg_optimizer = sum(optimizer_times) / len(optimizer_times)
    avg_queue = sum(queue_times) / len(queue_times)

    # Compute CPU time vs GPU time estimate
    # Forward+backward is GPU-bound; data loading should be hidden by prefetch
    total_data_load = sum(data_load_times)
    total_forward = sum(forward_times)
    total_optimizer = sum(optimizer_times)
    total_queue = sum(queue_times)

    # Samples per second
    total_samples = len(train_pairs)
    samples_per_sec = total_samples / epoch_elapsed

    # ---- Print comprehensive report ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS — ONE EPOCH")
    logger.info("=" * 70)
    logger.info("")

    logger.info("--- Dataset ---")
    logger.info("  Total pairs:       %d", total_samples)
    logger.info("  Train pairs:       %d", len(train_pairs))
    logger.info("  Val pairs:         %d", len(val_pairs))
    logger.info("  Batch size:        %d", batch_size)
    logger.info("  Batches per epoch: %d", len(train_loader))
    logger.info("  num_workers:       %d", num_workers)
    logger.info("  persistent_workers: True")
    logger.info("  prefetch_factor:   4")
    logger.info("")

    logger.info("--- Timing ---")
    logger.info(
        "  Epoch duration:      %.1f seconds (%.1f minutes)",
        epoch_elapsed,
        epoch_elapsed / 60,
    )
    logger.info("  Avg batch time:      %.3f seconds", avg_batch_time)
    logger.info("  Avg forward+backward: %.3f seconds", avg_forward)
    logger.info("  Avg optimizer step:  %.3f seconds", avg_optimizer)
    logger.info("  Avg queue enqueue:   %.3f seconds", avg_queue)
    logger.info("  Avg data load time:  %.3f seconds", avg_data_load)
    logger.info("  Max data load time:  %.3f seconds", max_data_load)
    logger.info("  Validation time:     %.1f seconds", val_time)
    logger.info("  Checkpoint time:     %.3f seconds", ckpt_time)
    logger.info("  TensorBoard time:    %.3f seconds", tb_time)
    logger.info("")

    logger.info("--- Throughput ---")
    logger.info("  Samples/second:      %.1f", samples_per_sec)
    logger.info("  Batches/second:      %.2f", len(train_loader) / epoch_elapsed)
    logger.info("")

    logger.info("--- Memory ---")
    logger.info("  GPU allocated:       %.2f GB", gpu_stats["gpu_allocated_gb"])
    logger.info("  GPU reserved:        %.2f GB", gpu_stats["gpu_reserved_gb"])
    logger.info("  GPU peak allocated:  %.2f GB", gpu_stats["gpu_max_allocated_gb"])
    logger.info("  Checkpoint size:     %.1f MB", ckpt_size_mb)
    logger.info("")

    logger.info("--- Component Time Breakdown ---")
    logger.info(
        "  Forward+backward:    %.1f%% (%.1fs)",
        total_forward / epoch_elapsed * 100,
        total_forward,
    )
    logger.info(
        "  Optimizer step:      %.1f%% (%.1fs)",
        total_optimizer / epoch_elapsed * 100,
        total_optimizer,
    )
    logger.info(
        "  Queue enqueue:       %.1f%% (%.1fs)",
        total_queue / epoch_elapsed * 100,
        total_queue,
    )
    logger.info(
        "  Data loading (sum):  %.1f%% (%.1fs)",
        total_data_load / epoch_elapsed * 100,
        total_data_load,
    )
    logger.info(
        "  Other (logging etc): %.1f%% (%.1fs)",
        (
            1
            - (total_forward + total_optimizer + total_queue + total_data_load)
            / epoch_elapsed
        )
        * 100,
        epoch_elapsed - total_forward - total_optimizer - total_queue - total_data_load,
    )
    logger.info("")

    logger.info("--- Training Metrics ---")
    logger.info("  Avg loss:            %.4f", avg_loss)
    logger.info("  Avg gradient norm:   %.4f", avg_grad_norm)
    logger.info("  Temperature:         %.4f", model.temperature.item())
    logger.info("  Learning rate:       %.2e", current_lr)
    logger.info(
        "  Memory queue size:   %d / %d",
        memory_queue.current_size,
        mq_cfg["queue_size"],
    )
    logger.info("")

    logger.info("--- Validation (1 epoch) ---")
    logger.info("  Recall@1:            %.4f", val_metrics["recall@1"])
    logger.info("  Recall@5:            %.4f", val_metrics["recall@5"])
    logger.info("  Recall@10:           %.4f", val_metrics["recall@10"])
    logger.info(
        "  Random baseline R@10: %.4f (10 / %d)", 10 / len(val_pairs), len(val_pairs)
    )
    logger.info("")

    logger.info("--- Runtime Estimates (based on this benchmark) ---")
    est_10 = epoch_elapsed * 10
    est_20 = epoch_elapsed * 20
    est_40 = epoch_elapsed * 40
    logger.info(
        "  10 epochs:  %.0f seconds (%.1f minutes / %.1f hours)",
        est_10,
        est_10 / 60,
        est_10 / 3600,
    )
    logger.info(
        "  20 epochs:  %.0f seconds (%.1f minutes / %.1f hours)",
        est_20,
        est_20 / 60,
        est_20 / 3600,
    )
    logger.info(
        "  40 epochs:  %.0f seconds (%.1f minutes / %.1f hours)",
        est_40,
        est_40 / 60,
        est_40 / 3600,
    )
    logger.info(
        "  Storage/epoch: ~%.1f MB (checkpoints) + ~%.1f MB (TensorBoard)",
        ckpt_size_mb,
        0.1,
    )
    logger.info(
        "  Storage/10 epochs: ~%.0f MB", ckpt_size_mb * 5
    )  # save every 2 epochs
    logger.info("")

    # ---- Bottleneck analysis ----
    logger.info("--- Bottleneck Analysis ---")
    data_pct = total_data_load / epoch_elapsed * 100
    compute_pct = total_forward / epoch_elapsed * 100
    if data_pct > 30:
        logger.info(
            "  PRIMARY BOTTLENECK: Data loading (%.1f%% of epoch time)", data_pct
        )
        logger.info("  Recommendation: increase num_workers or prefetch_factor")
    elif compute_pct > 70:
        logger.info(
            "  PRIMARY BOTTLENECK: GPU compute (%.1f%% of epoch time)", compute_pct
        )
        logger.info(
            "  Recommendation: GPU is well-utilized. Consider AMP or smaller model if faster training needed."
        )
    else:
        logger.info("  BALANCED: No single dominant bottleneck")
        logger.info(
            "  Data loading: %.1f%%, GPU compute: %.1f%%", data_pct, compute_pct
        )

    # Check if data loading is hidden by prefetch
    avg_batch_compute = avg_forward + avg_optimizer + avg_queue
    if avg_data_load < avg_batch_compute:
        logger.info(
            "  DataLoader prefetch is EFFECTIVE (avg load %.3fs < avg compute %.3fs)",
            avg_data_load,
            avg_batch_compute,
        )
    else:
        logger.info(
            "  DataLoader prefetch is INSUFFICIENT (avg load %.3fs > avg compute %.3fs)",
            avg_data_load,
            avg_batch_compute,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK COMPLETE — STOPPED AFTER 1 EPOCH (as designed)")
    logger.info("=" * 70)
    logger.info("  Checkpoint: %s", ckpt_path)
    logger.info("  TensorBoard: %s", LOG_DIR)
    logger.info("=" * 70)

    # Clean up benchmark checkpoint
    ckpt_path.unlink(missing_ok=True)
    logger.info("  Benchmark checkpoint cleaned up.")


if __name__ == "__main__":
    main()
