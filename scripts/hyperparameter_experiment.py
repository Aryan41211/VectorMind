"""Hyperparameter experiment: Lower learning rate.

Purpose: Test if a lower learning rate (5e-4 vs 1e-3) improves training
stability and final performance.

Hypothesis: Lower LR may provide more stable training and better final
Recall@10 by avoiding overshooting in the loss landscape.

Experiment:
- Resume from best checkpoint (Epoch 8, with memory queue enabled)
- Train for 5 epochs with lr=5e-4
- Compare against baseline (lr=1e-3)

Usage:
    python scripts/hyperparameter_experiment.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# Ensure src/ and scripts/ are on the path
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
CHECKPOINT_DIR = Path("checkpoints/experiment")
LOG_DIR = Path("logs/experiment")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VectorMind Hyperparameter Experiment: Lower LR"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/train/best_model.pt",
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate to test (default: 5e-4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    return parser.parse_args()


def compute_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
    k: int = 1,
) -> float:
    """Compute image-level Recall@K for image-to-text retrieval."""
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
    """Evaluate the model on a dataset split."""
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    
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
    image_embeds_unique = image_embeds.view(N_unique, captions_per_image, -1).mean(dim=1)
    
    r1 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=1)
    r5 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=5)
    r10 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=10)
    
    # Embedding diagnostics
    img_dim_var = image_embeds_unique.var(dim=0).mean().item()
    txt_dim_var = text_embeds.var(dim=0).mean().item()
    
    return {
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "image_dim_variance": img_dim_var,
        "text_dim_variance": txt_dim_var,
    }


def main() -> None:
    """Run hyperparameter experiment."""
    args = parse_args()
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("VectorMind Hyperparameter Experiment: Lower LR")
    logger.info("=" * 60)
    logger.info("  Testing lr=%.1e (baseline: lr=1e-3)", args.lr)
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    
    # Load configurations
    logger.info("Loading configurations...")
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    training_config = load_config("configs/training.yaml")
    require_keys(data_config, ["dataset", "transforms"])
    require_keys(model_config, ["image_encoder", "text_encoder", "embedding"])
    require_keys(training_config, ["optimizer", "scheduler", "memory_queue", "epochs"])
    
    optim_cfg = training_config["optimizer"]
    sched_cfg = training_config["scheduler"]
    mq_cfg = training_config["memory_queue"]
    
    # Optimized DataLoader settings
    data_config_optimized = dict(data_config)
    data_config_optimized["dataset"] = dict(data_config["dataset"])
    data_config_optimized["dataset"]["batch_size"] = 128
    data_config_optimized["dataset"]["num_workers"] = args.num_workers
    data_config_optimized["dataset"]["pin_memory"] = True
    data_config_optimized["dataset"]["persistent_workers"] = False
    data_config_optimized["dataset"]["prefetch_factor"] = 4
    
    # Load dataset
    logger.info("Loading Flickr30k dataset...")
    cache_dir = data_config["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)
    logger.info("  Loaded %d pairs (%d unique images)", len(image_paths), len(set(image_paths)))
    
    # Create splits
    logger.info("Splitting dataset...")
    train_pairs, val_pairs, test_pairs = create_splits(
        config=data_config,
        image_paths=[Path(p) for p in image_paths],
        captions=captions,
    )
    logger.info("  Train: %d pairs, Val: %d pairs, Test: %d pairs", len(train_pairs), len(val_pairs), len(test_pairs))
    
    # Build DataLoaders
    logger.info("Building DataLoaders...")
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
    
    # Initialize model
    logger.info("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Model: %d parameters, device=%s", n_params, device)
    
    # Initialize optimizer, scaler, scheduler
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=optim_cfg["weight_decay"])
    scaler = create_scaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["T_max"],
        eta_min=sched_cfg["eta_min"],
    )
    
    # Initialize memory queue (ENABLED)
    memory_queue = MemoryQueue(
        queue_size=mq_cfg["queue_size"],
        embed_dim=model_config["embedding"]["shared_dim"],
        device=device,
    )
    logger.info("  Memory queue: ENABLED (size=%d)", mq_cfg["queue_size"])
    
    # Resume from checkpoint
    logger.info("Resuming from checkpoint %s...", args.checkpoint)
    
    # Load checkpoint manually to handle queue size mismatch
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # Don't restore queue state if sizes don't match
    queue_state = checkpoint.get("queue", {})
    if queue_state.get("queue_size", 0) == mq_cfg["queue_size"]:
        memory_queue.queue.copy_(queue_state["tensor"])
        memory_queue.pointer = queue_state["pointer"]
        memory_queue.num_filled = queue_state["num_filled"]
        logger.info("  Restored memory queue from checkpoint")
    else:
        logger.info("  Starting with fresh memory queue")
    
    start_epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("step", 0)
    start_epoch += 1  # Resume from next epoch
    logger.info("  Resumed: epoch=%d, step=%d", start_epoch, global_step)
    
    # Initialize logger
    logger.info("Initializing TensorBoard logger...")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    training_logger = TrainingLogger(log_dir=LOG_DIR)
    
    # Training loop
    num_epochs = args.epochs
    eval_every = training_config["eval_every_n_epochs"]
    save_every = training_config["save_every_n_epochs"]
    accum_steps = training_config.get("gradient_accumulation_steps", 1)
    
    logger.info("Starting training for %d epochs (from epoch %d)...", num_epochs, start_epoch)
    
    best_val_recall = 0.0
    best_val_recall10 = 0.0
    training_start = time.time()
    epochs_without_improvement = 0
    early_stop_patience = training_config.get("early_stopping", {}).get("patience", 5)
    early_stop_enabled = training_config.get("early_stopping", {}).get("enabled", True)
    min_delta = training_config.get("early_stopping", {}).get("min_delta", 0.001)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        epoch_grad_norms = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Forward + backward
            metrics = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                memory_queue=memory_queue,
                accumulation_steps=accum_steps,
                device=device,
            )
            
            # Gradient norm (BEFORE optimizer step to capture actual gradients)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            
            # Optimizer step at accumulation boundary
            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Enqueue text embeddings into memory queue
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                text_embeds = model.encode_text(input_ids, attention_mask)
                memory_queue.enqueue(text_embeds)
            
            epoch_losses.append(metrics["loss"])
            epoch_grad_norms.append(grad_norm)
            
            # Log every 50 steps
            if global_step % 50 == 0:
                step_metrics = {
                    "train/loss": metrics["loss"],
                    "train/temperature": metrics["temperature"],
                    "train/image_embed_norm": metrics["image_embed_norm"],
                    "train/text_embed_norm": metrics["text_embed_norm"],
                    "train/grad_norm": grad_norm,
                    "train/lr": args.lr,
                }
                training_logger.log_metrics(global_step, step_metrics)
                
                logger.info(
                    "  Epoch %d, Step %d/%d: loss=%.4f, temp=%.4f, queue_size=%d",
                    epoch + 1,
                    batch_idx + 1,
                    len(train_loader),
                    metrics["loss"],
                    metrics["temperature"],
                    memory_queue.current_size,
                )
            
            global_step += 1
        
        # End of epoch
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
        
        # Validation every N epochs
        if (epoch + 1) % eval_every == 0:
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
                "  Val: R@1=%.4f, R@5=%.4f, R@10=%.4f, queue_size=%d",
                val_metrics["recall@1"],
                val_metrics["recall@5"],
                val_metrics["recall@10"],
                memory_queue.current_size,
            )
            
            # Track best model based on Recall@10
            if val_metrics["recall@10"] > best_val_recall10 + min_delta:
                best_val_recall10 = val_metrics["recall@10"]
                best_val_recall = val_metrics["recall@1"]
                best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"
                save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    memory_queue=memory_queue,
                    epoch=epoch,
                    step=global_step,
                    config=training_config,
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
        
        training_logger.log_epoch(epoch, epoch_metrics)
        training_logger.flush()
        
        logger.info(
            "  Epoch %d/%d complete (%.1fs): loss=%.4f, lr=%.2e, queue_size=%d",
            epoch + 1,
            start_epoch + num_epochs,
            epoch_elapsed,
            avg_loss,
            current_lr,
            memory_queue.current_size,
        )
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch + 1:03d}.pt"
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
    
    # Final summary
    total_elapsed = time.time() - training_start
    
    logger.info("=" * 70)
    logger.info("Hyperparameter Experiment Complete (Lower LR)")
    logger.info("=" * 70)
    logger.info("  Learning rate tested: %.1e", args.lr)
    logger.info("  Total time: %.1f seconds (%.1f minutes)", total_elapsed, total_elapsed / 60)
    logger.info("  Total epochs: %d", epoch - start_epoch + 1)
    logger.info("  Total steps: %d", global_step)
    logger.info("  Best val Recall@10: %.4f", best_val_recall10)
    logger.info("  Best val Recall@1:  %.4f", best_val_recall)
    logger.info("  Final loss: %.4f", avg_loss)
    logger.info("  Memory queue: ENABLED (size=%d)", memory_queue.current_size)
    logger.info("  Checkpoints saved to: %s", CHECKPOINT_DIR)
    logger.info("  TensorBoard logs at: %s", LOG_DIR)
    logger.info("=" * 70)
    
    # Save final checkpoint
    final_ckpt_path = CHECKPOINT_DIR / "final_model.pt"
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        memory_queue=memory_queue,
        epoch=epoch,
        step=global_step,
        config=training_config,
    )
    logger.info("Final checkpoint saved: %s", final_ckpt_path)
    
    training_logger.close()


if __name__ == "__main__":
    main()
