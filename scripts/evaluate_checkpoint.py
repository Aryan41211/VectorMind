"""Evaluate a specific checkpoint to get detailed metrics.

Purpose: Load a checkpoint and run full evaluation to get Recall@1/5/10
and embedding diagnostics.

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/train/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
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
from vectormind.evaluation.memorization import compute_image_level_recall
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.utils.config import load_config

# Recall@K has one implementation, in the package. Aliased here under
# the name this script has always used (CLAUDE.md §3: no duplicate logic).
compute_recall_at_k = compute_image_level_recall


def evaluate_checkpoint(checkpoint_path: str) -> dict:
    """Evaluate a checkpoint on the validation set.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load configs
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    
    # Load dataset
    print("Loading Flickr30k dataset...")
    cache_dir = data_config["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)
    
    # Create splits
    print("Creating train/val/test splits...")
    train_pairs, val_pairs, test_pairs = create_splits(
        config=data_config,
        image_paths=[Path(p) for p in image_paths],
        captions=captions,
    )
    
    # Create dataloaders
    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    train_transform = get_train_transforms(data_config)
    eval_transform = get_eval_transforms(data_config)
    
    # Use smaller batch for evaluation to avoid memory issues
    eval_config = dict(data_config)
    eval_config["dataset"] = dict(data_config["dataset"])
    eval_config["dataset"]["batch_size"] = 64
    eval_config["dataset"]["num_workers"] = 2
    
    train_loader, val_loader, test_loader = create_dataloaders(
        config=eval_config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)
    
    # Load checkpoint (inference-only, no optimizer/scaler needed)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("step", 0)
    print(f"Loaded checkpoint: epoch={start_epoch}, step={global_step}")
    
    # Evaluate
    print("\nEvaluating on validation set...")
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(input_ids, attention_mask)
            all_image_embeds.append(img_emb)
            all_text_embeds.append(txt_emb)
    
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Compute metrics
    captions_per_image = 5
    N_total = image_embeds.shape[0]
    N_unique = N_total // captions_per_image
    image_embeds_unique = image_embeds.view(N_unique, captions_per_image, -1).mean(dim=1)
    
    r1 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=1)
    r5 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=5)
    r10 = compute_recall_at_k(image_embeds_unique, text_embeds, captions_per_image, k=10)
    
    # Embedding diagnostics
    img_dim_var = image_embeds_unique.var(dim=0).mean().item()
    txt_dim_var = text_embeds.var(dim=0).mean().item()
    
    # Pairwise distances (sample for efficiency)
    max_n = 500
    img_cpu = image_embeds_unique[:max_n].cpu()
    txt_cpu = text_embeds[:max_n].cpu()
    
    img_dist = torch.cdist(img_cpu, img_cpu, p=2)
    txt_dist = torch.cdist(txt_cpu, txt_cpu, p=2)
    
    mask_img = ~torch.eye(img_dist.shape[0], dtype=torch.bool)
    mask_txt = ~torch.eye(txt_dist.shape[0], dtype=torch.bool)
    
    results = {
        "checkpoint": checkpoint_path,
        "epoch": start_epoch,
        "step": global_step,
        "image_to_text": {
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
        },
        "embedding_diagnostics": {
            "image_dim_variance": img_dim_var,
            "text_dim_variance": txt_dim_var,
            "image_mean_pairwise_dist": img_dist[mask_img].mean().item(),
            "text_mean_pairwise_dist": txt_dist[mask_txt].mean().item(),
        },
    }
    
    return results


def print_results(results: dict) -> None:
    """Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary of evaluation results.
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nCheckpoint: {results['checkpoint']}")
    print(f"Epoch: {results['epoch']}")
    print(f"Step: {results['step']}")
    
    print("\n--- Image-to-Text Retrieval ---")
    r1 = results["image_to_text"]["recall@1"]
    r5 = results["image_to_text"]["recall@5"]
    r10 = results["image_to_text"]["recall@10"]
    print(f"  Recall@1:  {r1:.4f} ({r1*100:.2f}%)")
    print(f"  Recall@5:  {r5:.4f} ({r5*100:.2f}%)")
    print(f"  Recall@10: {r10:.4f} ({r10*100:.2f}%)")
    
    print("\n--- Embedding Diagnostics ---")
    diag = results["embedding_diagnostics"]
    print(f"  Image dim variance:     {diag['image_dim_variance']:.6f}")
    print(f"  Text dim variance:      {diag['text_dim_variance']:.6f}")
    print(f"  Image mean pairwise dist: {diag['image_mean_pairwise_dist']:.4f}")
    print(f"  Text mean pairwise dist:  {diag['text_mean_pairwise_dist']:.4f}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    random_baseline_r1 = 1.0 / (r10 * 100 / r1) if r1 > 0 else 0.01
    print("  Random baseline Recall@1: ~1% (for 100 candidate captions)")
    print(f"  Actual Recall@1: {r1*100:.2f}% ({r1/random_baseline_r1:.1f}x random)")
    
    if diag["image_dim_variance"] > 0.001 and diag["text_dim_variance"] > 0.001:
        print("  [OK] Embedding variance healthy (no collapse)")
    else:
        print("  [WARNING] Embedding variance very low (possible collapse)")
    
    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/train/best_model.pt",
        help="Path to checkpoint file",
    )
    args = parser.parse_args()
    
    results = evaluate_checkpoint(args.checkpoint)
    print_results(results)


if __name__ == "__main__":
    main()
