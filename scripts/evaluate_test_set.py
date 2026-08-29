"""Phase 5: Evaluate trained model on test set.

Purpose: Load the best checkpoint (Epoch 7), compute Recall@1/5/10
for both image→text and text→image retrieval on the held-out test set,
and generate comprehensive evaluation reports.

Usage:
    python scripts/evaluate_test_set.py
    python scripts/evaluate_test_set.py --checkpoint checkpoints/train/best_model.pt
    python scripts/evaluate_test_set.py --split test
    python scripts/evaluate_test_set.py --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import build_split_from_cache

from vectormind.data.dataloader import create_dataloaders
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms, get_train_transforms
from vectormind.evaluation.retrieval import (
    compute_bidirectional_recall,
    compute_comprehensive_embedding_diagnostics,
    compute_failure_analysis,
    compute_retrieval_examples,
)
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.utils.config import load_config
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VectorMind Phase 5: Test Set Evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/train/best_model.pt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers (default: 2).",
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        default=True,
        help="Save retrieval examples to JSON.",
    )
    return parser.parse_args()


@torch.no_grad()
def compute_embeddings(
    model: VectorMindModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute embeddings for all samples in the dataloader.

    Args:
        model: The trained VectorMindModel.
        dataloader: DataLoader yielding batches.
        device: Device to evaluate on.

    Returns:
        Tuple of (image_embeds, text_embeds) as L2-normalized tensors.
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

    return image_embeds, text_embeds


def main() -> None:
    """Run Phase 5 evaluation pipeline."""
    args = parse_args()
    setup_logging(level=logging.INFO)

    logger.info("=" * 70)
    logger.info("VectorMind Phase 5 -- Test Set Evaluation")
    logger.info("=" * 70)

    torch.backends.cudnn.benchmark = True

    logger.info("Step 1: Loading configurations...")
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    logger.info("Step 2: Loading Flickr30k dataset...")
    train_pairs, val_pairs, test_pairs = build_split_from_cache(data_config)
    logger.info(
        "  Train: %d, Val: %d, Test: %d pairs",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )

    eval_pairs = val_pairs if args.split == "val" else test_pairs
    logger.info("  Evaluating on %s split: %d pairs", args.split, len(eval_pairs))

    logger.info("Step 4: Building DataLoaders...")
    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    train_transform = get_train_transforms(data_config)
    eval_transform = get_eval_transforms(data_config)

    eval_config = dict(data_config)
    eval_config["dataset"] = dict(data_config["dataset"])
    eval_config["dataset"]["batch_size"] = args.batch_size
    eval_config["dataset"]["num_workers"] = args.num_workers

    train_loader, val_loader, test_loader = create_dataloaders(
        config=eval_config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    eval_loader = val_loader if args.split == "val" else test_loader

    # Diagnostic: print first/last 5 image filenames from eval split
    seen_for_log: list[str] = []
    for p, _ in eval_pairs:
        p_str = str(p)
        if p_str not in seen_for_log:
            seen_for_log.append(p_str)
    logger.info(
        "  EVAL SPLIT DIAGNOSTIC: using '%s' split (%d pairs, %d unique images)",
        args.split,
        len(eval_pairs),
        len(seen_for_log),
    )
    logger.info("  First 5 images: %s", [Path(s).name for s in seen_for_log[:5]])
    logger.info("  Last 5 images:  %s", [Path(s).name for s in seen_for_log[-5:]])

    logger.info("Step 5: Loading model from checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    logger.info("  Loaded checkpoint: epoch=%d, step=%d", epoch, step)

    logger.info("Step 6: Computing embeddings...")
    start_time = time.time()
    image_embeds, text_embeds = compute_embeddings(model, eval_loader, device)
    embed_time = time.time() - start_time
    logger.info(
        "  Computed %d image and %d text embeddings in %.1f seconds",
        image_embeds.shape[0],
        text_embeds.shape[0],
        embed_time,
    )

    captions_per_image = 5
    N_images = image_embeds.shape[0] // captions_per_image
    image_embeds_unique = image_embeds.view(
        N_images, captions_per_image, -1
    ).mean(dim=1)

    logger.info("Step 7: Computing retrieval metrics...")
    recall_results = compute_bidirectional_recall(
        image_embeds_unique, text_embeds, captions_per_image
    )

    logger.info("Step 8: Computing embedding diagnostics...")
    embed_diag = compute_comprehensive_embedding_diagnostics(
        image_embeds_unique, text_embeds, captions_per_image
    )

    logger.info("Step 9: Computing failure analysis...")
    failure_analysis = compute_failure_analysis(
        image_embeds_unique, text_embeds, captions_per_image
    )

    examples = None
    if args.save_examples:
        logger.info("Step 10: Generating retrieval examples...")
        # Build per-unique-image paths (one entry per image, not per caption)
        seen_images: list[str] = []
        for p, _ in eval_pairs:
            p_str = str(p)
            if p_str not in seen_images:
                seen_images.append(p_str)
        eval_all_captions = [c for _, c in eval_pairs]
        examples = compute_retrieval_examples(
            image_embeds_unique,
            text_embeds,
            image_paths=seen_images,
            captions=eval_all_captions,
            captions_per_image=captions_per_image,
            k=10,
            num_successes=5,
            num_failures=5,
        )

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Checkpoint: %s (epoch=%d, step=%d)", args.checkpoint, epoch, step)
    logger.info("Split: %s (%d images, %d captions)", args.split, N_images, len(text_embeds))
    logger.info("")
    logger.info("--- Image-to-Text Retrieval ---")
    logger.info("  Recall@1:  %.4f (%.1fx random baseline)", recall_results["image_to_text_recall@1"], recall_results["image_to_text_recall@1"] / 0.01)
    logger.info("  Recall@5:  %.4f (%.1fx random baseline)", recall_results["image_to_text_recall@5"], recall_results["image_to_text_recall@5"] / 0.05)
    logger.info("  Recall@10: %.4f (%.1fx random baseline)", recall_results["image_to_text_recall@10"], recall_results["image_to_text_recall@10"] / 0.10)
    logger.info("")
    logger.info("--- Text-to-Image Retrieval ---")
    logger.info("  Recall@1:  %.4f (%.1fx random baseline)", recall_results["text_to_image_recall@1"], recall_results["text_to_image_recall@1"] / 0.01)
    logger.info("  Recall@5:  %.4f (%.1fx random baseline)", recall_results["text_to_image_recall@5"], recall_results["text_to_image_recall@5"] / 0.05)
    logger.info("  Recall@10: %.4f (%.1fx random baseline)", recall_results["text_to_image_recall@10"], recall_results["text_to_image_recall@10"] / 0.10)
    logger.info("")
    logger.info("--- Embedding Diagnostics ---")
    logger.info("  Image dim variance:      %.6f", embed_diag["image_dim_variance"])
    logger.info("  Text dim variance:       %.6f", embed_diag["text_dim_variance"])
    logger.info("  Image mean pairwise dist: %.4f", embed_diag["image_mean_pairwise_dist"])
    logger.info("  Text mean pairwise dist:  %.4f", embed_diag["text_mean_pairwise_dist"])
    logger.info("  Image uniformity:        %.4f", embed_diag["image_uniformity"])
    logger.info("  Text uniformity:         %.4f", embed_diag["text_uniformity"])
    logger.info("  Alignment:               %.4f", embed_diag["alignment"])
    logger.info("")
    logger.info("--- Failure Analysis ---")
    logger.info("  Total images: %d", failure_analysis["total_images"])
    logger.info("  Total failures: %d", failure_analysis["total_failures"])
    logger.info("  Failure rate: %.4f", failure_analysis["failure_rate"])
    logger.info("  Success rate: %.4f", failure_analysis["success_rate"])
    logger.info("")
    logger.info("=" * 70)

    logger.info("Step 11: Saving results...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "step": step,
        "split": args.split,
        "split_size": {
            "images": N_images,
            "captions": len(text_embeds),
        },
        "image_to_text": {
            "recall@1": recall_results["image_to_text_recall@1"],
            "recall@5": recall_results["image_to_text_recall@5"],
            "recall@10": recall_results["image_to_text_recall@10"],
        },
        "text_to_image": {
            "recall@1": recall_results["text_to_image_recall@1"],
            "recall@5": recall_results["text_to_image_recall@5"],
            "recall@10": recall_results["text_to_image_recall@10"],
        },
        "embedding_diagnostics": embed_diag,
        "failure_analysis": failure_analysis,
        "random_baseline": {
            "recall@1": 0.01,
            "recall@5": 0.05,
            "recall@10": 0.10,
        },
    }

    if examples:
        results["retrieval_examples"] = examples

    output_path = REPORTS_DIR / f"phase5_{args.split}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("  Results saved to: %s", output_path)

    logger.info("=" * 70)
    logger.info("Phase 5 Evaluation Complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
