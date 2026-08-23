"""Phase 3.5 memorization evaluation: analyze retrieval quality.

Purpose: load the trained overfit checkpoint and run comprehensive
retrieval analysis — image->text and text->image Recall@K, similarity
matrix analysis, top-k ranking examples, and embedding diagnostics.

This is the evaluation companion to overfit_sanity_check.py. While
the training script logs Recall@1 per epoch, this script provides
the detailed analysis needed for the Phase 3.5 engineering review.

Usage:
    python scripts/evaluate_memorization.py

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch

# Ensure src/ and scripts/ are on the path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vectormind.data.dataloader import create_dataloaders
from vectormind.data.overfit_subset import load_subset_metadata
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms
from vectormind.evaluation.memorization import (
    compute_embedding_diagnostics,
    compute_image_level_recall,
    compute_similarity_analysis,
    compute_text_level_recall,
    compute_top_k_examples,
)
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.checkpoint import load_checkpoint
from vectormind.training.memory_queue import MemoryQueue
from vectormind.utils.config import load_config, require_keys
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

CAPTIONS_PER_IMAGE: int = 5
CHECKPOINT_DIR = Path("checkpoints/overfit")
REPORT_DIR = Path("reports/overfit")


def main() -> None:
    """Run the memorization evaluation."""
    setup_logging(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("Phase 3.5 Memorization Evaluation")
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

    # ---- Step 2: Load overfit subset ----
    logger.info("Step 2: Loading overfit subset...")
    subset_pairs = load_subset_metadata(subset_cfg["metadata_path"])
    logger.info(
        "  Loaded %d pairs from %s", len(subset_pairs), subset_cfg["metadata_path"]
    )

    # ---- Step 3: Build DataLoader ----
    logger.info("Step 3: Building DataLoader...")
    overfit_data_config = dict(data_config)
    overfit_data_config["dataset"] = dict(data_config["dataset"])
    overfit_data_config["dataset"]["batch_size"] = subset_cfg["batch_size"]

    tokenizer = CaptionTokenizer(
        tokenizer_name=data_config["dataset"]["tokenizer_name"],
        max_length=data_config["dataset"]["max_text_length"],
    )
    eval_transform = get_eval_transforms(data_config)

    train_loader, _, _ = create_dataloaders(
        config=overfit_data_config,
        train_pairs=subset_pairs,
        val_pairs=subset_pairs[:10],
        test_pairs=subset_pairs[:10],
        train_transform=eval_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    # ---- Step 4: Load trained model ----
    logger.info("Step 4: Loading trained model from checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)

    # Find the latest checkpoint
    checkpoints = sorted(CHECKPOINT_DIR.glob("overfit_epoch*.pt"))
    if not checkpoints:
        logger.error("No checkpoints found in %s", CHECKPOINT_DIR)
        sys.exit(1)

    ckpt_path = checkpoints[-1]
    logger.info("  Loading checkpoint: %s", ckpt_path)

    from vectormind.training.train_loop import create_optimizer, create_scaler

    optimizer = create_optimizer(model, lr=3e-4, weight_decay=0.01)
    scaler = create_scaler()
    memory_queue = MemoryQueue(
        queue_size=1, embed_dim=model_config["embedding"]["shared_dim"], device=device
    )

    epoch, step = load_checkpoint(ckpt_path, model, optimizer, scaler, memory_queue)
    logger.info("  Restored: epoch=%d, step=%d", epoch, step)
    model.eval()

    # ---- Step 5: Compute embeddings ----
    # Build a non-shuffled DataLoader over ALL subset pairs (no drop_last)
    # to ensure we get all 500 pairs in deterministic order.
    from torch.utils.data import DataLoader

    from vectormind.data.dataset import Flickr30kDataset

    eval_dataset = Flickr30kDataset(
        image_paths=[p for p, _ in subset_pairs],
        captions=[c for _, c in subset_pairs],
        transform=eval_transform,
        tokenizer=tokenizer,
        max_text_length=data_config["dataset"]["max_text_length"],
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=subset_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda batch: {
            "image": torch.stack([item["image"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        },
    )

    logger.info("Step 5: Computing embeddings on full training set...")
    all_image_embeds: list[torch.Tensor] = []
    all_text_embeds: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(input_ids, attention_mask)
            all_image_embeds.append(img_emb)
            all_text_embeds.append(txt_emb)

    all_image_embeds_cat = torch.cat(all_image_embeds, dim=0)
    all_text_embeds_cat = torch.cat(all_text_embeds, dim=0)

    # Deduplicate image embeddings: each image appears 5 times (once per caption).
    # Average the 5 embeddings per image to get one embedding per unique image.
    N_total = all_image_embeds_cat.shape[0]
    N_unique = N_total // CAPTIONS_PER_IMAGE
    image_embeds = all_image_embeds_cat.view(N_unique, CAPTIONS_PER_IMAGE, -1).mean(
        dim=1
    )
    text_embeds = all_text_embeds_cat  # [N_total, D] — all 500 captions

    N_images = image_embeds.shape[0]
    N_texts = text_embeds.shape[0]
    logger.info(
        "  Computed: %d unique image embeddings, %d text embeddings", N_images, N_texts
    )

    # ---- Step 6: Image -> Text Recall ----
    logger.info("Step 6: Image -> Text Retrieval Analysis")
    logger.info("-" * 50)

    i2t_r1 = compute_image_level_recall(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE, k=1
    )
    i2t_r5 = compute_image_level_recall(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE, k=5
    )
    i2t_r10 = compute_image_level_recall(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE, k=10
    )

    logger.info("  Image -> Text Recall@1:   %.4f (%.1f%%)", i2t_r1, i2t_r1 * 100)
    logger.info("  Image -> Text Recall@5:   %.4f (%.1f%%)", i2t_r5, i2t_r5 * 100)
    logger.info("  Image -> Text Recall@10:  %.4f (%.1f%%)", i2t_r10, i2t_r10 * 100)

    # ---- Step 7: Text -> Image Recall ----
    logger.info("Step 7: Text -> Image Retrieval Analysis")
    logger.info("-" * 50)

    t2i_r1 = compute_text_level_recall(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE, k=1
    )
    t2i_r5 = compute_text_level_recall(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE, k=5
    )
    t2i_r10 = compute_text_level_recall(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE, k=10
    )

    logger.info("  Text -> Image Recall@1:   %.4f (%.1f%%)", t2i_r1, t2i_r1 * 100)
    logger.info("  Text -> Image Recall@5:   %.4f (%.1f%%)", t2i_r5, t2i_r5 * 100)
    logger.info("  Text -> Image Recall@10:  %.4f (%.1f%%)", t2i_r10, t2i_r10 * 100)

    # ---- Step 8: Similarity Matrix Analysis ----
    logger.info("Step 8: Similarity Matrix Analysis")
    logger.info("-" * 50)

    sim_analysis = compute_similarity_analysis(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE
    )

    logger.info(
        "  Matched similarity (mean ± std):   %.4f ± %.4f",
        sim_analysis["matched_mean_similarity"],
        sim_analysis["matched_std_similarity"],
    )
    logger.info(
        "  Unmatched similarity (mean ± std): %.4f ± %.4f",
        sim_analysis["unmatched_mean_similarity"],
        sim_analysis["unmatched_std_similarity"],
    )
    logger.info("  Separation (matched - unmatched):  %.4f", sim_analysis["separation"])
    logger.info(
        "  Min matched similarity:            %.4f",
        sim_analysis["min_matched_similarity"],
    )
    logger.info(
        "  Max unmatched similarity:          %.4f",
        sim_analysis["max_unmatched_similarity"],
    )

    # ---- Step 9: Top-K Examples ----
    logger.info("Step 9: Top-K Ranking Examples")
    logger.info("-" * 50)

    examples = compute_top_k_examples(
        image_embeds, text_embeds, CAPTIONS_PER_IMAGE, k=10, num_examples=5
    )

    for i, ex in enumerate(examples):
        status = "HIT" if ex["recall_at_k"] else "MISS"
        logger.info(
            "  Example %d (image %d): %s | top scores: [%.4f, %.4f, %.4f, %.4f, %.4f]",
            i + 1,
            ex["image_index"],
            status,
            *ex["top_k_scores"][:5],
        )

    # ---- Step 10: Embedding Diagnostics ----
    logger.info("Step 10: Embedding Diagnostics")
    logger.info("-" * 50)

    embed_diag = compute_embedding_diagnostics(image_embeds, text_embeds)

    logger.info("  Image dim variance:       %.6f", embed_diag["image_dim_variance"])
    logger.info("  Text dim variance:        %.6f", embed_diag["text_dim_variance"])
    logger.info(
        "  Image mean pairwise dist: %.4f", embed_diag["image_mean_pairwise_dist"]
    )
    logger.info(
        "  Text mean pairwise dist:  %.4f", embed_diag["text_mean_pairwise_dist"]
    )
    logger.info(
        "  Image min pairwise dist:  %.4f", embed_diag["image_min_pairwise_dist"]
    )
    logger.info(
        "  Text min pairwise dist:   %.4f", embed_diag["text_min_pairwise_dist"]
    )

    # ---- Step 11: Verdict ----
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Phase 3.5 Memorization Evaluation — VERDICT")
    logger.info("=" * 60)

    # Random chance baselines
    random_i2t_r1 = CAPTIONS_PER_IMAGE / N_texts  # 5/500 = 0.01
    random_t2i_r1 = 1.0 / N_images  # 1/100 = 0.01

    logger.info("")
    logger.info("Random chance baselines:")
    logger.info("  Image -> Text Recall@1: %.4f (1/%d)", random_i2t_r1, N_texts)
    logger.info("  Text -> Image Recall@1: %.4f (1/%d)", random_t2i_r1, N_images)
    logger.info("")
    logger.info("Achieved results:")
    logger.info(
        "  Image -> Text Recall@1: %.4f (%.0fx above random)",
        i2t_r1,
        i2t_r1 / random_i2t_r1,
    )
    logger.info("  Image -> Text Recall@5: %.4f", i2t_r5)
    logger.info("  Image -> Text Recall@10: %.4f", i2t_r10)
    logger.info(
        "  Text -> Image Recall@1: %.4f (%.0fx above random)",
        t2i_r1,
        t2i_r1 / random_t2i_r1,
    )
    logger.info("  Text -> Image Recall@5: %.4f", t2i_r5)
    logger.info("  Text -> Image Recall@10: %.4f", t2i_r10)
    logger.info("  Similarity separation: %.4f", sim_analysis["separation"])
    logger.info("  Image embed variance:  %.6f", embed_diag["image_dim_variance"])
    logger.info("  Text embed variance:   %.6f", embed_diag["text_dim_variance"])
    logger.info("  Temperature:           %.4f", model.temperature.item())
    logger.info("  Total time:            %.1f seconds", elapsed)

    # Phase 3.5 acceptance criteria
    passed = True
    failure_reasons: list[str] = []

    # Criterion 1: Recall@1 well above random
    if i2t_r1 < 0.5:
        # Relaxed criterion: at least 50% for image-level recall
        # (each image has 5 correct captions, so this is generous)
        passed = False
        failure_reasons.append(f"Image->Text Recall@1={i2t_r1:.4f} < 0.5")

    # Criterion 2: Embedding variance healthy
    if embed_diag["image_dim_variance"] < 1e-6:
        passed = False
        failure_reasons.append(
            f"Image embedding variance={embed_diag['image_dim_variance']:.8f} < 1e-6 (collapse)"
        )

    if embed_diag["text_dim_variance"] < 1e-6:
        passed = False
        failure_reasons.append(
            f"Text embedding variance={embed_diag['text_dim_variance']:.8f} < 1e-6 (collapse)"
        )

    # Criterion 3: Positive similarity separation
    if sim_analysis["separation"] <= 0:
        passed = False
        failure_reasons.append(
            f"Similarity separation={sim_analysis['separation']:.4f} <= 0 (no discrimination)"
        )

    logger.info("")
    if passed:
        logger.info("VERDICT: PASSED")
        logger.info("Phase 3.5 acceptance criteria satisfied.")
        logger.info("Safe to proceed to Phase 4 (full training run).")
    else:
        logger.error("VERDICT: FAILED")
        for reason in failure_reasons:
            logger.error("  - %s", reason)
        logger.error("Do NOT proceed to Phase 4 until these issues are resolved.")

    logger.info("=" * 60)

    # ---- Save report ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "checkpoint": str(ckpt_path),
            "subset_size": N_images,
            "total_pairs": N_texts,
            "captions_per_image": CAPTIONS_PER_IMAGE,
            "epoch": epoch,
            "step": step,
            "temperature": model.temperature.item(),
        },
        "image_to_text": {
            "recall@1": i2t_r1,
            "recall@5": i2t_r5,
            "recall@10": i2t_r10,
            "random_baseline_r1": random_i2t_r1,
            "improvement_over_random": i2t_r1 / random_i2t_r1,
        },
        "text_to_image": {
            "recall@1": t2i_r1,
            "recall@5": t2i_r5,
            "recall@10": t2i_r10,
            "random_baseline_r1": random_t2i_r1,
            "improvement_over_random": t2i_r1 / random_t2i_r1,
        },
        "similarity_analysis": sim_analysis,
        "embedding_diagnostics": embed_diag,
        "top_k_examples": examples,
        "verdict": "PASSED" if passed else "FAILED",
        "failure_reasons": failure_reasons,
        "elapsed_seconds": elapsed,
    }

    report_path = REPORT_DIR / "phase3_5_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
