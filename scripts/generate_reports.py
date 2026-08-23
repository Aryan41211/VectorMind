r"""Regenerate every Phase 5 report from a checkpoint, in one pass.

Purpose: make ``reports/`` reproducible. The previous reports were
assembled by hand across several sessions and disagreed with each other
and with the data: ``phase5_embedding_diagnostics.json`` claimed a
matched-vs-unmatched separation of 0.33 (matched 0.45, unmatched 0.12)
where direct measurement gives 0.094 (matched 0.937, unmatched 0.843),
and ``checkpoint_summary.json`` shipped with an unresolved
``temperature_discrepancy`` field recording that two documents disagreed
about the learned temperature.

Numbers a reader cannot regenerate are not results. Every figure this
script writes comes from the checkpoint it was pointed at, in a single
run, so the files cannot drift apart again.

Usage:
    python scripts/generate_reports.py \\
        --checkpoint checkpoints/train/best_model.pt \\
        --output reports/

Writes:
    reports/metrics_<split>.json      per-split retrieval + health
    reports/embedding_diagnostics.json
    reports/checkpoint_summary.json
    reports/RESULTS.md                human-readable summary

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import load_flickr30k_from_hf

from vectormind.data.dataloader import create_dataloaders
from vectormind.data.splitter import create_splits
from vectormind.data.tokenizer import CaptionTokenizer
from vectormind.data.transforms import get_eval_transforms, get_train_transforms
from vectormind.evaluation.evaluator import SplitMetrics, evaluate_split
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.utils.config import load_config
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Random-chance baselines for the Flickr30k splits, used to state every
# recall figure as a multiple of chance rather than as a bare number.
# Image->text ranks 5 captions per image among all captions; text->image
# ranks one image among all images.
SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Regenerate all Phase 5 reports from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/train/best_model.pt",
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Directory to write reports into.",
    )
    parser.add_argument(
        "--model-config", type=str, default="configs/model.yaml"
    )
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override dataset.batch_size for evaluation.",
    )
    return parser.parse_args()


def random_baseline(k: int, n_candidates: int, n_relevant: int = 1) -> float:
    """Probability that a random top-K contains a relevant item.

    Uses the complement of drawing K non-relevant items without
    replacement, which is the honest baseline for Recall@K — the common
    shortcut ``k / n`` overstates chance when ``n_relevant > 1``.

    Args:
        k: Cutoff.
        n_candidates: Total items ranked.
        n_relevant: Relevant items among them.

    Returns:
        Chance probability in ``[0, 1]``.

    Raises:
        ValueError: If any argument is non-positive, or if there are
            more relevant items than candidates.
    """
    if k <= 0 or n_candidates <= 0 or n_relevant <= 0:
        raise ValueError(
            f"k, n_candidates and n_relevant must be positive; got "
            f"{k}, {n_candidates}, {n_relevant}."
        )
    if n_relevant > n_candidates:
        raise ValueError(
            f"n_relevant ({n_relevant}) exceeds n_candidates ({n_candidates})."
        )

    k = min(k, n_candidates)
    p_none = 1.0
    for i in range(k):
        remaining = n_candidates - i
        non_relevant = remaining - n_relevant
        if non_relevant <= 0:
            return 1.0
        p_none *= non_relevant / remaining
    return 1.0 - p_none


def build_loaders(
    data_config: dict[str, Any],
    batch_size: int | None,
) -> dict[str, Any]:
    """Build val and test dataloaders from the configured dataset.

    Args:
        data_config: Parsed configs/data.yaml.
        batch_size: Optional override for evaluation batch size.

    Returns:
        Mapping of split name to DataLoader.
    """
    cfg = dict(data_config)
    cfg["dataset"] = dict(data_config["dataset"])
    if batch_size is not None:
        cfg["dataset"]["batch_size"] = batch_size

    cache_dir = cfg["dataset"]["local_cache_dir"]
    image_paths, captions = load_flickr30k_from_hf(cache_dir)
    train_pairs, val_pairs, test_pairs = create_splits(
        config=cfg,
        image_paths=[Path(p) for p in image_paths],
        captions=captions,
    )

    tokenizer = CaptionTokenizer(
        tokenizer_name=cfg["dataset"]["tokenizer_name"],
        max_length=cfg["dataset"]["max_text_length"],
    )
    _, val_loader, test_loader = create_dataloaders(
        config=cfg,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=get_train_transforms(cfg),
        eval_transform=get_eval_transforms(cfg),
        tokenizer=tokenizer,
    )
    return {"val": val_loader, "test": test_loader}


def summarize(metrics: SplitMetrics, captions_per_image: int) -> dict[str, Any]:
    """Attach random-chance baselines to a split's recall figures.

    Args:
        metrics: Measured metrics for the split.
        captions_per_image: Relevant captions per image.

    Returns:
        Report-ready mapping with each recall value alongside its
        chance baseline and the ratio between them.
    """
    report = metrics.to_report_dict()

    i2t_baselines = {}
    for k, value in metrics.recall.items():
        chance = random_baseline(k, metrics.num_captions, captions_per_image)
        i2t_baselines[f"@{k}"] = {
            "measured": value,
            "random_baseline": chance,
            "times_baseline": (value / chance) if chance > 0 else None,
        }

    t2i_baselines = {}
    for k, value in metrics.text_to_image_recall.items():
        chance = random_baseline(k, metrics.num_images, 1)
        t2i_baselines[f"@{k}"] = {
            "measured": value,
            "random_baseline": chance,
            "times_baseline": (value / chance) if chance > 0 else None,
        }

    report["image_to_text_recall"] = i2t_baselines
    report["text_to_image_recall"] = t2i_baselines
    return report


def write_results_markdown(
    path: Path,
    checkpoint: Path,
    epoch: int,
    step: int,
    logit_scale: float,
    per_split: dict[str, dict[str, Any]],
    health: dict[str, Any],
) -> None:
    """Write the human-readable results summary.

    Args:
        path: Destination file.
        checkpoint: Checkpoint the numbers came from.
        epoch: Checkpoint epoch.
        step: Checkpoint global step.
        logit_scale: Learned logit scale at that checkpoint.
        per_split: Output of :func:`summarize` keyed by split.
        health: Embedding health dict for the test split.
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Results — VectorMind",
        "",
        "Every number here is produced by `scripts/generate_reports.py`",
        "in a single run against a single checkpoint. Regenerate with:",
        "",
        "```bash",
        f"python scripts/generate_reports.py --checkpoint {checkpoint.as_posix()}",
        "```",
        "",
        f"**Checkpoint:** `{checkpoint.as_posix()}` (epoch {epoch}, step {step})  ",
        f"**Learned logit scale:** {logit_scale:.2f}  ",
        f"**Generated:** {generated}",
        "",
        "---",
        "",
        "## Retrieval",
        "",
        "Recall@K against the random-chance baseline for each direction.",
        "Chance is computed as the complement of drawing K non-relevant",
        "items without replacement, not the `k/n` shortcut, which",
        "overstates chance when an image has five valid captions.",
        "",
    ]

    for split, report in per_split.items():
        lines += [
            f"### {split.capitalize()} split "
            f"({report['num_images']} images, {report['num_captions']} captions)",
            "",
            "| Direction | K | Measured | Chance | vs chance |",
            "|---|---|---|---|---|",
        ]
        for direction, key in (
            ("image → text", "image_to_text_recall"),
            ("text → image", "text_to_image_recall"),
        ):
            for cutoff, entry in report[key].items():
                ratio = entry["times_baseline"]
                lines.append(
                    f"| {direction} | {cutoff.lstrip('@')} | "
                    f"{entry['measured'] * 100:.2f}% | "
                    f"{entry['random_baseline'] * 100:.2f}% | "
                    f"{ratio:.1f}× |" if ratio else "| — |"
                )
        lines.append("")

    lines += [
        "## Embedding health",
        "",
        "Recall@K alone cannot tell you whether a contrastive model has",
        "learned a usable space. Phase 4 shipped a checkpoint whose",
        "embeddings all sat inside a narrow cone at separation 0.094,",
        "and whose report called it HEALTHY. These are the numbers that",
        "would have caught it.",
        "",
        "| Metric | Value | Healthy |",
        "|---|---|---|",
        f"| Matched similarity | {health['matched_similarity']:.4f} | high |",
        f"| Unmatched similarity | {health['unmatched_similarity']:.4f} | ≈ 0 |",
        f"| **Separation** | **{health['separation']:.4f}** | **large** |",
        f"| Mean image–image cosine | {health['image_mean_cosine']:.4f} | ≈ 0 |",
        f"| Mean text–text cosine | {health['text_mean_cosine']:.4f} | ≈ 0 |",
        f"| ‖mean image embedding‖ | {health['image_mean_norm']:.4f} | ≈ 0 |",
        f"| Per-dimension variance | {health['image_dim_variance']:.6f} | — |",
        "",
        f"**Grade:** {health['grade']}",
        "",
        f"{health['verdict']}",
        "",
        "Reference points from this project's own runs: the Phase 3.5",
        "tiny-subset overfit reached separation 0.964; the unclamped",
        "Phase 4 checkpoint measured 0.094. See",
        "[KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) §1.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    logger.info("Wrote %s", path)


def main() -> None:
    """Evaluate a checkpoint and write every report file."""
    args = parse_args()
    setup_logging(level=logging.INFO)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    captions_per_image = data_config["dataset"].get("captions_per_image", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating %s on %s", checkpoint_path, device)

    model = VectorMindModel(model_config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    epoch = int(ckpt.get("epoch", -1))
    step = int(ckpt.get("step", -1))
    logit_scale = float(model.temperature.item())

    loaders = build_loaders(data_config, args.batch_size)

    per_split: dict[str, dict[str, Any]] = {}
    test_metrics: SplitMetrics | None = None
    for split in SPLITS:
        logger.info("Evaluating %s split...", split)
        metrics = evaluate_split(model, loaders[split], device, captions_per_image)
        per_split[split] = summarize(metrics, captions_per_image)
        if split == "test":
            test_metrics = metrics

        out = output_dir / f"metrics_{split}.json"
        out.write_text(
            json.dumps(per_split[split], indent=2), encoding="utf-8", newline="\n"
        )
        logger.info("Wrote %s", out)

    assert test_metrics is not None
    health = test_metrics.health.to_dict()

    (output_dir / "embedding_diagnostics.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_path.as_posix(),
                "epoch": epoch,
                "step": step,
                "split": "test",
                "logit_scale": logit_scale,
                "health": health,
                "diagnostics": test_metrics.diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )

    val_r10 = per_split["val"]["image_to_text_recall"]["@10"]["measured"]
    test_r10 = per_split["test"]["image_to_text_recall"]["@10"]["measured"]
    (output_dir / "checkpoint_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_path.as_posix(),
                "epoch": epoch,
                "step": step,
                "logit_scale": logit_scale,
                "val_recall@10": val_r10,
                "test_recall@10": test_r10,
                "val_test_gap_r10": test_r10 - val_r10,
                "embedding_separation": health["separation"],
                "collapsed": health["collapsed"],
                "generated_by": "scripts/generate_reports.py",
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )

    write_results_markdown(
        output_dir / "RESULTS.md",
        checkpoint_path,
        epoch,
        step,
        logit_scale,
        per_split,
        health,
    )

    logger.info("=" * 60)
    logger.info("Val  R@10: %.2f%%", val_r10 * 100)
    logger.info("Test R@10: %.2f%%", test_r10 * 100)
    logger.info("Separation: %.4f — %s", health["separation"], health["verdict"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
