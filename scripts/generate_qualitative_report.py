r"""Regenerate the qualitative retrieval report from a checkpoint.

Purpose: ROADMAP.md Phase 5 requires more than Recall@K — it asks for
10+ retrieval successes and 10+ failures to be inspected, with the
patterns written down. The existing qualitative report describes the
retired Phase 4 checkpoint, so this produces the same evidence for
whichever checkpoint it is pointed at.

What it writes is the *evidence*, not the interpretation: the retrieved
captions, their scores, and where the correct caption actually ranked.
The patterns a reader is asked to accept are written by hand in
ROADMAP.md Phase 5 after reading this file, because "the model confuses
actions" is a judgement and this script cannot make it.

Usage:
    python scripts/generate_qualitative_report.py \\
        --checkpoint checkpoints/train/best_model.pt --split test

Writes:
    reports/qualitative_examples.md
    reports/qualitative_examples.json

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

from _data_helpers import build_eval_loaders, build_eval_pairs

from vectormind.evaluation.evaluator import collapse_image_embeddings, encode_split
from vectormind.evaluation.retrieval import (
    compute_failure_analysis,
    compute_retrieval_examples,
)
from vectormind.models.vectormind_model import VectorMindModel
from vectormind.utils.config import load_config
from vectormind.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_EXAMPLES: int = 12
DEFAULT_K: int = 10
#: Retrieved captions shown per example. The full top-10 is written to
#: the JSON; the Markdown shows enough to judge the result without
#: becoming unreadable.
SHOWN_PER_EXAMPLE: int = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Write the qualitative retrieval report for a checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/train/best_model.pt",
        help="Checkpoint to analyse.",
    )
    parser.add_argument(
        "--data-config", type=str, default="configs/data.yaml", help="Dataset config."
    )
    parser.add_argument(
        "--model-config", type=str, default="configs/model.yaml", help="Model config."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Split to analyse. Held-out splits only.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=DEFAULT_EXAMPLES,
        help="Successes and failures to collect, each.",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Retrieval depth to report."
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports", help="Where to write the report."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the evaluation batch size.",
    )
    return parser.parse_args()


def _format_example(
    example: dict[str, Any], captions_per_image: int, shown: int
) -> list[str]:
    """Render one retrieval example as Markdown lines.

    Args:
        example: One entry from ``compute_retrieval_examples``.
        captions_per_image: Captions belonging to each image.
        shown: Retrieved captions to list.

    Returns:
        Markdown lines, ready to join with newlines.
    """
    start, end = example["correct_caption_range"]
    correct_ranks = [
        example["top_k_indices"].index(idx) + 1 for idx in example["correct_in_top_k"]
    ]
    rank_note = (
        f"correct caption at rank {min(correct_ranks)}"
        if correct_ranks
        else f"no correct caption in the top {len(example['top_k_indices'])}"
    )

    lines = [
        f"**Image {example['image_index']}** — `{Path(example['image_path']).name}` "
        f"({rank_note})",
        "",
        f"- *Ground truth:* {example['query_caption']}",
    ]
    for rank, (caption, score) in enumerate(
        zip(
            example["top_k_captions"][:shown],
            example["top_k_scores"][:shown],
            strict=False,
        ),
        start=1,
    ):
        index = example["top_k_indices"][rank - 1]
        marker = "✅" if start <= index < end else "❌"
        lines.append(f"- {marker} *Rank {rank}* ({score:.3f}): {caption}")
    lines.append("")
    return lines


def write_markdown(
    path: Path,
    checkpoint: Path,
    split: str,
    epoch: int,
    examples: dict[str, Any],
    failure_stats: dict[str, Any],
    captions_per_image: int,
) -> None:
    """Write the human-readable qualitative report.

    Args:
        path: Destination file.
        checkpoint: Checkpoint the retrievals came from.
        split: Split analysed.
        epoch: Human-numbered checkpoint epoch.
        examples: Output of ``compute_retrieval_examples``.
        failure_stats: Output of ``compute_failure_analysis``.
        captions_per_image: Captions belonging to each image.
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    successes = examples["successes"]
    failures = examples["failures"]

    lines = [
        "# Qualitative retrieval examples — VectorMind",
        "",
        "Generated by `scripts/generate_qualitative_report.py`; every",
        "line below comes from one run against one checkpoint.",
        "",
        f"**Checkpoint:** `{checkpoint.as_posix()}` (epoch {epoch})  ",
        f"**Split:** {split}  ",
        f"**Generated:** {generated}",
        "",
        "Image→text retrieval: each image queries the whole caption pool",
        f"of the split, and its {captions_per_image} own captions are the",
        "only correct answers.",
        "",
        "---",
        "",
        "## What the numbers say",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| Images evaluated | {failure_stats.get('total_images', '—')} |",
        f"| Failures at K={examples['k']} | {failure_stats.get('total_failures', '—')} |",
        f"| Failure rate | {failure_stats.get('failure_rate', 0.0) * 100:.2f}% |",
        "",
    ]

    distribution = failure_stats.get("hit_rank_distribution")
    if distribution:
        lines += [
            "Where the first correct caption landed, among the successes:",
            "",
            "| Rank | Images |",
            "|---|---|",
        ]
        lines += [
            f"| {rank + 1} | {count} |" for rank, count in enumerate(distribution)
        ]
        lines.append("")

    for title, group, note in (
        (
            "Successes",
            successes,
            "The correct caption appeared in the top "
            f"{examples['k']}. Ranks 1-3 are shown.",
        ),
        (
            "Failures",
            failures,
            f"No correct caption in the top {examples['k']}. What the model "
            "returned instead is the interesting part.",
        ),
    ):
        lines += [f"## {title} ({len(group)})", "", note, ""]
        for example in group:
            lines += _format_example(example, captions_per_image, SHOWN_PER_EXAMPLE)

    lines += [
        "---",
        "",
        "The interpretation of these examples — which failure modes recur,",
        "and what they imply about the encoders — is written up in",
        "[ROADMAP.md](../ROADMAP.md) Phase 5. It is deliberately not",
        "generated here: this file is the evidence, that is the argument.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    logger.info("Wrote %s", path)


def main() -> None:
    """Encode the split, collect examples, and write both report files."""
    args = parse_args()
    setup_logging(level=logging.INFO)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    captions_per_image = data_config["dataset"].get("captions_per_image", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorMindModel(model_config)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    raw_epoch = int(state.get("epoch", -1))
    epoch = raw_epoch + 1 if raw_epoch >= 0 else -1

    loaders = build_eval_loaders(data_config, args.batch_size)
    pairs = build_eval_pairs(data_config)[args.split]
    captions = [caption for _, caption in pairs]
    image_paths = [str(path) for path, _ in pairs[::captions_per_image]]

    logger.info("Encoding %s split (%d pairs)...", args.split, len(pairs))
    with torch.no_grad():
        image_pairs, text_embeds = encode_split(model, loaders[args.split], device)
    image_embeds = collapse_image_embeddings(image_pairs, captions_per_image)

    if len(image_paths) != image_embeds.shape[0]:
        raise SystemExit(
            f"Split metadata and embeddings disagree: {len(image_paths)} paths "
            f"against {image_embeds.shape[0]} image embeddings. The data config "
            "used here must match the one the checkpoint was trained with."
        )

    # Spread the sample across the split. Without a stride every example
    # comes from its first rows, which is not a sample a reader should
    # be asked to generalise from.
    stride = max(1, image_embeds.shape[0] // (2 * max(args.examples, 1)))
    examples = compute_retrieval_examples(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        image_paths=image_paths,
        captions=captions,
        captions_per_image=captions_per_image,
        k=args.k,
        num_successes=args.examples,
        num_failures=args.examples,
        stride=stride,
    )
    failure_stats = compute_failure_analysis(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        captions_per_image=captions_per_image,
        k=args.k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "qualitative_examples.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.as_posix(),
                "epoch": epoch,
                "split": args.split,
                "k": args.k,
                "stride": stride,
                "examples": examples,
                "failure_analysis": failure_stats,
                "generated_by": "scripts/generate_qualitative_report.py",
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )

    write_markdown(
        path=output_dir / "qualitative_examples.md",
        checkpoint=checkpoint,
        split=args.split,
        epoch=epoch,
        examples=examples,
        failure_stats=failure_stats,
        captions_per_image=captions_per_image,
    )

    logger.info(
        "Collected %d successes and %d failures (stride %d)",
        len(examples["successes"]),
        len(examples["failures"]),
        stride,
    )


if __name__ == "__main__":
    main()
