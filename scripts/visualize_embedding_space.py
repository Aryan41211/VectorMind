r"""Plot the learned embedding space in 2D, from a checkpoint.

Purpose: ROADMAP.md Phase 6.5 asks for a 2D projection of the shared
embedding space for the write-up. This produces it, plus the one number
the picture always prompts — the distance between the image and text
centroids — so the figure is read against a measurement rather than an
impression.

The figure has two panels:

* **Left** — image and text embeddings projected together, coloured by
  modality. Two clouds is the expected result, not a bug: contrastive
  training aligns matched pairs *relative to other pairs*, which never
  requires the two towers to share a region of the sphere.
* **Right** — the same projection with matched image/caption pairs
  joined by a line. Short, non-crossing lines mean pairs land near each
  other; a tangle means they do not.

Usage:
    python scripts/visualize_embedding_space.py \\
        --checkpoint checkpoints/train/best_model.pt \\
        --split test --samples 800 --method tsne

Writes:
    reports/figures/embedding_space_<method>.png
    reports/embedding_projection.json   modality-gap statistics

This is an entry-point script, NOT imported by src/vectormind/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

# Chosen before pyplot is imported: the script writes files and is run
# from CI and from a terminal, where no display backend exists.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data_helpers import build_eval_loaders  # noqa: E402

from vectormind.evaluation.evaluator import (  # noqa: E402
    collapse_image_embeddings,
    encode_split,
)
from vectormind.evaluation.projection import (  # noqa: E402
    DEFAULT_SEED,
    ProjectionMethod,
    measure_modality_gap,
    project_2d,
)
from vectormind.models.vectormind_model import VectorMindModel  # noqa: E402
from vectormind.utils.config import load_config  # noqa: E402
from vectormind.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES: int = 800
DEFAULT_PAIR_LINES: int = 40
FIGURE_SIZE: tuple[float, float] = (14.0, 6.5)
FIGURE_DPI: int = 150
POINT_SIZE: float = 10.0
POINT_ALPHA: float = 0.55
LINE_ALPHA: float = 0.5
LINE_WIDTH: float = 0.8

IMAGE_COLOUR: str = "#2b6cb0"
TEXT_COLOUR: str = "#c05621"
LINE_COLOUR: str = "#4a5568"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Project a checkpoint's embedding space to 2D."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/train/best_model.pt",
        help="Checkpoint to project.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Dataset config.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Model config; must match the checkpoint's architecture.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Split to project. Held-out splits only.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Images to project. Cost is superlinear in this number.",
    )
    parser.add_argument(
        "--pair-lines",
        type=int,
        default=DEFAULT_PAIR_LINES,
        help="Matched pairs to join with a line in the right-hand panel.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["tsne", "umap", "pca"],
        help="Projection method. umap requires umap-learn.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the stochastic projections.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Root for the figure and the statistics file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the evaluation batch size.",
    )
    return parser.parse_args()


def encode_sample(
    checkpoint: Path,
    model_config_path: str,
    data_config_path: str,
    split: str,
    samples: int,
    batch_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the first ``samples`` images of a split, and one caption each.

    Args:
        checkpoint: Checkpoint to load weights from.
        model_config_path: Path to the model config.
        data_config_path: Path to the data config.
        split: ``"val"`` or ``"test"``.
        samples: Number of unique images to keep.
        batch_size: Optional evaluation batch-size override.

    Returns:
        ``(image_embeddings, text_embeddings)`` as ``[S, D]`` float
        arrays, row-aligned so row *i* of each is a matched pair.

    Raises:
        SystemExit: If the checkpoint does not exist.

    Limitations:
        Takes a prefix of the split rather than a random subset. The
        splitter already shuffles with a fixed seed, so a prefix is an
        arbitrary sample, and taking it keeps the figure reproducible
        without a second RNG.
    """
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)
    captions_per_image = data_config["dataset"].get("captions_per_image", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Encoding %s split from %s on %s", split, checkpoint, device)

    model = VectorMindModel(model_config)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    loaders = build_eval_loaders(data_config, batch_size)
    with torch.no_grad():
        image_pairs, text_pairs = encode_split(model, loaders[split], device)

    images = collapse_image_embeddings(image_pairs, captions_per_image)
    # One caption per image, so the right-hand panel joins each image to
    # a single point rather than to five stacked ones.
    texts = text_pairs[::captions_per_image]

    keep = min(samples, images.shape[0], texts.shape[0])
    logger.info("Projecting %d of %d images", keep, images.shape[0])
    return (
        images[:keep].float().cpu().numpy(),
        texts[:keep].float().cpu().numpy(),
    )


def draw(
    image_2d: np.ndarray,
    text_2d: np.ndarray,
    pair_lines: int,
    method: str,
    checkpoint: Path,
    split: str,
    gap_note: str,
    output: Path,
) -> None:
    """Render and save the two-panel figure.

    Args:
        image_2d: ``[S, 2]`` projected image coordinates.
        text_2d: ``[S, 2]`` projected caption coordinates, row-aligned
            with ``image_2d``.
        pair_lines: Matched pairs to join in the right-hand panel.
        method: Projection name, for the titles.
        checkpoint: Checkpoint the embeddings came from, for the caption.
        split: Split name, for the caption.
        gap_note: One-line summary of the measured modality gap.
        output: Destination PNG path.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    for axis in (left, right):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_alpha(0.2)

    left.scatter(
        image_2d[:, 0],
        image_2d[:, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        c=IMAGE_COLOUR,
        label="image embeddings",
        linewidths=0,
    )
    left.scatter(
        text_2d[:, 0],
        text_2d[:, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        c=TEXT_COLOUR,
        label="caption embeddings",
        linewidths=0,
    )
    left.set_title(f"Shared space, both modalities ({method.upper()})")
    left.legend(loc="best", frameon=False, fontsize=9)

    shown = min(pair_lines, image_2d.shape[0])
    right.scatter(
        image_2d[:, 0],
        image_2d[:, 1],
        s=POINT_SIZE,
        alpha=0.2,
        c=IMAGE_COLOUR,
        linewidths=0,
    )
    right.scatter(
        text_2d[:, 0],
        text_2d[:, 1],
        s=POINT_SIZE,
        alpha=0.2,
        c=TEXT_COLOUR,
        linewidths=0,
    )
    for index in range(shown):
        right.plot(
            [image_2d[index, 0], text_2d[index, 0]],
            [image_2d[index, 1], text_2d[index, 1]],
            c=LINE_COLOUR,
            alpha=LINE_ALPHA,
            linewidth=LINE_WIDTH,
        )
    right.set_title(f"{shown} matched image-caption pairs, joined")

    figure.suptitle(
        f"VectorMind embedding space — {checkpoint.as_posix()} ({split} split)",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.02,
        gap_note,
        ha="center",
        fontsize=9,
        color="#4a5568",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)
    logger.info("Wrote %s", output)


def main() -> None:
    """Encode a sample, project it, and write the figure and statistics."""
    args = parse_args()
    setup_logging(level=logging.INFO)

    checkpoint = Path(args.checkpoint)
    images, texts = encode_sample(
        checkpoint=checkpoint,
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        split=args.split,
        samples=args.samples,
        batch_size=args.batch_size,
    )

    gap = measure_modality_gap(images, texts)
    logger.info(
        "Modality gap: centroid distance %.4f, centroid cosine %.4f",
        gap.centroid_distance,
        gap.centroid_cosine,
    )

    # Both modalities are projected in one call. Projecting them
    # separately and overlaying the results would place them in two
    # unrelated coordinate systems, and the distance between the clouds
    # would mean nothing at all.
    method: ProjectionMethod = args.method
    joint = project_2d(
        np.vstack([images, texts]), method=method, seed=args.seed
    )
    image_2d, text_2d = joint[: images.shape[0]], joint[images.shape[0] :]

    gap_note = (
        f"Measured in the full {images.shape[1]}-d space: centroid distance "
        f"{gap.centroid_distance:.3f}, centroid cosine {gap.centroid_cosine:.3f}. "
        "The 2D layout is illustrative; the numbers are the evidence."
    )

    output_dir = Path(args.output_dir)
    draw(
        image_2d=image_2d,
        text_2d=text_2d,
        pair_lines=args.pair_lines,
        method=args.method,
        checkpoint=checkpoint,
        split=args.split,
        gap_note=gap_note,
        output=output_dir / "figures" / f"embedding_space_{args.method}.png",
    )

    stats_path = output_dir / "embedding_projection.json"
    stats_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.as_posix(),
                "split": args.split,
                "method": args.method,
                "seed": args.seed,
                "samples": int(images.shape[0]),
                "embedding_dim": int(images.shape[1]),
                "modality_gap": gap.to_dict(),
                "generated_by": "scripts/visualize_embedding_space.py",
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", stats_path)


if __name__ == "__main__":
    main()
