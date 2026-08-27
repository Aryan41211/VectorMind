"""Shared split-evaluation loop for every training and analysis script.

Purpose: one implementation of "run a dataloader through the model and
report retrieval quality plus embedding health", used by
``scripts/train.py`` via ``vectormind.training.trainer``.

The four Phase 4 scripts (``train.py``, ``resume_training.py``,
``benchmark_epoch.py``, ``hyperparameter_experiment.py``) each carried
their own ``compute_recall_at_k`` and ``evaluate`` — roughly 2,500 lines
with a duplicated core, in violation of CLAUDE.md §3's "no duplicate
logic". Worse than the duplication was the drift risk: a metric fix
applied to one copy silently left the other three reporting different
numbers for the same checkpoint.

Input:
  - A VectorMindModel and a DataLoader over an evaluation split
Output:
  - SplitMetrics: Recall@1/5/10, embedding diagnostics, health verdict

Dependencies: evaluation.memorization and evaluation.embedding_health.
No dependency on training/ or data/ (FOLDER_STRUCTURE.md).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from vectormind.evaluation.embedding_health import (
    EmbeddingHealth,
    compute_embedding_health,
)
from vectormind.evaluation.memorization import (
    compute_embedding_diagnostics,
    compute_image_level_recall,
    compute_text_level_recall,
)
from vectormind.models.vectormind_model import VectorMindModel

logger = logging.getLogger(__name__)

# Recall cutoffs reported everywhere in this project (ROADMAP.md Phase 5).
RECALL_K_VALUES: tuple[int, ...] = (1, 5, 10)


@dataclass
class SplitMetrics:
    """Retrieval and embedding-health metrics for one evaluation split.

    Attributes:
        recall: Image->text Recall@K, keyed by K.
        text_to_image_recall: Text->image Recall@K, keyed by K.
        diagnostics: Per-dimension variance and pairwise distances,
            retained for continuity with earlier reports.
        health: Collapse diagnostics. Read ``health.separation`` before
            trusting any recall number — Phase 4 showed the two can move
            in opposite directions.
        num_images: Unique images evaluated.
        num_captions: Captions evaluated.
    """

    recall: dict[int, float]
    text_to_image_recall: dict[int, float]
    diagnostics: dict[str, float]
    health: EmbeddingHealth
    num_images: int
    num_captions: int

    def to_flat_dict(self) -> dict[str, float]:
        """Flatten to the scalar mapping the training scripts log.

        Returns:
            Keys such as ``recall@1``, ``t2i_recall@10``, ``separation``
            and ``collapsed`` (as 0.0/1.0), suitable for TensorBoard and
            for JSON reports.
        """
        flat: dict[str, float] = {}
        for k, value in self.recall.items():
            flat[f"recall@{k}"] = value
        for k, value in self.text_to_image_recall.items():
            flat[f"t2i_recall@{k}"] = value
        flat.update(self.diagnostics)
        flat.update(
            {
                "separation": self.health.separation,
                "matched_similarity": self.health.matched_similarity,
                "unmatched_similarity": self.health.unmatched_similarity,
                "image_mean_cosine": self.health.image_mean_cosine,
                "text_mean_cosine": self.health.text_mean_cosine,
                "image_mean_norm": self.health.image_mean_norm,
                "collapsed": float(self.health.collapsed),
            }
        )
        return flat

    def to_report_dict(self) -> dict[str, Any]:
        """Return the nested structure written into ``reports/``.

        Returns:
            JSON-serializable mapping preserving the recall/diagnostics/
            health grouping.
        """
        return {
            "image_to_text_recall": {f"@{k}": v for k, v in self.recall.items()},
            "text_to_image_recall": {
                f"@{k}": v for k, v in self.text_to_image_recall.items()
            },
            "diagnostics": dict(self.diagnostics),
            "embedding_health": asdict(self.health),
            "num_images": self.num_images,
            "num_captions": self.num_captions,
        }


@torch.no_grad()
def encode_split(
    model: VectorMindModel,
    dataloader: DataLoader[Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every batch in a split into image and text embeddings.

    Args:
        model: Model to evaluate. Switched to eval mode; the caller is
            responsible for restoring train mode afterwards.
        dataloader: Yields batches with ``image``, ``input_ids`` and
            ``attention_mask``.
        device: Device to run inference on.

    Returns:
        Tuple of ``(image_embeddings, text_embeddings)``, each
        ``[N_pairs, D]`` and L2-normalized. One row per (image, caption)
        pair, so image rows repeat once per caption.

    Raises:
        RuntimeError: If a batch is missing a required key, or its
            tensors cannot be moved to ``device``.
    """
    model.eval()
    image_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        image_chunks.append(model.encode_image(images))
        text_chunks.append(model.encode_text(input_ids, attention_mask))

    return torch.cat(image_chunks, dim=0), torch.cat(text_chunks, dim=0)


def collapse_image_embeddings(
    image_embeds: torch.Tensor,
    captions_per_image: int,
) -> torch.Tensor:
    """Reduce per-pair image embeddings to one row per unique image.

    The dataloader emits one row per (image, caption) pair, so each
    image appears ``captions_per_image`` times. Recall@K is defined per
    image, so those rows must collapse first.

    Args:
        image_embeds: Per-pair image embeddings ``[N_pairs, D]``.
        captions_per_image: Rows belonging to each image.

    Returns:
        Embeddings of shape ``[N_pairs // captions_per_image, D]``.

    Raises:
        ValueError: If ``captions_per_image`` is not positive, or does
            not divide the number of rows — an indivisible count means
            the split was built with a different grouping than assumed,
            and averaging would silently mix images together.

    Assumptions:
        Rows for one image are contiguous, which the splitter
        guarantees.
    """
    if captions_per_image <= 0:
        raise ValueError(
            f"captions_per_image must be positive, got {captions_per_image}."
        )

    n_pairs = image_embeds.shape[0]
    if n_pairs % captions_per_image != 0:
        raise ValueError(
            f"{n_pairs} pairs is not divisible by captions_per_image="
            f"{captions_per_image}; the split grouping is not what this "
            f"function assumes."
        )

    n_images = n_pairs // captions_per_image
    # The rows are identical (same image, same deterministic transform),
    # so the mean is that same vector — averaging simply states the
    # intent without depending on which duplicate is picked.
    return image_embeds.view(n_images, captions_per_image, -1).mean(dim=1)


def evaluate_split(
    model: VectorMindModel,
    dataloader: DataLoader[Any],
    device: torch.device,
    captions_per_image: int = 5,
    k_values: tuple[int, ...] = RECALL_K_VALUES,
) -> SplitMetrics:
    """Evaluate a model on one split, end to end.

    Reports retrieval quality in both directions plus embedding health.
    Health is not optional decoration: Phase 4 produced a checkpoint
    whose Recall@10 looked acceptable while its embedding space had
    collapsed into a narrow cone, and the reports of the time called
    that healthy (docs/KNOWN_ISSUES.md §1).

    Args:
        model: Model to evaluate.
        dataloader: DataLoader over the split.
        device: Device to run inference on.
        captions_per_image: Captions per image in this dataset.
        k_values: Recall cutoffs to report.

    Returns:
        A :class:`SplitMetrics` summary.

    Raises:
        ValueError: If the split size is not divisible by
            ``captions_per_image``.
    """
    image_embeds, text_embeds = encode_split(model, dataloader, device)
    image_unique = collapse_image_embeddings(image_embeds, captions_per_image)
    n_images = image_unique.shape[0]

    recall = {
        k: compute_image_level_recall(
            image_unique, text_embeds, captions_per_image, k=k
        )
        for k in k_values
    }
    # Text->image ranks over images, so K cannot exceed the image count.
    text_to_image_recall = {
        k: compute_text_level_recall(
            image_unique, text_embeds, captions_per_image, k=min(k, n_images)
        )
        for k in k_values
    }

    diagnostics = compute_embedding_diagnostics(image_unique, text_embeds)

    # One caption per image keeps the pairing one-to-one, which is what
    # compute_embedding_health() requires to separate matched from
    # unmatched. Passing all five would count four true positives as
    # unmatched and bias separation downward.
    first_captions = text_embeds[::captions_per_image][:n_images]
    health = compute_embedding_health(image_unique, first_captions)

    logger.info(
        "Split evaluated: %d images, %d captions | R@1=%.4f R@5=%.4f "
        "R@10=%.4f | %s",
        n_images,
        text_embeds.shape[0],
        recall.get(1, float("nan")),
        recall.get(5, float("nan")),
        recall.get(10, float("nan")),
        health.verdict,
    )

    return SplitMetrics(
        recall=recall,
        text_to_image_recall=text_to_image_recall,
        diagnostics=diagnostics,
        health=health,
        num_images=n_images,
        num_captions=text_embeds.shape[0],
    )
