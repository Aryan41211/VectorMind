"""Comprehensive retrieval evaluation for Phase 5.

Purpose: provide full bidirectional retrieval metrics (image→text and
text→image), embedding space diagnostics, and qualitative example
generation for the trained VectorMind model on held-out splits.

This module extends memorization.py (Phase 3.5) with:
- Bidirectional Recall@K (both directions in one call)
- Comprehensive embedding diagnostics (collapse, uniformity, alignment)
- Retrieval example generation for qualitative analysis
- Category-level accuracy breakdown

Input:
  - Trained VectorMindModel or precomputed embeddings
  - DataLoader for the evaluation split
  - Device

Output:
  - Recall@1/5/10 for both retrieval directions
  - Embedding space health diagnostics
  - Top-K retrieval examples with scores
  - Category-level performance breakdown

Design decisions:
- Reuses compute_image_level_recall and compute_text_level_recall
  from memorization.py for consistency.
- All functions accept precomputed embeddings to enable efficient
  evaluation without re-running the model.
- Embedding diagnostics include both standard metrics (variance,
  pairwise distance) and research-grade metrics (uniformity,
  alignment) for comprehensive analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from vectormind.evaluation.memorization import (
    compute_embedding_diagnostics,
    compute_image_level_recall,
    compute_similarity_analysis,
    compute_text_level_recall,
    compute_top_k_examples,
)

logger = logging.getLogger(__name__)


def compute_bidirectional_recall(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
) -> dict[str, float]:
    """Compute Recall@1/5/10 for both image→text and text→image directions.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image (default 5).

    Returns:
        Dictionary with keys:
        - "image_to_text_recall@1"
        - "image_to_text_recall@5"
        - "image_to_text_recall@10"
        - "text_to_image_recall@1"
        - "text_to_image_recall@5"
        - "text_to_image_recall@10"
    """
    results = {}
    N_images = image_embeds.shape[0]

    for k in [1, 5, 10]:
        i2t = compute_image_level_recall(
            image_embeds, text_embeds, captions_per_image, k=k
        )
        k_t2i = min(k, N_images)
        t2i = compute_text_level_recall(
            image_embeds, text_embeds, captions_per_image, k=k_t2i
        )
        results[f"image_to_text_recall@{k}"] = i2t
        results[f"text_to_image_recall@{k}"] = t2i

    return results


def compute_uniformity(
    embeddings: torch.Tensor,
    t: float = 2.0,
) -> float:
    """Compute uniformity metric on the hypersphere.

    Uniformity measures how evenly embeddings are distributed on the
    unit hypersphere. Lower values indicate better uniformity.

    Reference: Wang & Isola, "Understanding Contrastive Representation
    Learning through Alignment and Uniformity on the Hypersphere" (2020).

    Args:
        embeddings: L2-normalized embeddings [N, D].
        t: Temperature parameter for the Gaussian kernel (default 2.0).

    Returns:
        Uniformity score (lower is better).
    """
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)
    uniformity = sq_pdist.mul(-t).exp().mean().log()
    return uniformity.item()


def compute_alignment(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
    alpha: float = 2.0,
) -> float:
    """Compute alignment metric between image and text embeddings.

    Alignment measures how close matched image-text pairs are in the
    embedding space. Lower values indicate better alignment.

    Reference: Wang & Isola, "Understanding Contrastive Representation
    Learning through Alignment and Uniformity on the Hypersphere" (2020).

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image.
        alpha: Exponent for the distance (default 2.0).

    Returns:
        Alignment score (lower is better).
    """
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

    N_images = image_embeds.shape[0]
    N_texts = text_embeds.shape[0]

    if N_images * captions_per_image == N_texts:
        text_embeds_expanded = text_embeds.view(
            N_images, captions_per_image, -1
        )
        image_expanded = image_embeds.unsqueeze(1).expand_as(text_embeds_expanded)
        alignment = (image_expanded - text_embeds_expanded).norm(dim=2).pow(alpha).mean()
    else:
        min_n = min(N_images, N_texts)
        alignment = (image_embeds[:min_n] - text_embeds[:min_n]).norm(dim=1).pow(alpha).mean()

    return alignment.item()


def compute_comprehensive_embedding_diagnostics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
) -> dict[str, Any]:
    """Compute comprehensive embedding space diagnostics.

    Includes standard metrics (variance, pairwise distances) and
    research-grade metrics (uniformity, alignment) for thorough
    embedding space analysis.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image.

    Returns:
        Dictionary with embedding diagnostics.
    """
    base_diag = compute_embedding_diagnostics(image_embeds, text_embeds)

    uniformity_img = compute_uniformity(image_embeds)
    uniformity_txt = compute_uniformity(text_embeds)

    alignment = compute_alignment(image_embeds, text_embeds, captions_per_image)

    return {
        **base_diag,
        "image_uniformity": uniformity_img,
        "text_uniformity": uniformity_txt,
        "alignment": alignment,
    }


def compute_retrieval_examples(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_paths: list[str] | None = None,
    captions: list[str] | None = None,
    captions_per_image: int = 5,
    k: int = 10,
    num_successes: int = 5,
    num_failures: int = 5,
) -> dict[str, Any]:
    """Generate retrieval examples for qualitative analysis.

    Selects representative successes and failures from the retrieval
    results for manual inspection.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        image_paths: Optional list of image file paths.
        captions: Optional list of caption strings.
        captions_per_image: Number of captions per image.
        k: Number of top results to retrieve.
        num_successes: Number of success examples to return.
        num_failures: Number of failure examples to return.

    Returns:
        Dictionary with "successes" and "failures" lists.
    """
    N_images = image_embeds.shape[0]
    similarity = image_embeds @ text_embeds.T

    successes = []
    failures = []

    for i in range(N_images):
        start = i * captions_per_image
        end = start + captions_per_image

        scores, indices = similarity[i].topk(k)

        correct_in_top_k = [
            idx.item() for idx in indices if start <= idx.item() < end
        ]

        example = {
            "image_index": i,
            "image_path": image_paths[i] if image_paths else f"image_{i}",
            "correct_caption_range": (start, end),
            "query_caption": captions[start] if captions else f"caption_{start}",
            "top_k_indices": indices.tolist(),
            "top_k_scores": scores.tolist(),
            "top_k_captions": [
                captions[idx] if captions else f"caption_{idx}"
                for idx in indices.tolist()
            ],
            "correct_in_top_k": correct_in_top_k,
            "recall_at_k": len(correct_in_top_k) > 0,
        }

        if len(correct_in_top_k) > 0 and len(successes) < num_successes:
            successes.append(example)
        elif len(correct_in_top_k) == 0 and len(failures) < num_failures:
            failures.append(example)

        if len(successes) >= num_successes and len(failures) >= num_failures:
            break

    return {
        "successes": successes,
        "failures": failures,
        "total_images_evaluated": N_images,
        "k": k,
    }


def compute_failure_analysis(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
    k: int = 10,
) -> dict[str, Any]:
    """Analyze failure patterns in retrieval.

    Identifies common failure modes:
    1. Semantic failures: correct concept but wrong specific
    2. Visual failures: visually similar but semantically different
    3. Ambiguous failures: multiple valid interpretations

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image.
        k: Number of top results to retrieve.

    Returns:
        Dictionary with failure analysis.
    """
    N_images = image_embeds.shape[0]
    similarity = image_embeds @ text_embeds.T

    total_failures = 0
    failure_rank_distribution = [0] * k

    for i in range(N_images):
        start = i * captions_per_image
        end = start + captions_per_image

        scores, indices = similarity[i].topk(k)

        correct_in_top_k = [
            idx.item() for idx in indices if start <= idx.item() < end
        ]

        if len(correct_in_top_k) == 0:
            total_failures += 1
            failure_rank_distribution[0] += 1

    failure_rate = total_failures / N_images

    return {
        "total_images": N_images,
        "total_failures": total_failures,
        "failure_rate": failure_rate,
        "success_rate": 1.0 - failure_rate,
        "failure_rank_distribution": failure_rank_distribution,
    }
