"""Memorization evaluation for Phase 3.5 overfit sanity check.

Purpose: evaluate whether the trained model has successfully memorized
the overfit subset, using both image→text and text→image retrieval
with image-level Recall@K (any of 5 captions counts as a correct match).

Design decisions:
- Image-level Recall@K: for each image, check if any of its 5 captions
  appears in the top-K results. This is the correct metric because all
  5 captions describe the same image — retrieving any of them is a
  success.
- Both retrieval directions: image→text and text→image.
- Similarity matrix analysis: diagonal (matched) vs off-diagonal
  (unmatched) similarity distributions.
- Top-k ranking inspection for qualitative analysis.

Input:
  - Trained VectorMindModel
  - DataLoader containing the overfit subset
  - Device

Output:
  - Recall@1/5/10 for both directions
  - Similarity matrix statistics
  - Top-k ranking examples
  - Embedding space diagnostics
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from vectormind.models.vectormind_model import VectorMindModel

logger = logging.getLogger(__name__)


def compute_image_level_recall(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
    k: int = 1,
) -> float:
    """Compute image-level Recall@K.

    For each image, check whether ANY of its 5 captions appears in the
    top-K results when ranking all captions by cosine similarity.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
            N_images = total_pairs / captions_per_image.
        text_embeds: L2-normalized text embeddings [N_pairs, D].
            N_pairs = N_images * captions_per_image.
        captions_per_image: Number of captions per image (default 5).
        k: Number of top results to consider.

    Returns:
        Image-level Recall@K as a float between 0 and 1.
    """
    N_images = image_embeds.shape[0]
    N_texts = text_embeds.shape[0]

    # Similarity matrix: [N_images, N_texts]
    similarity = image_embeds @ text_embeds.T

    # For each image, the matching caption indices are:
    # [i * captions_per_image, i * captions_per_image + 1, ..., i * captions_per_image + 4]
    correct_indices = []
    for i in range(N_images):
        start = i * captions_per_image
        end = start + captions_per_image
        correct_indices.append(set(range(start, end)))

    # Top-K retrieval
    _, top_k_indices = similarity.topk(k, dim=1)

    # Check if any correct index is in the top-K
    correct_count = 0
    for i in range(N_images):
        top_k_set = set(top_k_indices[i].tolist())
        if top_k_set & correct_indices[i]:
            correct_count += 1

    return correct_count / N_images


def compute_text_level_recall(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
    k: int = 1,
) -> float:
    """Compute text-level Recall@K (text→image direction).

    For each caption, check whether its matching image appears in the
    top-K results when ranking all images by cosine similarity.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image (default 5).
        k: Number of top results to consider.

    Returns:
        Text-level Recall@K as a float between 0 and 1.
    """
    N_images = image_embeds.shape[0]
    N_texts = text_embeds.shape[0]

    # Similarity matrix: [N_texts, N_images]
    similarity = text_embeds @ image_embeds.T

    # For each caption at index j, the matching image is j // captions_per_image
    labels = torch.arange(N_texts, device=similarity.device) // captions_per_image

    # Top-K retrieval
    _, top_k_indices = similarity.topk(k, dim=1)

    # Check if the correct image is in the top-K
    correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def compute_similarity_analysis(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
) -> dict[str, Any]:
    """Analyze the similarity matrix structure.

    Computes statistics about matched (diagonal-block) vs unmatched
    (off-diagonal) similarities to understand retrieval quality.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image.

    Returns:
        Dictionary with similarity analysis results.
    """
    N_images = image_embeds.shape[0]

    # Full similarity matrix: [N_images, N_pairs]
    similarity = image_embeds @ text_embeds.T

    # Matched similarities: for each image, the similarity to its own captions
    matched_sims = []
    for i in range(N_images):
        start = i * captions_per_image
        end = start + captions_per_image
        matched_sims.append(similarity[i, start:end].mean().item())

    # Unmatched similarities: for each image, the similarity to all other captions
    unmatched_sims = []
    for i in range(N_images):
        start = i * captions_per_image
        end = start + captions_per_image
        # Mask out the matched block
        mask = torch.ones(N_images * captions_per_image, dtype=torch.bool)
        mask[start:end] = False
        unmatched_sims.append(similarity[i, mask].mean().item())

    matched_tensor = torch.tensor(matched_sims)
    unmatched_tensor = torch.tensor(unmatched_sims)

    # Separation: how well separated are matched vs unmatched
    separation = matched_tensor.mean().item() - unmatched_tensor.mean().item()

    return {
        "matched_mean_similarity": matched_tensor.mean().item(),
        "matched_std_similarity": matched_tensor.std().item(),
        "unmatched_mean_similarity": unmatched_tensor.mean().item(),
        "unmatched_std_similarity": unmatched_tensor.std().item(),
        "separation": separation,
        "min_matched_similarity": matched_tensor.min().item(),
        "max_unmatched_similarity": unmatched_tensor.max().item(),
    }


def compute_top_k_examples(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
    k: int = 5,
    num_examples: int = 5,
) -> list[dict[str, Any]]:
    """Get top-K ranking examples for qualitative inspection.

    Args:
        image_embeds: L2-normalized image embeddings [N_images, D].
        text_embeds: L2-normalized text embeddings [N_pairs, D].
        captions_per_image: Number of captions per image.
        k: Number of top results to show.
        num_examples: Number of image examples to return.

    Returns:
        List of dictionaries, each containing an example's rankings.
    """
    N_images = image_embeds.shape[0]
    similarity = image_embeds @ text_embeds.T

    examples = []
    for i in range(min(num_examples, N_images)):
        start = i * captions_per_image
        end = start + captions_per_image

        # Top-K for this image
        scores, indices = similarity[i].topk(k)
        correct_in_top_k = [
            idx.item() for idx in indices if start <= idx.item() < end
        ]

        examples.append(
            {
                "image_index": i,
                "correct_caption_range": (start, end),
                "top_k_indices": indices.tolist(),
                "top_k_scores": scores.tolist(),
                "correct_in_top_k": correct_in_top_k,
                "recall_at_k": len(correct_in_top_k) > 0,
            }
        )

    return examples


def compute_embedding_diagnostics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> dict[str, Any]:
    """Compute comprehensive embedding space diagnostics.

    Args:
        image_embeds: L2-normalized image embeddings [N, D].
        text_embeds: L2-normalized text embeddings [N, D].

    Returns:
        Dictionary with embedding diagnostics.
    """
    # Per-dimension variance
    img_dim_var = image_embeds.var(dim=0).mean().item()
    txt_dim_var = text_embeds.var(dim=0).mean().item()

    # Pairwise distances
    img_pairwise = torch.cdist(image_embeds, image_embeds, p=2)
    txt_pairwise = torch.cdist(text_embeds, text_embeds, p=2)

    mask_img = ~torch.eye(img_pairwise.shape[0], dtype=torch.bool, device=img_pairwise.device)
    mask_txt = ~torch.eye(txt_pairwise.shape[0], dtype=torch.bool, device=txt_pairwise.device)

    return {
        "image_dim_variance": img_dim_var,
        "text_dim_variance": txt_dim_var,
        "image_mean_pairwise_dist": img_pairwise[mask_img].mean().item(),
        "text_mean_pairwise_dist": txt_pairwise[mask_txt].mean().item(),
        "image_min_pairwise_dist": img_pairwise[mask_img].min().item(),
        "text_min_pairwise_dist": txt_pairwise[mask_txt].min().item(),
    }
