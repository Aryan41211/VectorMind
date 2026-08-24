"""Uniformity regularizer for the shared embedding space.

Purpose: pull embeddings apart on the hypersphere, targeting the one
health threshold the converged model still fails —
``‖mean embedding‖ 0.621`` against a 0.5 ceiling
(docs/KNOWN_ISSUES.md §12).

The metric this optimizes is Wang & Isola's uniformity, "Understanding
Contrastive Representation Learning through Alignment and Uniformity on
the Hypersphere" (2020):

    L_uniform = log E[ exp(-t · ‖x_i - x_j‖²) ]   over distinct pairs

Lower is better. A perfectly uniform distribution on the sphere
minimises it; a collapsed one maximises it, because every squared
distance is near zero and every exponential near one.

Why this and not a direct penalty on ‖mean embedding‖. The mean norm is
a *symptom* — it can be driven to zero by an antipodal split into two
tight clusters, which is not a better space. Uniformity penalises the
whole pairwise distance distribution, so it cannot be satisfied that
way, and the mean norm falls as a consequence rather than as the target.

Why it is separate from the InfoNCE term. InfoNCE already contains an
implicit uniformity pressure through its denominator, but the strength
of that pressure is tied to the logit scale, which is clamped here for
unrelated reasons (§5.1). An explicit weighted term makes the tradeoff
adjustable and, more importantly, measurable: `weight=0.0` reproduces
the current model exactly, so an A/B is a one-config-value change.

Input:
  - L2-normalized embeddings, [N, D]
Output:
  - Scalar loss contribution, differentiable

Dependencies: torch only.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Wang & Isola's default. Larger t weights close pairs more heavily.
DEFAULT_UNIFORMITY_T: float = 2.0

# Cap on rows entering the O(N^2) pairwise computation. At batch 128 the
# full matrix is trivial, so this only matters if batch size grows.
MAX_UNIFORMITY_SAMPLES: int = 1024


def uniformity_loss(
    embeddings: torch.Tensor,
    t: float = DEFAULT_UNIFORMITY_T,
) -> torch.Tensor:
    """Compute the Wang & Isola uniformity loss.

    Args:
        embeddings: L2-normalized embeddings, shape ``[N, D]``. The
            projection heads guarantee normalization; this function does
            not renormalize, so a caller passing unnormalized vectors
            gets a meaningless number rather than a silent correction.
        t: Temperature of the Gaussian kernel. Must be positive.

    Returns:
        Scalar tensor, differentiable with respect to ``embeddings``.
        Lower means more uniformly spread.

    Raises:
        ValueError: If ``embeddings`` is not 2D, has fewer than two
            rows, or if ``t`` is not positive.

    Assumptions:
        Rows are L2-normalized. On the unit sphere squared Euclidean
        distance and cosine similarity are equivalent up to a constant,
        which is what makes this a spread measure rather than a scale
        measure.

    Limitations:
        O(N²) in the batch dimension. Rows beyond
        :data:`MAX_UNIFORMITY_SAMPLES` are dropped rather than
        subsampled randomly — inside a training step the batch is
        already a random sample of the data, so taking a prefix adds no
        bias and avoids a per-step RNG call.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D, got ndim={embeddings.ndim}."
        )
    if embeddings.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 embeddings for a pairwise distance, got "
            f"{embeddings.shape[0]}."
        )
    if t <= 0:
        raise ValueError(f"t must be positive, got {t}.")

    if embeddings.shape[0] > MAX_UNIFORMITY_SAMPLES:
        embeddings = embeddings[:MAX_UNIFORMITY_SAMPLES]

    # pdist gives the condensed vector of pairwise distances, which is
    # half the memory of the full matrix and excludes the zero diagonal
    # that would otherwise dominate the mean.
    squared_distances = torch.pdist(embeddings, p=2).pow(2)
    return squared_distances.mul(-t).exp().mean().log()


def combined_uniformity_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    t: float = DEFAULT_UNIFORMITY_T,
) -> torch.Tensor:
    """Average the uniformity loss across both modalities.

    Both towers are regularized because embedding health is measured as
    the worse of the two: a spread image space paired with a collapsed
    text space still fails, and penalising only one tower would let the
    other drift.

    Args:
        image_embeds: L2-normalized image embeddings ``[B, D]``.
        text_embeds: L2-normalized text embeddings ``[B, D]``.
        t: Temperature of the Gaussian kernel.

    Returns:
        Scalar tensor: the mean of the two per-modality losses.

    Raises:
        ValueError: Propagated from :func:`uniformity_loss` for invalid
            shapes or ``t``.
    """
    image_term = uniformity_loss(image_embeds, t=t)
    text_term = uniformity_loss(text_embeds, t=t)
    return (image_term + text_term) / 2.0
