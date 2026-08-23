"""Embedding-space health metrics that detect representational collapse.

Purpose: measure whether a contrastive embedding space is genuinely
spread out, rather than merely non-degenerate. Phase 4 of this project
shipped a checkpoint whose per-dimension variance looked acceptable
(0.00075) while every embedding actually sat inside a narrow cone —
mean off-diagonal image-image cosine 0.81, matched-vs-unmatched
separation 0.094. Variance alone did not catch that; the metrics here
do.

Why variance is the wrong instrument: embeddings are L2-normalized onto
the unit hypersphere, so per-dimension variance conflates "spread out"
with "spread across dimensions". A cone of vectors all pointing the
same way can still show nonzero per-dimension variance. The direct
questions are: how far is the mean embedding from the origin, and how
much larger is a matched similarity than an unmatched one.

Reference points measured on this project:

    Run                     separation   mean_cosine   ||mean embed||
    Phase 3.5 (healthy)         0.964        ~0.0            low
    Phase 4 (collapsed)         0.094         0.81           0.90

Input:
  - L2-normalized image and text embeddings, [N, D]

Output:
  - EmbeddingHealth dataclass, JSON-serializable via to_dict()

Dependencies: torch only. No dependency on models/ or training/, so
this can be called from evaluation scripts, the training loop, and
tests alike (FOLDER_STRUCTURE.md).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Below this matched-vs-unmatched separation the space is treated as
# collapsed. Calibrated against this project's own runs: Phase 3.5's
# healthy overfit reached 0.964, Phase 4's collapsed checkpoint 0.094.
SEPARATION_COLLAPSE_THRESHOLD: float = 0.25

# ||mean embedding|| above this means most vectors share a direction.
# 0.0 is ideal (mass spread over the sphere); 1.0 is total collapse.
MEAN_NORM_COLLAPSE_THRESHOLD: float = 0.5

# Mean off-diagonal cosine above this indicates a narrow cone.
ANISOTROPY_COLLAPSE_THRESHOLD: float = 0.5

# Cap on how many embeddings enter the O(N^2) similarity computations.
# 4096 rows is a 4096x4096 float32 matrix (~67MB), which is safe on a
# 6GB card alongside a loaded model.
_MAX_PAIRWISE_SAMPLES: int = 4096


@dataclass
class EmbeddingHealth:
    """Diagnostic summary of a shared embedding space.

    Attributes:
        matched_similarity: Mean cosine between correctly paired
            image-text embeddings. Higher is better.
        unmatched_similarity: Mean cosine between mismatched pairs.
            Should be near 0 in a healthy space.
        separation: ``matched_similarity - unmatched_similarity``. The
            headline number — this is what retrieval quality rests on.
        image_mean_cosine: Mean off-diagonal image-image cosine.
            Near 0 is healthy; near 1 means a narrow cone.
        text_mean_cosine: Mean off-diagonal text-text cosine.
        image_mean_norm: L2 norm of the mean image embedding, in
            ``[0, 1]``. 0 is ideal, 1 is total collapse.
        text_mean_norm: L2 norm of the mean text embedding.
        image_dim_variance: Mean per-dimension variance of image
            embeddings. Retained for continuity with earlier reports;
            do not read it alone (see module docstring).
        text_dim_variance: Mean per-dimension variance of text
            embeddings.
        num_samples: Number of embedding pairs the statistics used.
        collapsed: True if any collapse threshold was crossed.
        verdict: Short human-readable summary.
    """

    matched_similarity: float
    unmatched_similarity: float
    separation: float
    image_mean_cosine: float
    text_mean_cosine: float
    image_mean_norm: float
    text_mean_norm: float
    image_dim_variance: float
    text_dim_variance: float
    num_samples: int
    collapsed: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of every field.

        Returns:
            Mapping of field name to value, suitable for
            ``json.dump`` into ``reports/``.
        """
        return asdict(self)


def _mean_offdiagonal_cosine(embeddings: torch.Tensor) -> float:
    """Mean cosine similarity between distinct rows.

    Args:
        embeddings: L2-normalized embeddings, shape ``[N, D]``.

    Returns:
        Mean of the off-diagonal entries of the ``[N, N]`` similarity
        matrix. Returns 0.0 when fewer than two rows are given, since
        no distinct pair exists.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0

    sim = embeddings @ embeddings.T
    # Subtract the diagonal (all exactly 1.0 for normalized rows)
    # rather than materializing an [N, N] boolean mask.
    total = sim.sum() - sim.diagonal().sum()
    return (total / (n * (n - 1))).item()


def _subsample(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take a deterministic aligned subsample of both modalities.

    Pairing must be preserved, so both tensors are indexed with the
    same indices. A fixed generator seed keeps the metric reproducible
    across runs — an unseeded sample would make separation wobble
    between evaluations of the same checkpoint.

    Args:
        image_embeds: Image embeddings ``[N, D]``.
        text_embeds: Text embeddings ``[N, D]``.
        max_samples: Maximum rows to keep.

    Returns:
        Tuple of subsampled ``(image_embeds, text_embeds)``.
    """
    n = image_embeds.shape[0]
    if n <= max_samples:
        return image_embeds, text_embeds

    generator = torch.Generator(device="cpu").manual_seed(42)
    idx = torch.randperm(n, generator=generator)[:max_samples]
    idx = idx.to(image_embeds.device)
    return image_embeds[idx], text_embeds[idx]


def compute_embedding_health(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    max_samples: int = _MAX_PAIRWISE_SAMPLES,
) -> EmbeddingHealth:
    """Measure whether an embedding space has collapsed.

    Computes matched-vs-unmatched similarity separation, per-modality
    anisotropy, and the norm of the mean embedding, then applies the
    module-level thresholds to produce a verdict.

    Args:
        image_embeds: L2-normalized image embeddings, shape ``[N, D]``.
        text_embeds: L2-normalized text embeddings, shape ``[N, D]``.
            Row ``i`` must be the caption paired with image ``i``.
        max_samples: Cap on rows used for the ``O(N^2)`` statistics.
            Rows beyond this are subsampled deterministically.

    Returns:
        An :class:`EmbeddingHealth` summary.

    Raises:
        ValueError: If the two tensors disagree in shape, are not 2D,
            or contain fewer than two rows.

    Assumptions:
        Both tensors are already L2-normalized and row-aligned. The
        model's projection heads guarantee normalization; alignment is
        the caller's responsibility.

    Limitations:
        Assumes exactly one positive caption per image. Flickr30k has
        five, so callers evaluating the full caption set should pass
        one caption per image (or accept that four true positives are
        counted as unmatched, which biases separation downward).
    """
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D, got image_embeds.ndim="
            f"{image_embeds.ndim}, text_embeds.ndim={text_embeds.ndim}."
        )
    if image_embeds.shape != text_embeds.shape:
        raise ValueError(
            f"Image and text embeddings must have the same shape, got "
            f"{tuple(image_embeds.shape)} and {tuple(text_embeds.shape)}."
        )
    if image_embeds.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 embeddings to measure separation, got "
            f"{image_embeds.shape[0]}."
        )

    image_embeds = image_embeds.detach().float()
    text_embeds = text_embeds.detach().float()

    # Variance is computed on the FULL set — it is O(N) and callers
    # compare it against previously published numbers.
    image_dim_variance = image_embeds.var(dim=0).mean().item()
    text_dim_variance = text_embeds.var(dim=0).mean().item()
    image_mean_norm = image_embeds.mean(dim=0).norm(p=2).item()
    text_mean_norm = text_embeds.mean(dim=0).norm(p=2).item()

    img_s, txt_s = _subsample(image_embeds, text_embeds, max_samples)
    n = img_s.shape[0]

    cross = img_s @ txt_s.T
    matched = cross.diagonal().mean().item()
    unmatched = ((cross.sum() - cross.diagonal().sum()) / (n * (n - 1))).item()
    separation = matched - unmatched

    image_mean_cosine = _mean_offdiagonal_cosine(img_s)
    text_mean_cosine = _mean_offdiagonal_cosine(txt_s)

    failures: list[str] = []
    if separation < SEPARATION_COLLAPSE_THRESHOLD:
        failures.append(
            f"separation {separation:.3f} < {SEPARATION_COLLAPSE_THRESHOLD}"
        )
    if max(image_mean_norm, text_mean_norm) > MEAN_NORM_COLLAPSE_THRESHOLD:
        failures.append(
            f"||mean embedding|| {max(image_mean_norm, text_mean_norm):.3f} "
            f"> {MEAN_NORM_COLLAPSE_THRESHOLD}"
        )
    if max(image_mean_cosine, text_mean_cosine) > ANISOTROPY_COLLAPSE_THRESHOLD:
        failures.append(
            f"mean off-diagonal cosine "
            f"{max(image_mean_cosine, text_mean_cosine):.3f} > "
            f"{ANISOTROPY_COLLAPSE_THRESHOLD}"
        )

    collapsed = bool(failures)
    verdict = (
        "COLLAPSED — " + "; ".join(failures)
        if collapsed
        else f"HEALTHY — separation {separation:.3f}, "
        f"max mean-cosine {max(image_mean_cosine, text_mean_cosine):.3f}"
    )

    health = EmbeddingHealth(
        matched_similarity=matched,
        unmatched_similarity=unmatched,
        separation=separation,
        image_mean_cosine=image_mean_cosine,
        text_mean_cosine=text_mean_cosine,
        image_mean_norm=image_mean_norm,
        text_mean_norm=text_mean_norm,
        image_dim_variance=image_dim_variance,
        text_dim_variance=text_dim_variance,
        num_samples=n,
        collapsed=collapsed,
        verdict=verdict,
    )

    log = logger.warning if collapsed else logger.info
    log("Embedding health: %s", verdict)

    return health
