"""Project embeddings to 2D, and measure the gap between modalities.

Purpose: the numbers in ``embedding_health`` say *whether* the shared
space is well formed; they cannot show *how*. This module supplies the
two things a reader needs to see it — a 2D projection suitable for
plotting, and the one statistic a joint image/text projection always
raises: how far apart the two modalities sit.

Nothing here is used by training or serving. It exists for the reports
and the write-up, and lives in ``src/`` rather than inside the plotting
script so that it can be unit tested (CLAUDE.md §4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ProjectionMethod = Literal["tsne", "umap", "pca"]

#: t-SNE perplexity. 30 is sklearn's own default and suits the 1-2k
#: samples this project projects. sklearn requires it to stay below the
#: sample count, which :func:`project_2d` enforces.
DEFAULT_PERPLEXITY: float = 30.0

#: Seed for every stochastic projection. Fixed so that a regenerated
#: figure is the same figure — an unseeded t-SNE lays the same data out
#: differently on every run, which makes two reports impossible to
#: compare.
DEFAULT_SEED: int = 42

#: Output dimensionality. 2 everywhere; named so the slicing below is
#: not a bare literal.
PROJECTION_DIMS: int = 2

#: Lowest perplexity worth running, used when the sample count forces
#: the requested value down.
MIN_PERPLEXITY: float = 2.0

#: t-SNE degenerates as perplexity approaches the sample count, so it is
#: capped at this fraction of N rather than at N itself.
PERPLEXITY_SAMPLE_FRACTION: float = 3.0


@dataclass(frozen=True)
class ModalityGap:
    """How far the image and text clouds sit from each other.

    Attributes:
        centroid_distance: Euclidean distance between the mean image
            embedding and the mean text embedding. 0 means the two
            clouds share a centre.
        centroid_cosine: Cosine similarity of the two centroids. Near
            1.0 means both modalities lean in the same direction, which
            is the shared-component form of anisotropy this project
            tracks as ``||mean embedding||``.
        image_centroid_norm: Length of the mean image embedding.
        text_centroid_norm: Length of the mean text embedding.
        num_samples: Rows available from the smaller modality.
    """

    centroid_distance: float
    centroid_cosine: float
    image_centroid_norm: float
    text_centroid_norm: float
    num_samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable view, for the report files."""
        return {
            "centroid_distance": self.centroid_distance,
            "centroid_cosine": self.centroid_cosine,
            "image_centroid_norm": self.image_centroid_norm,
            "text_centroid_norm": self.text_centroid_norm,
            "num_samples": self.num_samples,
        }


def _validate(embeddings: np.ndarray, minimum_rows: int) -> None:
    """Raise if an embedding matrix cannot be projected or compared.

    Args:
        embeddings: Candidate matrix.
        minimum_rows: Fewest rows the caller can work with.

    Raises:
        ValueError: If the array is not 2D, has too few rows, or holds
            non-finite values. Non-finite values are rejected here
            because a single NaN propagates silently through t-SNE into
            a blank figure rather than an error.
    """
    if embeddings.ndim != PROJECTION_DIMS:
        raise ValueError(
            f"Embeddings must be 2D [N, D], got ndim={embeddings.ndim}."
        )
    if embeddings.shape[0] < minimum_rows:
        raise ValueError(
            f"Need at least {minimum_rows} embeddings, got "
            f"{embeddings.shape[0]}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf values.")


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project with PCA, via an SVD rather than a sklearn dependency.

    Args:
        embeddings: ``[N, D]`` matrix.

    Returns:
        ``[N, 2]`` projection onto the top two principal components.

    Limitations:
        Centres the data, so the shared directional component this
        project tracks as ``||mean embedding||`` is removed by
        construction and cannot be seen in a PCA panel.
    """
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return np.asarray(centred @ components[:PROJECTION_DIMS].T)


def project_2d(
    embeddings: np.ndarray,
    method: ProjectionMethod = "tsne",
    seed: int = DEFAULT_SEED,
    perplexity: float = DEFAULT_PERPLEXITY,
) -> np.ndarray:
    """Reduce embeddings to two dimensions for plotting.

    Args:
        embeddings: ``[N, D]`` float array. L2-normalization is not
            required, but is what this project always passes.
        method: ``"tsne"`` (neighbourhood structure), ``"umap"`` (the
            same, faster at scale, extra dependency), or ``"pca"``
            (linear, dependency-free, preserves global geometry).
        seed: Random seed for the stochastic methods.
        perplexity: t-SNE perplexity, lowered automatically when the
            sample count is too small for it. sklearn raises rather
            than adjusting, which would turn a small run into a crash
            instead of a coarser plot.

    Returns:
        ``[N, 2]`` float array of 2D coordinates.

    Raises:
        ValueError: If the input is not a finite 2D array with at least
            two rows, or if ``method`` is unknown.
        ImportError: If ``method`` needs a package that is not
            installed. The message carries the install command.

    Assumptions:
        The caller has already subsampled. Nothing here caps N, and
        t-SNE is O(N log N) at best.

    Limitations:
        A 2D projection of a 256-dimensional space discards nearly all
        of it. Distances in the output are indicative, never evidence —
        the measured figures in ``embedding_health`` are the evidence.
    """
    _validate(embeddings, minimum_rows=PROJECTION_DIMS)
    matrix = np.asarray(embeddings, dtype=np.float64)

    if method == "pca":
        return _pca_2d(matrix)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "t-SNE needs scikit-learn: pip install scikit-learn"
            ) from exc

        safe_perplexity = float(
            min(
                perplexity,
                max(
                    MIN_PERPLEXITY,
                    matrix.shape[0] / PERPLEXITY_SAMPLE_FRACTION,
                ),
            )
        )
        if safe_perplexity != perplexity:
            logger.info(
                "Perplexity lowered %.1f -> %.1f for %d samples",
                perplexity,
                safe_perplexity,
                matrix.shape[0],
            )
        return np.asarray(
            TSNE(
                n_components=PROJECTION_DIMS,
                perplexity=safe_perplexity,
                init="pca",
                random_state=seed,
            ).fit_transform(matrix)
        )

    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "UMAP needs umap-learn: pip install umap-learn"
            ) from exc
        return np.asarray(
            umap.UMAP(
                n_components=PROJECTION_DIMS, random_state=seed
            ).fit_transform(matrix)
        )

    raise ValueError(
        f"Unknown projection method {method!r}; expected tsne, umap or pca."
    )


def measure_modality_gap(
    image_embeds: np.ndarray, text_embeds: np.ndarray
) -> ModalityGap:
    """Measure how far the image and text clouds sit from each other.

    Why it exists: a joint projection of both towers reliably shows two
    separated clusters, and a reader's first question is whether that is
    an artefact of the plot or a property of the space. It is a property
    — contrastive training pulls *matched pairs* together relative to
    other pairs, which never requires the two modalities to occupy the
    same region — and this returns the number that says so.

    Args:
        image_embeds: ``[N, D]`` image embeddings.
        text_embeds: ``[M, D]`` text embeddings. N and M need not match.

    Returns:
        A :class:`ModalityGap`.

    Raises:
        ValueError: If either array is not a finite 2D array with at
            least one row, or if the two disagree on dimensionality.
    """
    _validate(image_embeds, minimum_rows=1)
    _validate(text_embeds, minimum_rows=1)
    if image_embeds.shape[1] != text_embeds.shape[1]:
        raise ValueError(
            f"Dimension mismatch: images are {image_embeds.shape[1]}-d, "
            f"text is {text_embeds.shape[1]}-d."
        )

    image_centroid = image_embeds.mean(axis=0)
    text_centroid = text_embeds.mean(axis=0)
    image_norm = float(np.linalg.norm(image_centroid))
    text_norm = float(np.linalg.norm(text_centroid))

    denominator = image_norm * text_norm
    cosine = (
        float(image_centroid @ text_centroid / denominator)
        if denominator > 0.0
        else 0.0
    )

    return ModalityGap(
        centroid_distance=float(np.linalg.norm(image_centroid - text_centroid)),
        centroid_cosine=cosine,
        image_centroid_norm=image_norm,
        text_centroid_norm=text_norm,
        num_samples=min(image_embeds.shape[0], text_embeds.shape[0]),
    )
