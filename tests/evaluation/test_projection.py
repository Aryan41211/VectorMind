"""Tests for :mod:`vectormind.evaluation.projection`.

Covers the 2D projections used by the embedding-space figure and the
modality-gap statistic reported beside it. The projections themselves
are stochastic, so the assertions are on the properties a caller
depends on — shape, determinism under a fixed seed, and refusal of
inputs that would otherwise produce a silently empty figure.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vectormind.evaluation.projection import (
    DEFAULT_PERPLEXITY,
    PROJECTION_DIMS,
    ModalityGap,
    measure_modality_gap,
    project_2d,
)

RNG_SEED = 7
SAMPLE_COUNT = 60
EMBED_DIM = 16


def _unit_rows(count: int, dim: int, seed: int = RNG_SEED) -> np.ndarray:
    """Return ``count`` random L2-normalized rows of width ``dim``."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, dim))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class TestProject2D:
    """Shape, determinism and input validation for :func:`project_2d`."""

    @pytest.mark.parametrize("method", ["pca", "tsne"])
    def test_returns_two_columns_per_row(self, method: str) -> None:
        embeddings = _unit_rows(SAMPLE_COUNT, EMBED_DIM)
        projected = project_2d(embeddings, method=method)  # type: ignore[arg-type]
        assert projected.shape == (SAMPLE_COUNT, PROJECTION_DIMS)
        assert np.isfinite(projected).all()

    @pytest.mark.parametrize("method", ["pca", "tsne"])
    def test_is_deterministic_under_a_fixed_seed(self, method: str) -> None:
        # A regenerated report must produce the same figure; an unseeded
        # t-SNE lays the same data out differently every run.
        embeddings = _unit_rows(SAMPLE_COUNT, EMBED_DIM)
        first = project_2d(embeddings, method=method)  # type: ignore[arg-type]
        second = project_2d(embeddings, method=method)  # type: ignore[arg-type]
        np.testing.assert_allclose(first, second)

    def test_pca_preserves_a_planted_two_cluster_split(self) -> None:
        # Two well-separated clusters in D dimensions must stay
        # separated in 2D, or the figure means nothing.
        rng = np.random.default_rng(RNG_SEED)
        offset = np.zeros(EMBED_DIM)
        offset[0] = 10.0
        cluster_a = rng.normal(scale=0.1, size=(30, EMBED_DIM))
        cluster_b = rng.normal(scale=0.1, size=(30, EMBED_DIM)) + offset
        projected = project_2d(np.vstack([cluster_a, cluster_b]), method="pca")

        centre_a = projected[:30].mean(axis=0)
        centre_b = projected[30:].mean(axis=0)
        within = np.linalg.norm(projected[:30] - centre_a, axis=1).mean()
        assert np.linalg.norm(centre_a - centre_b) > 10 * within

    def test_perplexity_is_lowered_for_small_samples(self) -> None:
        # sklearn raises when perplexity >= n_samples. A small sample
        # should produce a coarser plot, not a crash.
        embeddings = _unit_rows(10, EMBED_DIM)
        projected = project_2d(
            embeddings, method="tsne", perplexity=DEFAULT_PERPLEXITY
        )
        assert projected.shape == (10, PROJECTION_DIMS)

    def test_rejects_one_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            project_2d(np.zeros(EMBED_DIM), method="pca")

    def test_rejects_a_single_row(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            project_2d(np.zeros((1, EMBED_DIM)), method="pca")

    def test_rejects_non_finite_values(self) -> None:
        embeddings = _unit_rows(SAMPLE_COUNT, EMBED_DIM)
        embeddings[3, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            project_2d(embeddings, method="pca")

    def test_rejects_an_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown projection method"):
            project_2d(
                _unit_rows(SAMPLE_COUNT, EMBED_DIM),
                method="isomap",  # type: ignore[arg-type]
            )


class TestModalityGap:
    """The statistic reported beside the joint projection."""

    def test_identical_clouds_have_no_gap(self) -> None:
        embeddings = _unit_rows(SAMPLE_COUNT, EMBED_DIM)
        gap = measure_modality_gap(embeddings, embeddings)
        assert gap.centroid_distance == pytest.approx(0.0, abs=1e-9)
        assert gap.centroid_cosine == pytest.approx(1.0, abs=1e-9)
        assert gap.num_samples == SAMPLE_COUNT

    def test_opposed_clouds_report_a_negative_cosine(self) -> None:
        embeddings = _unit_rows(SAMPLE_COUNT, EMBED_DIM)
        gap = measure_modality_gap(embeddings, -embeddings)
        assert gap.centroid_cosine == pytest.approx(-1.0, abs=1e-9)
        assert gap.centroid_distance > 0.0

    def test_distance_matches_the_hand_computed_value(self) -> None:
        images = np.array([[1.0, 0.0], [1.0, 0.0]])
        texts = np.array([[0.0, 1.0], [0.0, 1.0]])
        gap = measure_modality_gap(images, texts)
        assert gap.centroid_distance == pytest.approx(np.sqrt(2.0))
        assert gap.centroid_cosine == pytest.approx(0.0)
        assert gap.image_centroid_norm == pytest.approx(1.0)
        assert gap.text_centroid_norm == pytest.approx(1.0)

    def test_a_zero_centroid_does_not_divide_by_zero(self) -> None:
        # Antipodal rows average to the origin; the cosine is undefined
        # and must be reported as 0.0 rather than raising or returning
        # NaN into a report file.
        images = np.array([[1.0, 0.0], [-1.0, 0.0]])
        texts = np.array([[0.0, 1.0], [0.0, 1.0]])
        gap = measure_modality_gap(images, texts)
        assert gap.centroid_cosine == 0.0
        assert gap.image_centroid_norm == pytest.approx(0.0)

    def test_accepts_different_row_counts(self) -> None:
        gap = measure_modality_gap(
            _unit_rows(40, EMBED_DIM), _unit_rows(25, EMBED_DIM, seed=11)
        )
        assert gap.num_samples == 25

    def test_rejects_mismatched_dimensions(self) -> None:
        with pytest.raises(ValueError, match="Dimension mismatch"):
            measure_modality_gap(
                _unit_rows(10, EMBED_DIM), _unit_rows(10, EMBED_DIM + 1)
            )

    def test_to_dict_is_json_serializable(self) -> None:
        gap = measure_modality_gap(
            _unit_rows(20, EMBED_DIM), _unit_rows(20, EMBED_DIM, seed=3)
        )
        payload = gap.to_dict()
        assert set(payload) == {
            "centroid_distance",
            "centroid_cosine",
            "image_centroid_norm",
            "text_centroid_norm",
            "num_samples",
        }
        assert all(isinstance(v, (int, float)) for v in payload.values())

    def test_is_immutable(self) -> None:
        gap = ModalityGap(
            centroid_distance=0.1,
            centroid_cosine=0.2,
            image_centroid_norm=0.3,
            text_centroid_norm=0.4,
            num_samples=5,
        )
        with pytest.raises(FrozenInstanceError):
            gap.centroid_distance = 0.9  # type: ignore[misc]
