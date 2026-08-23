"""Tests for embedding-space health metrics.

The point of this module is that it must fire on the exact failure the
Phase 4 checkpoint exhibited — a narrow cone of embeddings that
per-dimension variance called acceptable. The synthetic cone test below
reproduces that geometry, so a regression in the thresholds or the
separation math fails here rather than in a shipped report.
"""

from __future__ import annotations

import math

import pytest
import torch

from vectormind.evaluation.embedding_health import (
    ANISOTROPY_COLLAPSE_THRESHOLD,
    MEAN_NORM_COLLAPSE_THRESHOLD,
    SEPARATION_COLLAPSE_THRESHOLD,
    EmbeddingHealth,
    compute_embedding_health,
)

DIM = 64
N = 256


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def _healthy_pair(n: int = N, dim: int = DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """Well-separated space: matched pairs close, everything else spread."""
    torch.manual_seed(0)
    image = _normalize(torch.randn(n, dim))
    # Text sits very near its matched image, so separation is large.
    text = _normalize(image + 0.05 * torch.randn(n, dim))
    return image, text


def _collapsed_pair(n: int = N, dim: int = DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow cone: all embeddings share a dominant direction.

    This is the Phase 4 geometry — a shared mean direction with small
    per-sample jitter, which keeps per-dimension variance nonzero while
    driving every pairwise cosine close to 1. The jitter is tuned so the
    mean off-diagonal cosine lands near 0.8, matching what was measured
    on the real Phase 4 checkpoint (0.810 image / 0.881 text).
    """
    torch.manual_seed(0)
    axis = _normalize(torch.randn(1, dim))
    image = _normalize(axis + 0.06 * torch.randn(n, dim))
    text = _normalize(axis + 0.06 * torch.randn(n, dim))
    return image, text


class TestHealthySpace:
    def test_reports_not_collapsed(self) -> None:
        health = compute_embedding_health(*_healthy_pair())
        assert health.collapsed is False
        assert health.verdict.startswith("HEALTHY")

    def test_separation_is_large(self) -> None:
        health = compute_embedding_health(*_healthy_pair())
        assert health.separation > 0.9

    def test_mean_embedding_near_origin(self) -> None:
        health = compute_embedding_health(*_healthy_pair())
        assert health.image_mean_norm < MEAN_NORM_COLLAPSE_THRESHOLD
        assert health.text_mean_norm < MEAN_NORM_COLLAPSE_THRESHOLD

    def test_offdiagonal_cosine_near_zero(self) -> None:
        health = compute_embedding_health(*_healthy_pair())
        assert abs(health.image_mean_cosine) < 0.1


class TestCollapsedSpace:
    """The regression guard for the Phase 4 failure."""

    def test_detects_cone(self) -> None:
        health = compute_embedding_health(*_collapsed_pair())
        assert health.collapsed is True
        assert health.verdict.startswith("COLLAPSED")

    def test_separation_below_threshold(self) -> None:
        health = compute_embedding_health(*_collapsed_pair())
        assert health.separation < SEPARATION_COLLAPSE_THRESHOLD

    def test_anisotropy_detected(self) -> None:
        health = compute_embedding_health(*_collapsed_pair())
        assert health.image_mean_cosine > ANISOTROPY_COLLAPSE_THRESHOLD

    def test_variance_alone_would_have_missed_it(self) -> None:
        """Documents *why* this module exists.

        The collapsed space still has nonzero per-dimension variance —
        which is exactly why the Phase 4 report called it healthy. The
        separation check is what catches it.
        """
        health = compute_embedding_health(*_collapsed_pair())
        assert health.image_dim_variance > 0.0
        assert health.collapsed is True

    def test_identical_embeddings_are_total_collapse(self) -> None:
        vec = _normalize(torch.randn(1, DIM)).repeat(N, 1)
        health = compute_embedding_health(vec, vec.clone())
        assert health.collapsed is True
        assert health.image_mean_norm == pytest.approx(1.0, abs=1e-4)
        assert health.separation == pytest.approx(0.0, abs=1e-4)


class TestValidation:
    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            compute_embedding_health(torch.randn(N, DIM), torch.randn(N, DIM * 2))

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            compute_embedding_health(torch.randn(N), torch.randn(N))

    def test_rejects_single_embedding(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_embedding_health(torch.randn(1, DIM), torch.randn(1, DIM))


class TestSubsampling:
    def test_caps_sample_count(self) -> None:
        image, text = _healthy_pair(n=500)
        health = compute_embedding_health(image, text, max_samples=100)
        assert health.num_samples == 100

    def test_is_deterministic(self) -> None:
        image, text = _healthy_pair(n=500)
        a = compute_embedding_health(image, text, max_samples=100)
        b = compute_embedding_health(image, text, max_samples=100)
        assert a.separation == pytest.approx(b.separation)

    def test_variance_uses_full_set_not_subsample(self) -> None:
        """Variance is O(N); it should not be degraded by the cap."""
        image, text = _healthy_pair(n=500)
        capped = compute_embedding_health(image, text, max_samples=100)
        full = compute_embedding_health(image, text, max_samples=500)
        assert capped.image_dim_variance == pytest.approx(full.image_dim_variance)


class TestSerialization:
    def test_to_dict_round_trips(self) -> None:
        health = compute_embedding_health(*_healthy_pair())
        payload = health.to_dict()
        assert set(payload) == set(EmbeddingHealth.__dataclass_fields__)
        assert isinstance(payload["separation"], float)
        assert isinstance(payload["collapsed"], bool)

    def test_all_values_are_finite(self) -> None:
        health = compute_embedding_health(*_healthy_pair())
        for key, value in health.to_dict().items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{key} is not finite"
