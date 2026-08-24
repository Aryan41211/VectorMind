"""Tests for the uniformity regularizer.

The property that matters is directional: a spread space must score
lower than a clustered one, and the gradient must actually push toward
spreading. A regularizer that is merely finite and differentiable would
pass a weaker test suite while doing nothing.
"""

from __future__ import annotations

import pytest
import torch

from vectormind.training.uniformity import (
    DEFAULT_UNIFORMITY_T,
    MAX_UNIFORMITY_SAMPLES,
    combined_uniformity_loss,
    uniformity_loss,
)

DIM = 32
N = 128


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def _spread(n: int = N, dim: int = DIM) -> torch.Tensor:
    torch.manual_seed(0)
    return _normalize(torch.randn(n, dim))


def _cone(n: int = N, dim: int = DIM, jitter: float = 0.05) -> torch.Tensor:
    torch.manual_seed(0)
    axis = _normalize(torch.randn(1, dim))
    return _normalize(axis + jitter * torch.randn(n, dim))


class TestDirection:
    """The regularizer must prefer spread to clustered."""

    def test_cone_scores_worse_than_spread(self) -> None:
        assert uniformity_loss(_cone()) > uniformity_loss(_spread())

    def test_tighter_cone_scores_worse(self) -> None:
        loose = uniformity_loss(_cone(jitter=0.30))
        tight = uniformity_loss(_cone(jitter=0.02))
        assert tight > loose

    def test_identical_embeddings_are_the_worst_case(self) -> None:
        """Every distance is zero, so every exponential is one, so log = 0."""
        vec = _normalize(torch.randn(1, DIM)).repeat(N, 1)
        assert uniformity_loss(vec) == pytest.approx(0.0, abs=1e-5)

    def test_spread_is_negative(self) -> None:
        assert uniformity_loss(_spread()) < 0.0


class TestGradient:
    def test_is_differentiable(self) -> None:
        x = _spread().requires_grad_(True)
        uniformity_loss(x).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_descending_it_actually_spreads_a_cone(self) -> None:
        """The property the whole module exists for.

        A few gradient steps on the loss alone must measurably widen a
        collapsed set — otherwise the term is decoration.
        """
        x = _cone().clone().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr=1.0)

        def mean_norm(t: torch.Tensor) -> float:
            return _normalize(t).mean(dim=0).norm().item()

        before = mean_norm(x.detach())
        for _ in range(30):
            optimizer.zero_grad()
            uniformity_loss(_normalize(x)).backward()
            optimizer.step()
        after = mean_norm(x.detach())

        assert after < before, f"mean norm rose: {before:.4f} -> {after:.4f}"


class TestValidation:
    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            uniformity_loss(torch.randn(N))

    def test_rejects_single_embedding(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            uniformity_loss(torch.randn(1, DIM))

    @pytest.mark.parametrize("bad_t", [0.0, -1.0])
    def test_rejects_non_positive_t(self, bad_t: float) -> None:
        with pytest.raises(ValueError, match="t must be positive"):
            uniformity_loss(_spread(), t=bad_t)


class TestSampleCap:
    def test_caps_the_pairwise_computation(self) -> None:
        big = _spread(n=MAX_UNIFORMITY_SAMPLES + 500)
        assert torch.isfinite(uniformity_loss(big))

    def test_cap_does_not_change_the_verdict(self) -> None:
        """Capping must not flip spread-versus-clustered."""
        n = MAX_UNIFORMITY_SAMPLES + 500
        assert uniformity_loss(_cone(n=n)) > uniformity_loss(_spread(n=n))


class TestCombined:
    def test_averages_both_modalities(self) -> None:
        image, text = _spread(), _cone()
        expected = (uniformity_loss(image) + uniformity_loss(text)) / 2
        assert combined_uniformity_loss(image, text) == pytest.approx(
            expected.item(), rel=1e-5
        )

    def test_one_collapsed_tower_worsens_the_score(self) -> None:
        """Health is the worse tower, so one bad tower must show up."""
        both_good = combined_uniformity_loss(_spread(), _spread())
        one_bad = combined_uniformity_loss(_spread(), _cone())
        assert one_bad > both_good

    def test_is_differentiable_through_both(self) -> None:
        image = _spread().requires_grad_(True)
        text = _spread().requires_grad_(True)
        combined_uniformity_loss(image, text).backward()
        assert image.grad is not None
        assert text.grad is not None


class TestDefaults:
    def test_default_t_matches_the_paper(self) -> None:
        assert DEFAULT_UNIFORMITY_T == 2.0

    def test_t_changes_the_value(self) -> None:
        x = _spread()
        assert uniformity_loss(x, t=1.0) != uniformity_loss(x, t=4.0)
