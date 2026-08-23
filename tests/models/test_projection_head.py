"""Unit tests for vectormind.models.projection_head."""

from __future__ import annotations

import pytest
import torch

from vectormind.models.projection_head import ProjectionHead

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_head() -> ProjectionHead:
    """Image projection head: 512 -> 256."""
    return ProjectionHead(input_dim=512, shared_dim=256)


@pytest.fixture
def text_head() -> ProjectionHead:
    """Text projection head: 256 -> 256."""
    return ProjectionHead(input_dim=256, shared_dim=256)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestProjectionHeadInit:
    """Tests for projection head initialization."""

    def test_stores_dimensions(self, image_head: ProjectionHead) -> None:
        """input_dim and shared_dim are stored as attributes."""
        assert image_head.input_dim == 512
        assert image_head.shared_dim == 256

    def test_linear_layer_shape(self, image_head: ProjectionHead) -> None:
        """Linear layer weight shape matches input_dim x shared_dim."""
        weight = image_head.projection.weight
        assert weight.shape == (256, 512)

    def test_linear_layer_has_bias(self, image_head: ProjectionHead) -> None:
        """Linear layer has a bias term."""
        assert image_head.projection.bias is not None
        assert image_head.projection.bias.shape == (256,)

    def test_text_head_layer_shape(self, text_head: ProjectionHead) -> None:
        """Text head linear layer shape is 256 x 256."""
        assert text_head.projection.weight.shape == (256, 256)

    def test_zero_input_dim_raises(self) -> None:
        """Zero input_dim raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ProjectionHead(input_dim=0, shared_dim=256)

    def test_negative_shared_dim_raises(self) -> None:
        """Negative shared_dim raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ProjectionHead(input_dim=512, shared_dim=-1)

    def test_both_zero_raises(self) -> None:
        """Both dimensions zero raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ProjectionHead(input_dim=0, shared_dim=0)


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestProjectionHeadForward:
    """Tests for projection head forward pass."""

    def test_output_shape(self, image_head: ProjectionHead) -> None:
        """Output shape is [B, shared_dim]."""
        x = torch.randn(4, 512)
        out = image_head(x)
        assert out.shape == (4, 256)

    def test_text_head_output_shape(self, text_head: ProjectionHead) -> None:
        """Text head output shape is [B, 256]."""
        x = torch.randn(8, 256)
        out = text_head(x)
        assert out.shape == (8, 256)

    def test_batch_size_one(self, image_head: ProjectionHead) -> None:
        """Works with batch size 1."""
        x = torch.randn(1, 512)
        out = image_head(x)
        assert out.shape == (1, 256)

    def test_large_batch(self, image_head: ProjectionHead) -> None:
        """Works with large batch size."""
        x = torch.randn(256, 512)
        out = image_head(x)
        assert out.shape == (256, 256)

    def test_output_is_l2_normalized(self, image_head: ProjectionHead) -> None:
        """Output vectors have unit L2 norm."""
        x = torch.randn(16, 512)
        out = image_head(x)
        norms = torch.norm(out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5)

    def test_l2_norm_exact(self, text_head: ProjectionHead) -> None:
        """L2 norm is exactly 1.0 within float tolerance."""
        x = torch.randn(32, 256)
        out = text_head(x)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-6)

    def test_deterministic(self, image_head: ProjectionHead) -> None:
        """Same input produces same output."""
        image_head.eval()
        x = torch.randn(4, 512)
        out1 = image_head(x)
        out2 = image_head(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_different_inputs_differ(self, image_head: ProjectionHead) -> None:
        """Different inputs produce different outputs."""
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)
        out1 = image_head(x1)
        out2 = image_head(x2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self, image_head: ProjectionHead) -> None:
        """Gradients propagate through the projection layer."""
        x = torch.randn(4, 512, requires_grad=True)
        out = image_head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 512)

    def test_linear_weight_gradient(self, image_head: ProjectionHead) -> None:
        """Linear layer weight receives gradients."""
        x = torch.randn(4, 512)
        out = image_head(x)
        loss = out.sum()
        loss.backward()
        assert image_head.projection.weight.grad is not None

    def test_train_eval_same_output(self, image_head: ProjectionHead) -> None:
        """Train and eval modes produce the same output (no dropout)."""
        x = torch.randn(4, 512)
        image_head.train()
        out_train = image_head(x)
        image_head.eval()
        out_eval = image_head(x)
        assert torch.allclose(out_train, out_eval, atol=1e-6)


# ---------------------------------------------------------------------------
# Cosine similarity test
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests verifying L2-normalized outputs work with dot-product similarity."""

    def test_dot_equals_cosine(self, image_head: ProjectionHead) -> None:
        """Dot product of L2-normalized vectors equals cosine similarity."""
        x1 = torch.randn(1, 512)
        x2 = torch.randn(1, 512)
        emb1 = image_head(x1)
        emb2 = image_head(x2)
        dot = (emb1 * emb2).sum(dim=-1)
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
        assert torch.allclose(dot, cos_sim, atol=1e-5)

    def test_self_similarity_is_one(self, image_head: ProjectionHead) -> None:
        """Dot product of a vector with itself is 1.0."""
        x = torch.randn(1, 512)
        emb = image_head(x)
        sim = (emb * emb).sum(dim=-1)
        assert torch.allclose(sim, torch.ones(1), atol=1e-5)
