"""Unit tests for vectormind.models.vectormind_model."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from vectormind.models.projection_head import ProjectionHead
from vectormind.models.vectormind_model import (
    DEFAULT_MAX_LOGIT_SCALE,
    _INITIAL_LOG_TEMPERATURE,
    VectorMindModel,
)


# ---------------------------------------------------------------------------
# Config fixtures — mirrors configs/model.yaml structure
# ---------------------------------------------------------------------------


@pytest.fixture
def full_config() -> dict:
    """Full model configuration matching configs/model.yaml."""
    return {
        "image_encoder": {
            "in_channels": 3,
            "base_channels": 64,
            "output_dim": 512,
        },
        "text_encoder": {
            "vocab_size": 30522,
            "max_seq_len": 77,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "ffn_dim": 1024,
            "dropout": 0.1,
        },
        "embedding": {"shared_dim": 256},
    }


@pytest.fixture
def small_config() -> dict:
    """Smaller config for faster unit tests."""
    return {
        "image_encoder": {
            "in_channels": 3,
            "base_channels": 16,
            "output_dim": 128,
        },
        "text_encoder": {
            "vocab_size": 1000,
            "max_seq_len": 32,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_dim": 256,
            "dropout": 0.0,
        },
        "embedding": {"shared_dim": 64},
    }


@pytest.fixture
def model(full_config: dict) -> VectorMindModel:
    """Full-size VectorMindModel."""
    return VectorMindModel(full_config)


@pytest.fixture
def small_model(small_config: dict) -> VectorMindModel:
    """Small VectorMindModel for faster tests."""
    return VectorMindModel(small_config)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestVectorMindModelInit:
    """Tests for model initialization."""

    def test_creates_all_components(self, model: VectorMindModel) -> None:
        """Model contains both encoders, both heads, and temperature."""
        assert hasattr(model, "image_encoder")
        assert hasattr(model, "text_encoder")
        assert hasattr(model, "image_projection")
        assert hasattr(model, "text_projection")
        assert hasattr(model, "log_temperature")

    def test_projection_head_types(self, model: VectorMindModel) -> None:
        """Both projection heads are ProjectionHead instances."""
        assert isinstance(model.image_projection, ProjectionHead)
        assert isinstance(model.text_projection, ProjectionHead)

    def test_image_projection_dimensions(self, model: VectorMindModel) -> None:
        """Image projection maps 512 -> 256."""
        assert model.image_projection.input_dim == 512
        assert model.image_projection.shared_dim == 256

    def test_text_projection_dimensions(self, model: VectorMindModel) -> None:
        """Text projection maps 256 -> 256."""
        assert model.text_projection.input_dim == 256
        assert model.text_projection.shared_dim == 256

    def test_temperature_init_value(self, model: VectorMindModel) -> None:
        """Temperature is initialized to CLIP convention log(1/0.07)."""
        assert torch.allclose(
            model.log_temperature,
            torch.tensor(_INITIAL_LOG_TEMPERATURE),
        )

    def test_temperature_is_learnable(self, model: VectorMindModel) -> None:
        """Temperature is a nn.Parameter (learnable)."""
        assert isinstance(model.log_temperature, nn.Parameter)

    def test_temperature_property(self, model: VectorMindModel) -> None:
        """temperature property returns exponentiated log_temperature."""
        expected_temp = torch.exp(model.log_temperature)
        assert torch.allclose(model.temperature, expected_temp)

    def test_total_params_positive(self, model: VectorMindModel) -> None:
        """Model has a positive number of parameters."""
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_missing_image_encoder_key_raises(self, full_config: dict) -> None:
        """Missing image_encoder config key raises KeyError."""
        del full_config["image_encoder"]
        with pytest.raises(KeyError):
            VectorMindModel(full_config)

    def test_missing_embedding_key_raises(self, full_config: dict) -> None:
        """Missing embedding config key raises KeyError."""
        del full_config["embedding"]
        with pytest.raises(KeyError):
            VectorMindModel(full_config)


# ---------------------------------------------------------------------------
# encode_image tests
# ---------------------------------------------------------------------------


class TestEncodeImage:
    """Tests for encode_image() API."""

    def test_output_shape(self, model: VectorMindModel) -> None:
        """Output shape is [B, shared_dim]."""
        images = torch.randn(4, 3, 224, 224)
        emb = model.encode_image(images)
        assert emb.shape == (4, 256)

    def test_l2_normalized(self, model: VectorMindModel) -> None:
        """Output vectors have unit L2 norm."""
        images = torch.randn(8, 3, 224, 224)
        emb = model.encode_image(images)
        norms = emb.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_batch_size_one(self, model: VectorMindModel) -> None:
        """Works with batch size 1."""
        images = torch.randn(1, 3, 224, 224)
        emb = model.encode_image(images)
        assert emb.shape == (1, 256)

    def test_deterministic(self, model: VectorMindModel) -> None:
        """Same input produces same output."""
        model.eval()
        images = torch.randn(4, 3, 224, 224)
        emb1 = model.encode_image(images)
        emb2 = model.encode_image(images)
        assert torch.allclose(emb1, emb2, atol=1e-6)


# ---------------------------------------------------------------------------
# encode_text tests
# ---------------------------------------------------------------------------


class TestEncodeText:
    """Tests for encode_text() API."""

    def test_output_shape(self, model: VectorMindModel) -> None:
        """Output shape is [B, shared_dim]."""
        input_ids = torch.randint(0, 30522, (4, 77))
        emb = model.encode_text(input_ids)
        assert emb.shape == (4, 256)

    def test_l2_normalized(self, model: VectorMindModel) -> None:
        """Output vectors have unit L2 norm."""
        input_ids = torch.randint(0, 30522, (8, 77))
        emb = model.encode_text(input_ids)
        norms = emb.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_with_attention_mask(self, model: VectorMindModel) -> None:
        """Works with an attention mask provided."""
        input_ids = torch.randint(0, 30522, (4, 77))
        attention_mask = torch.ones(4, 77)
        attention_mask[:, 50:] = 0  # pad last 27 positions
        emb = model.encode_text(input_ids, attention_mask)
        assert emb.shape == (4, 256)

    def test_batch_size_one(self, model: VectorMindModel) -> None:
        """Works with batch size 1."""
        input_ids = torch.randint(0, 30522, (1, 77))
        emb = model.encode_text(input_ids)
        assert emb.shape == (1, 256)

    def test_deterministic(self, model: VectorMindModel) -> None:
        """Same input produces same output."""
        model.eval()
        input_ids = torch.randint(0, 30522, (4, 77))
        emb1 = model.encode_text(input_ids)
        emb2 = model.encode_text(input_ids)
        assert torch.allclose(emb1, emb2, atol=1e-6)


# ---------------------------------------------------------------------------
# forward() tests
# ---------------------------------------------------------------------------


class TestForward:
    """Tests for the combined forward() method."""

    def test_returns_all_keys(self, model: VectorMindModel) -> None:
        """forward() returns image_embeddings, text_embeddings, temperature."""
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 77))
        result = model(images, input_ids)
        assert "image_embeddings" in result
        assert "text_embeddings" in result
        assert "temperature" in result

    def test_output_shapes(self, model: VectorMindModel) -> None:
        """Both embedding tensors have shape [B, shared_dim]."""
        B = 4
        images = torch.randn(B, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (B, 77))
        result = model(images, input_ids)
        assert result["image_embeddings"].shape == (B, 256)
        assert result["text_embeddings"].shape == (B, 256)

    def test_temperature_matches_property(self, model: VectorMindModel) -> None:
        """Temperature in output dict matches the property."""
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 77))
        result = model(images, input_ids)
        assert torch.allclose(result["temperature"], model.temperature)

    def test_l2_normalized(self, model: VectorMindModel) -> None:
        """Both embeddings are L2-normalized."""
        images = torch.randn(8, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (8, 77))
        result = model(images, input_ids)
        img_norms = result["image_embeddings"].norm(p=2, dim=-1)
        txt_norms = result["text_embeddings"].norm(p=2, dim=-1)
        assert torch.allclose(img_norms, torch.ones(8), atol=1e-5)
        assert torch.allclose(txt_norms, torch.ones(8), atol=1e-5)

    def test_with_attention_mask(self, model: VectorMindModel) -> None:
        """forward() passes attention mask to text encoder."""
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 77))
        attention_mask = torch.ones(2, 77)
        attention_mask[:, 60:] = 0
        result = model(images, input_ids, attention_mask)
        assert result["text_embeddings"].shape == (2, 256)


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradients:
    """Tests verifying gradient flow through the full model."""

    def test_image_encoder_gradients(self, small_model: VectorMindModel) -> None:
        """Gradients flow back to image encoder parameters."""
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        result = small_model(images, input_ids)
        loss = result["image_embeddings"].sum() + result["text_embeddings"].sum()
        loss.backward()

        # Check that image encoder conv1 gets gradients
        conv1_grad = small_model.image_encoder.conv1.weight.grad
        assert conv1_grad is not None
        assert conv1_grad.abs().sum() > 0

    def test_text_encoder_gradients(self, small_model: VectorMindModel) -> None:
        """Gradients flow back to text encoder parameters."""
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        result = small_model(images, input_ids)
        loss = result["image_embeddings"].sum() + result["text_embeddings"].sum()
        loss.backward()

        # Check that text encoder token embedding gets gradients
        emb_grad = small_model.text_encoder.token_embedding.weight.grad
        assert emb_grad is not None
        assert emb_grad.abs().sum() > 0

    def test_projection_head_gradients(self, small_model: VectorMindModel) -> None:
        """Gradients flow back to projection head parameters."""
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        result = small_model(images, input_ids)
        loss = result["image_embeddings"].sum() + result["text_embeddings"].sum()
        loss.backward()

        img_proj_grad = small_model.image_projection.projection.weight.grad
        txt_proj_grad = small_model.text_projection.projection.weight.grad
        assert img_proj_grad is not None
        assert img_proj_grad.abs().sum() > 0
        assert txt_proj_grad is not None
        assert txt_proj_grad.abs().sum() > 0

    def test_temperature_gradient(self, small_model: VectorMindModel) -> None:
        """Temperature parameter receives gradients."""
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        result = small_model(images, input_ids)
        loss = result["temperature"]
        loss.backward()
        assert small_model.log_temperature.grad is not None


class TestClampLogTemperature:
    """Regression guard for the Phase 4 collapse (docs/KNOWN_ISSUES.md §1).

    The learnable logit scale is unconstrained. Left unclamped it ran to
    500+ during Phase 4 while the embedding space collapsed into a cone.
    These tests pin the ceiling in place.
    """

    def test_clamps_value_above_ceiling(
        self, small_model: VectorMindModel
    ) -> None:
        with torch.no_grad():
            small_model.log_temperature.fill_(6.5)  # logit scale ~665
        assert small_model.clamp_log_temperature(100.0) == pytest.approx(
            100.0, rel=1e-4
        )

    def test_leaves_value_below_ceiling_untouched(
        self, small_model: VectorMindModel
    ) -> None:
        with torch.no_grad():
            small_model.log_temperature.fill_(2.0)
        before = small_model.temperature.item()
        after = small_model.clamp_log_temperature(100.0)
        assert after == pytest.approx(before)

    def test_clip_init_is_below_default_ceiling(
        self, small_model: VectorMindModel
    ) -> None:
        """CLIP's log(1/0.07) init must survive the default clamp."""
        expected = math.exp(_INITIAL_LOG_TEMPERATURE)
        assert expected < DEFAULT_MAX_LOGIT_SCALE
        assert small_model.clamp_log_temperature() == pytest.approx(expected)

    def test_is_idempotent(self, small_model: VectorMindModel) -> None:
        with torch.no_grad():
            small_model.log_temperature.fill_(9.0)
        first = small_model.clamp_log_temperature(100.0)
        second = small_model.clamp_log_temperature(100.0)
        assert first == pytest.approx(second)

    def test_mutates_in_place(self, small_model: VectorMindModel) -> None:
        with torch.no_grad():
            small_model.log_temperature.fill_(9.0)
        small_model.clamp_log_temperature(100.0)
        assert small_model.log_temperature.item() == pytest.approx(
            math.log(100.0), rel=1e-5
        )

    def test_does_not_track_gradients(
        self, small_model: VectorMindModel
    ) -> None:
        """The clamp must not enter the autograd graph."""
        with torch.no_grad():
            small_model.log_temperature.fill_(9.0)
        small_model.clamp_log_temperature(100.0)
        assert small_model.log_temperature.grad is None
        assert small_model.log_temperature.requires_grad is True

    def test_rejects_ceiling_at_or_below_one(
        self, small_model: VectorMindModel
    ) -> None:
        with pytest.raises(ValueError, match="must be > 1.0"):
            small_model.clamp_log_temperature(1.0)
        with pytest.raises(ValueError, match="must be > 1.0"):
            small_model.clamp_log_temperature(0.5)

    def test_default_ceiling_matches_clip(self) -> None:
        assert DEFAULT_MAX_LOGIT_SCALE == 100.0
