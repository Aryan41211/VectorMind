"""Unit tests for vectormind.models.image_encoder."""

from __future__ import annotations

import pytest
import torch

from vectormind.models.image_encoder import BasicBlock, ImageEncoder

# ---------------------------------------------------------------------------
# Config fixture — mirrors configs/model.yaml structure
# ---------------------------------------------------------------------------


@pytest.fixture
def model_config() -> dict:
    """Minimal model configuration matching configs/model.yaml."""
    return {
        "image_encoder": {
            "in_channels": 3,
            "base_channels": 64,
            "output_dim": 512,
        },
    }


@pytest.fixture
def small_config() -> dict:
    """Smaller config for faster unit tests (base_channels=16)."""
    return {
        "image_encoder": {
            "in_channels": 3,
            "base_channels": 16,
            "output_dim": 128,
        },
    }


# ---------------------------------------------------------------------------
# BasicBlock tests
# ---------------------------------------------------------------------------


class TestBasicBlock:
    """Tests for the ResNet basic residual block."""

    def test_forward_shape_preserved(self) -> None:
        """Output spatial dimensions match input when stride=1."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_forward_shape_with_stride(self) -> None:
        """Output spatial dimensions are halved when stride=2."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_shortcut_projection_on_dimension_mismatch(self) -> None:
        """Shortcut uses 1x1 projection when channels change."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=2)
        # Verify the shortcut is a Sequential (has projection), not Identity
        assert not isinstance(block.shortcut, torch.nn.Identity)

    def test_shortcut_identity_on_dimension_match(self) -> None:
        """Shortcut is identity when channels and stride match."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=1)
        assert isinstance(block.shortcut, torch.nn.Identity)

    def test_no_nan_in_output(self) -> None:
        """Forward pass produces no NaN values with random input."""
        block = BasicBlock(in_channels=32, out_channels=32)
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        assert not torch.isnan(out).any()

    def test_invalid_channels_raises(self) -> None:
        """Non-positive channel dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            BasicBlock(in_channels=0, out_channels=64)
        with pytest.raises(ValueError, match="must be positive"):
            BasicBlock(in_channels=64, out_channels=-1)


# ---------------------------------------------------------------------------
# ImageEncoder tests
# ---------------------------------------------------------------------------


class TestImageEncoder:
    """Tests for the full ImageEncoder module."""

    def test_output_shape(self, small_config: dict) -> None:
        """Output shape is [B, output_dim] for standard 224x224 input."""
        encoder = ImageEncoder(small_config)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_output_shape_full_config(self, model_config: dict) -> None:
        """Output shape with full ResNet-18 config."""
        encoder = ImageEncoder(model_config)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 512)

    def test_no_nan_in_output(self, small_config: dict) -> None:
        """Forward pass produces no NaN values."""
        encoder = ImageEncoder(small_config)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self, small_config: dict) -> None:
        """Forward pass produces no Inf values."""
        encoder = ImageEncoder(small_config)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert not torch.isinf(out).any()

    def test_batch_independence(self, small_config: dict) -> None:
        """Different batch elements produce different outputs."""
        encoder = ImageEncoder(small_config)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        # Outputs for different samples should not be identical
        assert not torch.equal(out[0], out[1])

    def test_gradient_flow(self, small_config: dict) -> None:
        """Gradients flow through the entire encoder."""
        encoder = ImageEncoder(small_config)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_output_dim_raises(self) -> None:
        """Config with mismatched output_dim raises ValueError."""
        bad_config = {
            "image_encoder": {
                "in_channels": 3,
                "base_channels": 64,
                "output_dim": 999,  # not 64*8=512
            },
        }
        with pytest.raises(ValueError, match="output_dim"):
            ImageEncoder(bad_config)

    def test_output_dim_property(self, small_config: dict) -> None:
        """output_dim property returns the configured output dimension."""
        encoder = ImageEncoder(small_config)
        assert encoder.output_dim == 128

    def test_deterministic_eval(self, small_config: dict) -> None:
        """Same input produces same output in eval mode."""
        encoder = ImageEncoder(small_config)
        encoder.eval()
        x = torch.randn(1, 3, 224, 224)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_parameter_count(self, small_config: dict) -> None:
        """Encoder has a reasonable number of parameters."""
        encoder = ImageEncoder(small_config)
        num_params = sum(p.numel() for p in encoder.parameters())
        # ResNet-18 style with base_channels=16 should have ~100k params
        assert num_params > 10_000
        assert num_params < 1_000_000

    def test_small_batch(self, small_config: dict) -> None:
        """Encoder works with batch size 1."""
        encoder = ImageEncoder(small_config)
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (1, 128)

    def test_stages_produce_correct_spatial_dims(self, small_config: dict) -> None:
        """Feature maps have correct spatial dimensions at each stage."""
        encoder = ImageEncoder(small_config)
        x = torch.randn(1, 3, 224, 224)

        # After conv1 (stride 2) + maxpool (stride 2): 224 -> 56
        x = encoder.relu(encoder.bn1(encoder.conv1(x)))
        x = encoder.maxpool(x)
        assert x.shape == (1, 16, 56, 56)

        # Stage 1 (stride 1): 56x56
        x = encoder.stage1(x)
        assert x.shape == (1, 16, 56, 56)

        # Stage 2 (stride 2): 28x28
        x = encoder.stage2(x)
        assert x.shape == (1, 32, 28, 28)

        # Stage 3 (stride 2): 14x14
        x = encoder.stage3(x)
        assert x.shape == (1, 64, 14, 14)

        # Stage 4 (stride 2): 7x7
        x = encoder.stage4(x)
        assert x.shape == (1, 128, 7, 7)
