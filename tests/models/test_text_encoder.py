"""Unit tests for vectormind.models.text_encoder."""

from __future__ import annotations

import pytest
import torch

from vectormind.models.text_encoder import TextEncoder, TransformerBlock

# ---------------------------------------------------------------------------
# Config fixture — mirrors configs/model.yaml structure
# ---------------------------------------------------------------------------


@pytest.fixture
def model_config() -> dict:
    """Minimal model configuration matching configs/model.yaml."""
    return {
        "text_encoder": {
            "vocab_size": 30522,
            "max_seq_len": 77,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "ffn_dim": 1024,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def small_config() -> dict:
    """Smaller config for faster unit tests."""
    return {
        "text_encoder": {
            "vocab_size": 1000,
            "max_seq_len": 32,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_dim": 256,
            "dropout": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# TransformerBlock tests
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    """Tests for the Transformer encoder block."""

    def test_forward_shape(self) -> None:
        """Output shape matches input shape."""
        block = TransformerBlock(
            embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.0,
        )
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_no_nan_in_output(self) -> None:
        """Forward pass produces no NaN values."""
        block = TransformerBlock(
            embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.0,
        )
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert not torch.isnan(out).any()

    def test_incompatible_heads_raises(self) -> None:
        """embed_dim not divisible by num_heads raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            TransformerBlock(
                embed_dim=65, num_heads=4, ffn_dim=256, dropout=0.0,
            )

    def test_residual_connection(self) -> None:
        """Block output differs from input (residual adds information)."""
        block = TransformerBlock(
            embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.0,
        )
        x = torch.randn(1, 5, 64)
        out = block(x)
        # With a trained block, output should differ from input
        # (even with random init, the attention + FFN modify the input)
        assert not torch.allclose(x, out, atol=1e-5)

    def test_deterministic_eval(self) -> None:
        """Same input produces same output in eval mode."""
        block = TransformerBlock(
            embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.0,
        )
        block.eval()
        x = torch.randn(1, 5, 64)
        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self) -> None:
        """Gradients flow through all parameters."""
        block = TransformerBlock(
            embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.0,
        )
        x = torch.randn(1, 5, 64)
        out = block(x)
        out.sum().backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# TextEncoder tests
# ---------------------------------------------------------------------------


class TestTextEncoder:
    """Tests for the full TextEncoder module."""

    def test_output_shape(self, small_config: dict) -> None:
        """Output shape is [B, embed_dim] for standard input."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        out = encoder(input_ids)
        assert out.shape == (2, 64)

    def test_output_shape_full_config(self, model_config: dict) -> None:
        """Output shape with full config."""
        encoder = TextEncoder(model_config)
        input_ids = torch.randint(0, 30522, (2, 77))
        out = encoder(input_ids)
        assert out.shape == (2, 256)

    def test_no_nan_in_output(self, small_config: dict) -> None:
        """Forward pass produces no NaN values."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        out = encoder(input_ids)
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self, small_config: dict) -> None:
        """Forward pass produces no Inf values."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        out = encoder(input_ids)
        assert not torch.isinf(out).any()

    def test_with_attention_mask(self, small_config: dict) -> None:
        """Encoder handles attention mask correctly."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        # Set last 5 positions as padding
        attention_mask[:, -5:] = 0

        out = encoder(input_ids, attention_mask=attention_mask)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()

    def test_padding_mask_changes_output(self, small_config: dict) -> None:
        """Different attention masks produce different pooled outputs."""
        encoder = TextEncoder(small_config)
        encoder.eval()
        input_ids = torch.randint(0, 1000, (1, 32))

        # Mask 1: all real tokens
        mask1 = torch.ones(1, 32, dtype=torch.long)
        out1 = encoder(input_ids, attention_mask=mask1)

        # Mask 2: first 16 real, last 16 padding
        mask2 = torch.ones(1, 32, dtype=torch.long)
        mask2[:, 16:] = 0
        out2 = encoder(input_ids, attention_mask=mask2)

        # Outputs should differ because mean pooling covers different tokens
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_batch_independence(self, small_config: dict) -> None:
        """Different batch elements produce different outputs."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        out = encoder(input_ids)
        assert not torch.equal(out[0], out[1])

    def test_gradient_flow(self, small_config: dict) -> None:
        """Gradients flow through the entire encoder."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        out = encoder(input_ids)
        out.sum().backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_eval(self, small_config: dict) -> None:
        """Same input produces same output in eval mode."""
        encoder = TextEncoder(small_config)
        encoder.eval()
        input_ids = torch.randint(0, 1000, (1, 32))
        out1 = encoder(input_ids)
        out2 = encoder(input_ids)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_parameter_count(self, small_config: dict) -> None:
        """Encoder has a reasonable number of parameters."""
        encoder = TextEncoder(small_config)
        num_params = sum(p.numel() for p in encoder.parameters())
        # Small config should have ~100k params
        assert num_params > 10_000
        assert num_params < 500_000

    def test_single_token_input(self, small_config: dict) -> None:
        """Encoder works with a single token (no padding)."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (1, 1))
        out = encoder(input_ids)
        assert out.shape == (1, 64)

    def test_all_padding_input(self, small_config: dict) -> None:
        """Encoder handles all-padding input without NaN (edge case)."""
        encoder = TextEncoder(small_config)
        input_ids = torch.zeros(1, 32, dtype=torch.long)
        attention_mask = torch.zeros(1, 32, dtype=torch.long)
        out = encoder(input_ids, attention_mask=attention_mask)
        assert out.shape == (1, 64)
        # Output will be zero (mean of zeros), not NaN
        assert not torch.isnan(out).any()

    def test_embeddings_are_learned(self, small_config: dict) -> None:
        """Token and position embeddings have gradient updates."""
        encoder = TextEncoder(small_config)
        input_ids = torch.randint(0, 1000, (1, 32))
        out = encoder(input_ids)
        out.sum().backward()

        # Check token embedding gradient exists
        assert encoder.token_embedding.weight.grad is not None
        # Check position embedding gradient exists
        assert encoder.position_embedding.weight.grad is not None
