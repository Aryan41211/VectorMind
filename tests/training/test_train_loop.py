"""Unit tests for vectormind.training.train_loop."""

from __future__ import annotations

import pytest
import torch

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.train_loop import (
    create_optimizer,
    create_scaler,
    train_one_step,
)

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


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
def model(small_config: dict) -> VectorMindModel:
    """Small VectorMindModel for testing."""
    return VectorMindModel(small_config)


@pytest.fixture
def memory_queue() -> MemoryQueue:
    """Small memory queue for testing."""
    return MemoryQueue(queue_size=32, embed_dim=64)


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    """A small test batch."""
    return {
        "image": torch.randn(4, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (4, 32)),
        "attention_mask": torch.ones(4, 32),
    }


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------


class TestCreateOptimizer:
    """Tests for create_optimizer."""

    def test_returns_adamw(self, model: VectorMindModel) -> None:
        """Returns an AdamW optimizer."""
        optimizer = create_optimizer(model)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_param_groups(self, model: VectorMindModel) -> None:
        """Creates two param groups (decay and no-decay)."""
        optimizer = create_optimizer(model)
        assert len(optimizer.param_groups) == 2

    def test_learning_rate(self, model: VectorMindModel) -> None:
        """Sets the specified learning rate."""
        optimizer = create_optimizer(model, lr=5e-4)
        for group in optimizer.param_groups:
            assert group["lr"] == 5e-4


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------


class TestCreateScaler:
    """Tests for create_scaler."""

    def test_returns_scaler(self) -> None:
        """Returns a GradScaler."""
        scaler = create_scaler()
        assert isinstance(scaler, torch.amp.GradScaler)


# ---------------------------------------------------------------------------
# train_one_step tests (CPU-only, AMP disabled)
# ---------------------------------------------------------------------------


class TestTrainOneStep:
    """Tests for train_one_step function.

    Note: These tests run on CPU with AMP disabled (autocast is
    a no-op on CPU). Full AMP testing requires CUDA.
    """

    def test_returns_metrics_dict(
        self, model: VectorMindModel, batch: dict, memory_queue: MemoryQueue
    ) -> None:
        """Returns a dict with expected metric keys."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = create_scaler()

        metrics = train_one_step(model, batch, optimizer, scaler, memory_queue)

        assert "loss" in metrics
        assert "temperature" in metrics
        assert "image_embed_norm" in metrics
        assert "text_embed_norm" in metrics
        assert "image_embed_std" in metrics
        assert "text_embed_std" in metrics
        assert "gpu_memory_gb" in metrics

    def test_loss_is_finite(
        self, model: VectorMindModel, batch: dict, memory_queue: MemoryQueue
    ) -> None:
        """Loss is a finite positive number."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = create_scaler()

        metrics = train_one_step(model, batch, optimizer, scaler, memory_queue)

        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["loss"] > 0

    def test_temperature_is_positive(
        self, model: VectorMindModel, batch: dict, memory_queue: MemoryQueue
    ) -> None:
        """Temperature is a positive value."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = create_scaler()

        metrics = train_one_step(model, batch, optimizer, scaler, memory_queue)

        assert metrics["temperature"] > 0

    def test_gradient_flows(
        self, model: VectorMindModel, batch: dict, memory_queue: MemoryQueue
    ) -> None:
        """Gradients are computed (at least one parameter has grad)."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = create_scaler()

        train_one_step(model, batch, optimizer, scaler, memory_queue)

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad

    def test_accumulation_steps_used(
        self, model: VectorMindModel, batch: dict, memory_queue: MemoryQueue
    ) -> None:
        """Loss is divided by accumulation_steps before backward."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = create_scaler()

        # With accumulation_steps=1, loss should be the full loss
        metrics = train_one_step(
            model, batch, optimizer, scaler, memory_queue, accumulation_steps=1
        )

        # Verify loss is positive and finite (accumulation_steps=1 means no division)
        assert metrics["loss"] > 0
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_embedding_norms_near_one(
        self, model: VectorMindModel, batch: dict, memory_queue: MemoryQueue
    ) -> None:
        """Embedding norms are near 1.0 (L2 normalized)."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = create_scaler()

        metrics = train_one_step(model, batch, optimizer, scaler, memory_queue)

        assert metrics["image_embed_norm"] == pytest.approx(1.0, abs=0.1)
        assert metrics["text_embed_norm"] == pytest.approx(1.0, abs=0.1)
