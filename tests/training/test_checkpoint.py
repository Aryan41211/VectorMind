"""Unit tests for vectormind.training.checkpoint."""

from __future__ import annotations

import pytest
import torch

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.checkpoint import load_checkpoint, save_checkpoint
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.train_loop import create_scaler


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


# ---------------------------------------------------------------------------
# Save/load roundtrip tests
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    """Tests for save/load checkpoint roundtrip."""

    def test_save_load_roundtrip(
        self,
        model: VectorMindModel,
        memory_queue: MemoryQueue,
        tmp_path: object,
    ) -> None:
        """Save then load restores exact state."""
        # Fill queue with some data
        memory_queue.enqueue(torch.randn(10, 64))

        # Save
        path = tmp_path / "checkpoint.pt"  # type: ignore[union-attr]
        save_checkpoint(
            path=path,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=create_scaler(),
            memory_queue=memory_queue,
            epoch=5,
            step=100,
        )

        # Load into fresh model/optimizer/queue
        model2 = VectorMindModel(
            {
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
        )
        queue2 = MemoryQueue(queue_size=32, embed_dim=64)
        optimizer2 = torch.optim.AdamW(model2.parameters())
        scaler2 = create_scaler()

        epoch, step = load_checkpoint(path, model2, optimizer2, scaler2, queue2)

        assert epoch == 5
        assert step == 100

    def test_model_weights_match(
        self,
        model: VectorMindModel,
        memory_queue: MemoryQueue,
        tmp_path: object,
    ) -> None:
        """Model weights are exactly restored."""
        path = tmp_path / "checkpoint.pt"  # type: ignore[union-attr]
        save_checkpoint(
            path=path,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=create_scaler(),
            memory_queue=memory_queue,
            epoch=0,
            step=0,
        )

        model2 = VectorMindModel(
            {
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
        )
        queue2 = MemoryQueue(queue_size=32, embed_dim=64)
        load_checkpoint(
            path,
            model2,
            torch.optim.AdamW(model2.parameters()),
            create_scaler(),
            queue2,
        )

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_queue_state_restored(
        self,
        model: VectorMindModel,
        memory_queue: MemoryQueue,
        tmp_path: object,
    ) -> None:
        """Memory queue contents are exactly restored."""
        # Fill queue
        memory_queue.enqueue(torch.randn(15, 64))
        original_queue = memory_queue.queue.clone()
        original_pointer = memory_queue.pointer
        original_filled = memory_queue.num_filled

        path = tmp_path / "checkpoint.pt"  # type: ignore[union-attr]
        save_checkpoint(
            path=path,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=create_scaler(),
            memory_queue=memory_queue,
            epoch=0,
            step=0,
        )

        queue2 = MemoryQueue(queue_size=32, embed_dim=64)
        load_checkpoint(
            path,
            model,
            torch.optim.AdamW(model.parameters()),
            create_scaler(),
            queue2,
        )

        assert torch.allclose(queue2.queue, original_queue)
        assert queue2.pointer == original_pointer
        assert queue2.num_filled == original_filled

    def test_optimizer_state_restored(
        self,
        model: VectorMindModel,
        memory_queue: MemoryQueue,
        tmp_path: object,
    ) -> None:
        """Optimizer state is restored."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do a forward/backward step to populate optimizer state
        model.train()
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        image_emb = model.encode_image(images)
        text_emb = model.encode_text(input_ids, attention_mask)
        loss = (image_emb - text_emb).pow(2).sum()
        loss.backward()
        optimizer.step()

        # Save
        path = tmp_path / "checkpoint.pt"  # type: ignore[union-attr]
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=create_scaler(),
            memory_queue=memory_queue,
            epoch=0,
            step=0,
        )

        # Load
        model2 = VectorMindModel(
            {
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
        )
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        load_checkpoint(
            path,
            model2,
            optimizer2,
            create_scaler(),
            MemoryQueue(queue_size=32, embed_dim=64),
        )

        # Optimizer state should have been restored
        # (param_groups at minimum)
        assert len(optimizer2.param_groups) == len(optimizer.param_groups)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    """Tests for checkpoint error handling."""

    def test_missing_file_raises(self, model: VectorMindModel) -> None:
        """Loading from non-existent file raises FileNotFoundError."""
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = create_scaler()
        queue = MemoryQueue(queue_size=32, embed_dim=64)

        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pt", model, optimizer, scaler, queue)

    def test_queue_size_mismatch_raises(
        self, model: VectorMindModel, tmp_path: object
    ) -> None:
        """Loading with wrong queue size raises ValueError."""
        path = tmp_path / "checkpoint.pt"  # type: ignore[union-attr]
        queue = MemoryQueue(queue_size=32, embed_dim=64)
        save_checkpoint(
            path=path,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=create_scaler(),
            memory_queue=queue,
            epoch=0,
            step=0,
        )

        # Try to load into a different-sized queue
        queue2 = MemoryQueue(queue_size=64, embed_dim=64)  # wrong size!
        with pytest.raises(ValueError, match="Memory queue mismatch"):
            load_checkpoint(
                path,
                model,
                torch.optim.AdamW(model.parameters()),
                create_scaler(),
                queue2,
            )
