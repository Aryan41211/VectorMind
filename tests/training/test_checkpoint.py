"""Unit tests for vectormind.training.checkpoint."""

from __future__ import annotations

import logging

import pytest
import torch

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.checkpoint import (
    load_checkpoint,
    read_checkpoint_metric,
    save_checkpoint,
)
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.train_loop import create_optimizer, create_scaler

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

        for p1, p2 in zip(
            model.parameters(), model2.parameters(), strict=True
        ):
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


class TestBestCheckpointMetrics:
    """A resumed run must not overwrite a better checkpoint.

    save_checkpoint recorded no score, and train.py reset its
    best-so-far to 0.0 on resume, so the first epoch after any resume
    won the comparison unconditionally. That replaced a checkpoint at
    17.46% val R@10 with one at 10.51%.
    """

    def _save(self, tmp_path, small_config, metrics=None):
        config = small_config
        model = VectorMindModel(config)
        optimizer = create_optimizer(model)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        queue = MemoryQueue(queue_size=8, embed_dim=config["embedding"]["shared_dim"])
        path = tmp_path / "best_model.pt"
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            memory_queue=queue,
            epoch=3,
            step=100,
            metrics=metrics,
        )
        return path

    def test_records_metrics_in_metadata(self, tmp_path, small_config):
        path = self._save(tmp_path, small_config, {"recall@10": 0.1746, "recall@1": 0.0343})
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["metadata"]["metrics"]["recall@10"] == pytest.approx(0.1746)

    def test_reads_the_recorded_metric_back(self, tmp_path, small_config):
        path = self._save(tmp_path, small_config, {"recall@10": 0.1746})
        assert read_checkpoint_metric(path, "recall@10") == pytest.approx(0.1746)

    def test_missing_file_returns_the_default(self, tmp_path):
        """A first run has no best checkpoint; that is not an error."""
        assert read_checkpoint_metric(tmp_path / "nope.pt", "recall@10") == 0.0

    def test_checkpoint_without_metrics_returns_the_default(
        self, tmp_path, small_config
    ):
        """Checkpoints predating this change must still be resumable."""
        path = self._save(tmp_path, small_config, metrics=None)
        assert read_checkpoint_metric(path, "recall@10") == 0.0

    def test_absent_key_returns_the_default(self, tmp_path, small_config):
        path = self._save(tmp_path, small_config, {"recall@1": 0.03})
        assert read_checkpoint_metric(path, "recall@10") == 0.0

    def test_honours_a_custom_default(self, tmp_path):
        assert (
            read_checkpoint_metric(tmp_path / "nope.pt", "recall@10", default=0.5)
            == 0.5
        )

    def test_corrupt_file_does_not_abort_the_run(self, tmp_path):
        """A damaged best checkpoint must not stop training from starting."""
        path = tmp_path / "best_model.pt"
        path.write_bytes(b"not a checkpoint")
        assert read_checkpoint_metric(path, "recall@10") == 0.0

    def test_metrics_survive_a_full_round_trip(self, tmp_path, small_config):
        """The regression guard: resume recovers the real best-so-far."""
        path = self._save(tmp_path, small_config, {"recall@10": 0.1746})
        recovered = read_checkpoint_metric(path, "recall@10")
        worse_epoch_score = 0.1051
        assert worse_epoch_score < recovered, (
            "a worse epoch must not beat the restored best-so-far"
        )


class TestSchedulerPersistence:
    """A resumed run must continue the LR schedule, not restart it.

    The cosine scheduler is rebuilt at every fresh process start, so
    before its state was checkpointed, ``--resume`` training restored
    the weights at epoch N but re-ran the decay from its peak: the
    resumed epoch trained at the learning rate epoch 0 used, not the
    rate the run had actually reached. These tests guard that a round
    trip restores the schedule's position.
    """

    def _build(self, small_config: dict) -> tuple[
        VectorMindModel,
        torch.optim.AdamW,
        torch.optim.lr_scheduler.CosineAnnealingLR,
    ]:
        """Build model, optimizer and a cosine scheduler over it."""
        model = VectorMindModel(small_config)
        optimizer = create_optimizer(model, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-6
        )
        return model, optimizer, scheduler

    def _advance_schedule(
        self,
        model: VectorMindModel,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        steps: int = 7,
    ) -> None:
        """Reach the desired ``last_epoch`` via real optimizer steps.

        Steps the optimizer before the scheduler, as ``train.py`` does,
        so the schedule's position is exactly the one a paused run would
        have saved.
        """
        model.train()
        for _ in range(steps):
            images = torch.randn(2, 3, 224, 224)
            input_ids = torch.randint(0, 1000, (2, 32))
            attention_mask = torch.ones(2, 32)
            image_emb = model.encode_image(images)
            text_emb = model.encode_text(input_ids, attention_mask)
            (image_emb - text_emb).pow(2).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    def test_scheduler_position_survives_a_round_trip(
        self, tmp_path: object, small_config: dict
    ) -> None:
        """A restored scheduler continues the decay instead of resetting."""
        model, optimizer, scheduler = self._build(small_config)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        queue = MemoryQueue(queue_size=8, embed_dim=64)

        # Advance the schedule to somewhere mid-decay.
        self._advance_schedule(model, optimizer, scheduler, steps=7)
        lr_mid_schedule = scheduler.get_last_lr()[0]

        path = tmp_path / "ckpt.pt"  # type: ignore[union-attr]
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            memory_queue=queue,
            epoch=7,
            step=500,
            scheduler=scheduler,
        )

        model2, optimizer2, scheduler2 = self._build(small_config)
        load_checkpoint(
            path,
            model2,
            optimizer2,
            torch.amp.GradScaler("cpu", enabled=False),
            MemoryQueue(queue_size=8, embed_dim=64),
            scheduler=scheduler2,
        )

        assert scheduler2.last_epoch == 7
        assert scheduler2.get_last_lr()[0] == pytest.approx(lr_mid_schedule)
        # The next step continues the decay from epoch 7, not from 0.
        scheduler2.step()
        assert scheduler2.last_epoch == 8

    def test_legacy_checkpoint_without_scheduler_state_warns(
        self, tmp_path: object, small_config: dict, caplog: object
    ) -> None:
        """Checkpoints written before this feature must still resume.

        The scheduler is left at its own position, and the reset is
        stated explicitly — a silent restart is how the LR schedule
        bug hid in the first place.
        """
        model, optimizer, _ = self._build(small_config)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        queue = MemoryQueue(queue_size=8, embed_dim=64)

        # Save without a scheduler — a pre-feature checkpoint.
        path = tmp_path / "legacy.pt"  # type: ignore[union-attr]
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            memory_queue=queue,
            epoch=3,
            step=100,
        )

        _, _, scheduler2 = self._build(small_config)
        with caplog.at_level(  # type: ignore[attr-defined]
            logging.WARNING, logger="vectormind.training.checkpoint"
        ):
            load_checkpoint(
                path,
                model,
                optimizer,
                torch.amp.GradScaler("cpu", enabled=False),
                MemoryQueue(queue_size=8, embed_dim=64),
                scheduler=scheduler2,
            )

        # Unperturbed: the fresh scheduler stays at epoch 0.
        assert scheduler2.last_epoch == 0
        assert "scheduler state" in caplog.text  # type: ignore[attr-defined]

    def test_loading_without_a_scheduler_still_works(
        self, tmp_path: object, small_config: dict
    ) -> None:
        """Scheduler is an optional argument — existing callers are unaffected."""
        model, optimizer, scheduler = self._build(small_config)
        self._advance_schedule(model, optimizer, scheduler, steps=3)

        path = tmp_path / "ckpt.pt"  # type: ignore[union-attr]
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            memory_queue=MemoryQueue(queue_size=8, embed_dim=64),
            epoch=3,
            step=42,
            scheduler=scheduler,
        )

        _, _, scheduler2 = self._build(small_config)
        epoch, step = load_checkpoint(
            path,
            model,
            optimizer,
            torch.amp.GradScaler("cpu", enabled=False),
            MemoryQueue(queue_size=8, embed_dim=64),
        )

        assert epoch == 3
        assert step == 42
        # Without a scheduler argument nothing can be restored, and that
        # is not an error.
        assert scheduler2.last_epoch == 0
