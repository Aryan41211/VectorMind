"""Unit tests for vectormind.training.trainer.

The trainer is the single shared epoch loop that replaced the four
Phase 4 scripts' duplicated loops (scripts/train.py, resume_training.py,
benchmark_epoch.py, hyperparameter_experiment.py). These tests pin the
loop's observable contract: scheduler stepping, checkpoint cadence,
early stopping, memory-queue warmup, resume, and the LR logging fix
(per-step ``train/lr`` must come from the scheduler, not the static
initial LR).

Note: these tests run on CPU with AMP disabled (autocast is a no-op on
CPU), matching the existing train_loop tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.checkpoint import read_checkpoint_metric
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.train_loop import create_optimizer, create_scaler
from vectormind.training.trainer import train

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLoader:
    """Minimal DataLoader stand-in: fixed-length, yields fixed batches.

    Re-iterable: each ``__iter__`` restarts from the original batches,
    so every training epoch sees the same steps.
    """

    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        """Store the original batches for replay on every iteration."""
        self._initial = list(batches)
        self._remaining: list[dict[str, torch.Tensor]] = []
        self._length = len(batches)

    def __iter__(self) -> FakeLoader:
        """Restart iteration from the original batches."""
        self._remaining = list(self._initial)
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        """Yield the next batch, or raise StopIteration when exhausted."""
        if not self._remaining:
            raise StopIteration
        return self._remaining.pop(0)

    def __len__(self) -> int:
        """Return the fixed number of batches per epoch."""
        return self._length


class FakeEvaluator:
    """Callable stand-in for the evaluation function.

    Returns results in FIFO order, so a test can script a run where the
    model improves, then regresses, without running a real forward pass.
    """

    def __init__(self, results: list[dict[str, float]]) -> None:
        """Store the scripted result sequence."""
        self.results = list(results)
        self.calls = 0

    def __call__(
        self,
        model: VectorMindModel,
        dataloader: object,
        device: torch.device,
        captions_per_image: int,
    ) -> dict[str, float]:
        """Return the next scripted result, keeping the last one warm."""
        del model, dataloader, device, captions_per_image
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]

    @staticmethod
    def metrics(recall10: float) -> dict[str, float]:
        """A flat metric dict with the keys the loop reads from eval."""
        return {
            "recall@1": recall10 * 2,
            "recall@5": recall10 * 1.5,
            "recall@10": recall10,
            "image_dim_variance": 0.8,
            "text_dim_variance": 0.7,
            "separation": recall10,
            "matched_similarity": 0.5,
            "unmatched_similarity": 0.4,
            "image_mean_cosine": 0.1,
            "image_mean_norm": 0.2,
            "collapsed": False,
        }


class FakeLogger:
    """Duck-typed TrainingLogger stand-in that records everything."""

    def __init__(self) -> None:
        """Create an empty log buffer."""
        self.step_logs: list[tuple[int, dict[str, float]]] = []
        self.epoch_logs: list[tuple[int, dict[str, float]]] = []

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        self.step_logs.append((step, dict(metrics)))

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        self.epoch_logs.append((epoch, dict(metrics)))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_config() -> dict[str, Any]:
    """Smaller model config for faster unit tests."""
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
def train_config(tmp_path: Path) -> dict[str, Any]:
    """Training config the loop reads, with every knob explicit."""
    return {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "tb"),
        "epochs": 3,
        "eval_every_n_epochs": 1,
        "save_every_n_epochs": 3,
        "gradient_accumulation_steps": 1,
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        "scheduler": {"T_max": 3, "eta_min": 1e-4},
        "memory_queue": {
            "enabled": False,
            "warmup_epochs": 0,
            "queue_size": 16,
        },
        "early_stopping": {"enabled": False, "patience": 5, "min_delta": 0.001},
        "uniformity": {"weight": 0.0},
        "temperature": {"clamp_enabled": False, "max_logit_scale": 40.0},
    }


@pytest.fixture
def model(small_model_config: dict[str, Any]) -> VectorMindModel:
    """Small VectorMindModel for testing."""
    return VectorMindModel(small_model_config)


@pytest.fixture
def batches() -> list[dict[str, torch.Tensor]]:
    """Three small positive-margin batches so training stays stable."""
    batches: list[dict[str, torch.Tensor]] = []
    vocab = 1000
    for _ in range(3):
        batches.append(
            {
                "image": torch.randn(2, 3, 64, 64),
                "input_ids": torch.randint(0, vocab, (2, 32)),
                "attention_mask": torch.ones(2, 32, dtype=torch.long),
            }
        )
    return batches


class Harness:
    """Everything a trainer call needs, wired for a synchronous run."""

    def __init__(
        self,
        model: VectorMindModel,
        train_config: dict[str, Any],
        batches: list[dict[str, torch.Tensor]],
        evaluator: FakeEvaluator,
        queue_active: bool,
    ) -> None:
        """Wire all components the trainer needs for a CPU run."""
        self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = create_optimizer(
            self.model, lr=1e-3, weight_decay=0.0
        )
        self.scaler = create_scaler()
        t_max = int(train_config["scheduler"]["T_max"])
        eta_min = float(train_config["scheduler"]["eta_min"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=eta_min
        )
        mq = train_config["memory_queue"]
        self.memory_queue = MemoryQueue(
            queue_size=int(mq["queue_size"]),
            embed_dim=int(model_config_shared_dim(model)),
            device=self.device,
            active=queue_active,
        )
        self.train_loader = FakeLoader(batches)
        self.val_loader = FakeLoader(batches)
        self.logger = FakeLogger()
        self.evaluator = evaluator


def model_config_shared_dim(model: VectorMindModel) -> int:
    """Recover the shared embedding dim from the live model."""
    for name, module in model.named_children():
        if "projection" in name or "embedding" in name:
            for p in module.parameters():
                return p.shape[0]
    return next(model.parameters()).shape[0]


@pytest.fixture
def harness(
    model: VectorMindModel,
    train_config: dict[str, Any],
    batches: list[dict[str, torch.Tensor]],
) -> Harness:
    """Wired harness with an evaluator that never improves."""
    fake = FakeEvaluator([FakeEvaluator.metrics(0.05)])
    return Harness(model, train_config, batches, fake, queue_active=False)


def run_train(
    h: Harness,
    train_config: dict[str, Any],
    num_epochs: int,
    start_epoch: int = 0,
    global_step: int = 0,
    best_val_recall10: float = 0.0,
) -> dict[str, Any]:
    """Call the trainer with the harness components, returning its result."""
    return train(
        model=h.model,
        optimizer=h.optimizer,
        scaler=h.scaler,
        scheduler=h.scheduler,
        memory_queue=h.memory_queue,
        train_loader=h.train_loader,
        val_loader=h.val_loader,
        device=h.device,
        training_logger=h.logger,
        train_config=train_config,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        global_step=global_step,
        best_val_recall10=best_val_recall10,
        log_every_steps=1,
        evaluate=h.evaluator,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scheduler_steps_once_per_epoch(
    harness: Harness, train_config: dict[str, Any]
) -> None:
    """The scheduler advances exactly once per epoch."""
    run_train(harness, train_config, num_epochs=3)
    assert harness.scheduler.last_epoch == 3


def test_logged_lr_tracks_scheduler_not_static_initial(
    harness: Harness, train_config: dict[str, Any]
) -> None:
    """Per-step ``train/lr`` must track the scheduler, not stay static.

    Regression for the LR-logging issue: TensorBoard previously showed a
    flat curve because the loop logged the initial LR every step.
    """
    run_train(harness, train_config, num_epochs=3)

    steps_per_epoch = len(harness.train_loader)
    by_epoch: list[list[float]] = [[] for _ in range(3)]
    for step, metrics in harness.logger.step_logs:
        epoch = step // steps_per_epoch
        by_epoch[epoch].append(metrics["train/lr"])

    assert by_epoch[0], "steps from epoch 0 should have been logged"
    initial_lr = float(train_config["optimizer"]["lr"])
    assert by_epoch[0][0] == pytest.approx(initial_lr)
    assert by_epoch[1][0] != pytest.approx(by_epoch[0][0])
    assert by_epoch[2][0] != pytest.approx(by_epoch[0][0])
    assert by_epoch[2][0] < by_epoch[1][0] < by_epoch[0][0]


def test_best_checkpoint_only_overwritten_by_improvement(
    small_model_config: dict[str, Any],
    train_config: dict[str, Any],
    batches: list[dict[str, torch.Tensor]],
    model: VectorMindModel,
) -> None:
    """A worse validation score must not clobber the saved best model."""
    train_config["save_every_n_epochs"] = 1
    evaluator = FakeEvaluator(
        [FakeEvaluator.metrics(0.20), FakeEvaluator.metrics(0.10)]
    )
    h = Harness(model, train_config, batches, evaluator, queue_active=False)
    run_train(h, train_config, num_epochs=2)

    best_path = Path(train_config["checkpoint_dir"]) / "best_model.pt"
    assert best_path.exists(), "best_model.pt should have been written"
    metric = read_checkpoint_metric(best_path, "recall@10")
    assert metric == pytest.approx(0.20)


def test_periodic_checkpoints_follow_save_cadence(
    harness: Harness, train_config: dict[str, Any]
) -> None:
    """Periodic checkpoints appear on the save cadence.

    At ``save_every_n_epochs`` intervals plus the final checkpoint.
    """
    train_config["save_every_n_epochs"] = 2
    ckpt_dir = Path(train_config["checkpoint_dir"])
    run_train(harness, train_config, num_epochs=3)

    assert (ckpt_dir / "epoch_002.pt").exists()
    assert not (ckpt_dir / "epoch_001.pt").exists()
    assert (ckpt_dir / "final_model.pt").exists()


def test_early_stopping_stops_before_last_epoch(
    harness: Harness, train_config: dict[str, Any]
) -> None:
    """Early stopping fires once patience epochs pass without improvement.

    The run stops before the last epoch but still saves the final
    checkpoint.
    """
    train_config["early_stopping"] = {
        "enabled": True,
        "patience": 2,
        "min_delta": 0.001,
    }
    result = run_train(
        harness,
        train_config,
        num_epochs=5,
        best_val_recall10=0.20,
    )
    ckpt_dir = Path(train_config["checkpoint_dir"])

    # The stopping epoch is evaluated and influences the final summary,
    # but its epoch block is skipped (preserved train.py behavior: the
    # early-stop break fires before log_epoch).
    assert [epoch for epoch, _ in harness.logger.epoch_logs] == [0]
    assert harness.evaluator.calls == 2
    assert result["best_val_recall10"] == pytest.approx(0.20)
    assert result["epochs_run"] == 2
    assert result["last_epoch_metrics"]["recall@10"] == pytest.approx(0.05)
    assert (ckpt_dir / "final_model.pt").exists()
    assert not (ckpt_dir / "epoch_005.pt").exists()


def test_resume_continues_from_start_epoch_and_step(
    harness: Harness, train_config: dict[str, Any]
) -> None:
    """A resumed run logs from the restored epoch/step, not from zero."""
    train_config["save_every_n_epochs"] = 1
    result = run_train(
        harness,
        train_config,
        num_epochs=3,
        start_epoch=1,
        global_step=41,
        best_val_recall10=0.20,
    )

    assert [epoch for epoch, _ in harness.logger.epoch_logs] == [1, 2]
    first_step = harness.logger.step_logs[0][0]
    assert first_step == 41
    steps_per_epoch = len(harness.train_loader)
    assert result["total_steps"] == 41 + 2 * steps_per_epoch


def test_memory_queue_activates_after_warmup(
    small_model_config: dict[str, Any],
    train_config: dict[str, Any],
    batches: list[dict[str, torch.Tensor]],
    model: VectorMindModel,
) -> None:
    """The queue starts inactive and activates once warmup clears."""
    train_config["memory_queue"]["enabled"] = True
    train_config["memory_queue"]["warmup_epochs"] = 1
    evaluator = FakeEvaluator([FakeEvaluator.metrics(0.05)])
    h = Harness(model, train_config, batches, evaluator, queue_active=False)

    run_train(h, train_config, num_epochs=3)
    assert h.memory_queue.active, "queue should have been activated by epoch 1"


def test_disabled_memory_queue_never_activates(
    small_model_config: dict[str, Any],
    train_config: dict[str, Any],
    batches: list[dict[str, torch.Tensor]],
    model: VectorMindModel,
) -> None:
    """With the queue disabled in config, warmup never flips it on."""
    train_config["memory_queue"]["enabled"] = False
    train_config["memory_queue"]["warmup_epochs"] = 1
    evaluator = FakeEvaluator([FakeEvaluator.metrics(0.05)])
    h = Harness(model, train_config, batches, evaluator, queue_active=False)

    run_train(h, train_config, num_epochs=3)
    assert not h.memory_queue.active


def test_returns_summary_dict(
    harness: Harness, train_config: dict[str, Any]
) -> None:
    """The trainer returns the run summary the CLI needs for its report."""
    result = run_train(harness, train_config, num_epochs=2)

    assert "total_steps" in result
    assert "epochs_run" in result
    assert "best_val_recall10" in result
    assert "best_val_recall1" in result
    assert "first_epoch_metrics" in result
    assert "last_epoch_metrics" in result
    assert result["total_steps"] == 6
    assert result["epochs_run"] == 2