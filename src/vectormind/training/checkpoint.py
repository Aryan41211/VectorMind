"""Checkpoint save/load for exact training state restoration.

Purpose: save and load the complete training state — model weights,
optimizer state, GradScaler state, memory queue contents, epoch,
and step counters — so training can be exactly resumed from any
checkpoint.

Design decisions:
- Single file per checkpoint (model + optimizer + scaler + queue +
  metadata), not separate files. Simpler to manage, and at this
  project's scale (queue_size=4096 * embed_dim=256 * 4 bytes ≈ 4MB
  for the queue alone), the file size is not a concern.
- Memory queue state is saved/loaded including the circular buffer
  tensor and pointer — not reconstructed from scratch on resume.
- Metadata includes config hash, git commit SHA, and timestamp for
  traceability (ARCHITECTURE.md §12).

Input:
  - save_checkpoint(path, model, optimizer, scaler, memory_queue,
    epoch, step, config=..., metrics=..., scheduler=...)
  - load_checkpoint(path, model, optimizer, scaler, memory_queue,
    scheduler=...)  → (epoch, step)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.memory_queue import MemoryQueue
from vectormind.utils.git_info import describe

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: VectorMindModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    memory_queue: MemoryQueue,
    epoch: int,
    step: int,
    config: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save complete training state to a checkpoint file.

    Args:
        path: File path for the checkpoint.
        model: The VectorMindModel to save.
        optimizer: The optimizer to save.
        scaler: The GradScaler to save.
        memory_queue: The memory queue to save.
        epoch: Current epoch number.
        step: Current global step number.
        config: Optional configuration dict to include in metadata.
        metrics: Optional validation metrics achieved by this
            checkpoint. Stored so a resumed run can recover the
            best-so-far score instead of restarting the comparison from
            zero and overwriting a better checkpoint with a worse one.
        scheduler: Optional LR scheduler to save. Without it, a resumed
            run rebuilds the scheduler from scratch and re-runs the
            schedule from its peak LR instead of continuing it
            (see :mod:`load_checkpoint`).

    Raises:
        OSError: If the file cannot be written.
        RuntimeError: If the model is on a device that can't be
            serialized.

    Assumptions:
        The caller ensures the path's parent directory exists.

    Limitations:
            Does not save the training data or DataLoader state.
            The DataLoader is stateless (shuffled each epoch), so
            resuming mid-epoch may produce slightly different
            batching.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "queue": {
            "tensor": memory_queue.queue.clone(),
            "pointer": memory_queue.pointer,
            "num_filled": memory_queue.num_filled,
            "queue_size": memory_queue.queue_size,
            "embed_dim": memory_queue.embed_dim,
        },
    }

    # The scheduler's position in the training budget is part of the
    # training state, not a preference: a resumed run that does not know
    # it re-ran the cosine decay from the top. Cf. load_checkpoint.
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Metadata for traceability (ARCHITECTURE.md §12). The git fields
    # are what make "which code produced this checkpoint?" answerable
    # later; without them the answer is inferred from timestamps, which
    # stops working as soon as two runs share an afternoon — and this
    # project has had six resumed runs in two days.
    metadata: dict[str, Any] = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": epoch,
        "step": step,
        "num_params": sum(p.numel() for p in model.parameters()),
        "git": describe(),
    }
    if metrics is not None:
        metadata["metrics"] = dict(metrics)
    if config is not None:
        metadata["config_hash"] = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]

    checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)

    # Log file size
    file_size_mb = path.stat().st_size / (1024**2)
    logger.info(
        "Checkpoint saved: %s (epoch=%d, step=%d, %.2f MB)",
        path,
        epoch,
        step,
        file_size_mb,
    )


def load_checkpoint(
    path: str | Path,
    model: VectorMindModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    memory_queue: MemoryQueue,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, int]:
    """Load complete training state from a checkpoint file.

    Args:
        path: File path of the checkpoint to load.
        model: The VectorMindModel to restore.
        optimizer: The optimizer to restore.
        scaler: The GradScaler to restore.
        memory_queue: The memory queue to restore.
        scheduler: Optional LR scheduler to restore. A checkpoint with a
            saved ``scheduler_state_dict`` resumes the schedule at the
            epoch it paused on; a checkpoint without one (written before
            this feature) leaves the scheduler where it is and logs that
            the schedule restarts, so a silent restart of the LR curve
            cannot hide again.

    Returns:
        Tuple of (epoch, step) restored from the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        KeyError: If required checkpoint keys are missing.
        RuntimeError: If state_dict shapes don't match the model.

    Assumptions:
        The model, optimizer, scaler, and memory queue have the same
        architecture/config as when the checkpoint was saved.

    Limitations:
        Does not restore the DataLoader state (see save_checkpoint
        docstring).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False is required because optimizer state dicts contain
    # non-tensor objects (parameter group metadata). Checkpoint files are
    # locally produced, not from untrusted sources.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Restore model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scaler
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore memory queue
    queue_state = checkpoint["queue"]
    if (
        queue_state["queue_size"] != memory_queue.queue_size
        or queue_state["embed_dim"] != memory_queue.embed_dim
    ):
        raise ValueError(
            f"Memory queue mismatch: checkpoint has "
            f"queue_size={queue_state['queue_size']}, "
            f"embed_dim={queue_state['embed_dim']}, but current queue "
            f"has queue_size={memory_queue.queue_size}, "
            f"embed_dim={memory_queue.embed_dim}."
        )

    memory_queue.queue.copy_(queue_state["tensor"])
    memory_queue.pointer = queue_state["pointer"]
    memory_queue.num_filled = queue_state["num_filled"]

    # Restore the LR scheduler when both halves exist.
    #
    # Pre-feature checkpoints simply have no scheduler state, and a fresh
    # CosineAnnealingLR then restarts at its peak LR — the exact bug this
    # feature exists to head off. So a restore that cannot happen is
    # logged, not silently swallowed.
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            logger.info(
                "Scheduler restored: last_epoch=%d", scheduler.last_epoch
            )
        else:
            logger.warning(
                "Checkpoint %s holds no scheduler state; the LR schedule "
                "restarts from its initial value",
                path,
            )

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]

    metadata = checkpoint.get("metadata", {})
    logger.info(
        "Checkpoint loaded: %s (epoch=%d, step=%d, timestamp=%s)",
        path,
        epoch,
        step,
        metadata.get("timestamp_iso", "unknown"),
    )

    return epoch, step


def read_checkpoint_metric(
    path: str | Path,
    key: str,
    default: float = 0.0,
) -> float:
    """Read one recorded validation metric from a checkpoint.

    Used on resume to recover the best-so-far score. Without it,
    ``best_val_recall10`` restarts at zero and the first epoch after a
    resume overwrites ``best_model.pt`` whatever its score — which is
    how a checkpoint at 17.46% R@10 was replaced by one at 10.51%.

    Args:
        path: Checkpoint file. A missing file is not an error.
        key: Metric name, e.g. ``"recall@10"``.
        default: Returned when the file, the metadata, or the key is
            absent.

    Returns:
        The recorded value, or ``default``.

    Assumptions:
        Reads with ``weights_only=False`` because the metadata is a
        plain dict rather than tensors. Only load checkpoints you
        produced.
    """
    path = Path(path)
    if not path.exists():
        return default

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - a corrupt file must not
        # abort a training run that is otherwise ready to start.
        logger.warning("Could not read %s: %s", path, error)
        return default

    metrics = checkpoint.get("metadata", {}).get("metrics")
    if not isinstance(metrics, dict) or key not in metrics:
        logger.warning(
            "%s records no '%s'; best-so-far starts at %.4f. A checkpoint "
            "saved before metrics were recorded may be overwritten.",
            path,
            key,
            default,
        )
        return default

    return float(metrics[key])
