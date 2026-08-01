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
    epoch, step, config)
  - load_checkpoint(path, model, optimizer, scaler, memory_queue)
    → (epoch, step)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.memory_queue import MemoryQueue

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

    # Metadata for traceability (ARCHITECTURE.md §12)
    metadata = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": epoch,
        "step": step,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    if config is not None:
        metadata["config_hash"] = str(hash(json.dumps(config, sort_keys=True)))

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
) -> tuple[int, int]:
    """Load complete training state from a checkpoint file.

    Args:
        path: File path of the checkpoint to load.
        model: The VectorMindModel to restore.
        optimizer: The optimizer to restore.
        scaler: The GradScaler to restore.
        memory_queue: The memory queue to restore.

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
