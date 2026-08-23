"""Training metrics logging via TensorBoard.

Purpose: log per-step and per-epoch training metrics (loss,
temperature, embedding norms, GPU memory) to TensorBoard for
visualization and monitoring.

Design decisions:
- TensorBoard chosen over W&B for this phase: no account setup
  required, works fully offline, and is the standard PyTorch
  logging backend. W&B is noted as an upgrade path in
  FUTURE_IDEAS.md.
- Per-step logging (not just per-epoch) as required by CLAUDE.md §5:
  "Training metrics (loss, temperature, embedding norms, gradient
  norms) go to Weights & Biases (or TensorBoard), not just console
  output."
- Simple wrapper around ``torch.utils.tensorboard.SummaryWriter``
  with a consistent API for all training metrics.

Input:
  - log_metrics(step, metrics_dict)
  - log_epoch(epoch, metrics_dict)
  - close()

Output:
  - TensorBoard event files in the specified log directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class TrainingLogger:
    """TensorBoard-based training metrics logger.

    Wraps ``torch.utils.tensorboard.SummaryWriter`` with a simple
    interface for logging per-step and per-epoch metrics.

    Attributes:
        log_dir: Directory where TensorBoard event files are written.
        writer: The underlying SummaryWriter instance.
    """

    def __init__(self, log_dir: str | Path) -> None:
        """Initialize the training logger.

        Args:
            log_dir: Directory for TensorBoard event files.
                Created if it doesn't exist.

        Raises:
            ImportError: If ``tensorboard`` is not installed.

        Assumptions:
            ``tensorboard`` is installed (listed in requirements.txt).
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.error(
                "TensorBoard is required for training logging. "
                "Install it with: pip install tensorboard"
            )
            raise

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        logger.info("TrainingLogger initialized: log_dir=%s", self.log_dir)

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log a dict of metrics at the given step.

        Each metric is logged as a separate scalar to TensorBoard.

        Args:
            step: Global training step.
            metrics: Dictionary of metric name to float value.
        """
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch-level summary metrics.

        Prefixes metric names with ``"epoch/"`` to distinguish from
        per-step metrics.

        Args:
            epoch: Epoch number.
            metrics: Dictionary of epoch-level metric name to value.
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f"epoch/{name}", value, epoch)

    def log_histogram(
        self, step: int, name: str, values: torch.Tensor
    ) -> None:
        """Log a histogram of values (e.g. embedding norms).

        Args:
            step: Global training step.
            name: Name for the histogram.
            values: Tensor of values to histogram.
        """
        self.writer.add_histogram(name, values, step)

    def flush(self) -> None:
        """Flush pending writes to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the writer and release resources."""
        self.writer.close()
        logger.info("TrainingLogger closed: log_dir=%s", self.log_dir)
