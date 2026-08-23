"""Survive transient CUDA out-of-memory errors during training.

Purpose: keep a long run alive when a single allocation fails for
reasons that have nothing to do with the model.

Why this exists. This project trains on an RTX 4050 laptop GPU, and that
GPU also drives the display. Its 6GB is shared with the desktop
compositor, the browser, and the editor, none of which ask permission
before allocating. A run that had reached epoch 5 of 20 died at step 879
with ``CUDA error: out of memory`` while peak training use was ~4.6GB —
the model had not grown, another process had simply taken the headroom
at the wrong moment.

Losing hours of training to a neighbouring process is an infrastructure
failure, not a modelling one, and it should not be fatal. A genuine
capacity problem still is: if the same step OOMs repeatedly after the
cache has been released, the batch really does not fit and the run
should stop with a message that says so.

Input:
  - A callable performing one training step
Output:
  - The step's return value, or a raised OutOfMemoryStepError

Dependencies: torch only.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

# Attempts per step, including the first. Two retries covers a
# neighbouring process spiking and releasing; more would just delay a
# genuine capacity failure.
DEFAULT_MAX_ATTEMPTS = 3

# Seconds to wait before retrying. Freeing our cache is instantaneous,
# but whatever else took the memory needs a moment to release it.
DEFAULT_BACKOFF_SECONDS = 2.0


class OutOfMemoryStepError(RuntimeError):
    """A training step ran out of memory on every attempt.

    Distinguishes a real capacity problem from a transient one: this is
    raised only after the cache has been released and the step retried,
    so it means the batch genuinely does not fit.
    """


# Substrings that identify an allocation failure, device or host.
#
# Torch and cuDNN report these inconsistently — as
# torch.cuda.OutOfMemoryError, as AcceleratorError, or as a bare
# RuntimeError — so matching on the message is the only reliable common
# denominator across versions.
#
# Host failures are included deliberately. This project's documented
# constraint is 6GB of VRAM, but on a 16GB laptop shared with a desktop
# the binding constraint is often system RAM: a run died at epoch 7 with
# CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED, which is cuDNN
# failing to allocate *host* memory, with 4.5GB free. Both are transient
# pressure from a neighbouring process and both deserve a retry.
_ALLOCATION_FAILURE_MARKERS: tuple[str, ...] = (
    "out of memory",
    "host_allocation_failed",
    "cudnn_status_alloc_failed",
    "cudnn_status_not_initialized",
    "not enough memory",
    "cannot allocate memory",
    "cuda_error_out_of_memory",
)


def is_out_of_memory(error: BaseException) -> bool:
    """Return whether an exception is a memory-allocation failure.

    Covers device and host allocation failures alike — see
    :data:`_ALLOCATION_FAILURE_MARKERS` for why both are treated the
    same.

    Args:
        error: The exception to classify.

    Returns:
        True if this represents an allocation failure worth retrying.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError | MemoryError):
        return True
    message = str(error).lower()
    return any(marker in message for marker in _ALLOCATION_FAILURE_MARKERS)


def release_cuda_memory() -> None:
    """Return cached allocator blocks to the driver.

    Torch keeps freed blocks in its own pool rather than handing them
    back, so a neighbouring process cannot reuse them and neither can a
    differently-shaped allocation of ours. ``gc.collect()`` runs first
    because tensors still referenced by a traceback are not free yet.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_step_with_oom_retry[T](
    step: Callable[[], T],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    context: str = "training step",
) -> T:
    """Run one training step, retrying transient CUDA OOM failures.

    On failure the allocator cache is released and the step retried after
    a short pause. Non-OOM exceptions propagate immediately — they are
    bugs, and swallowing them would hide the thing worth fixing.

    Args:
        step: Callable performing the step. Must be safe to call again:
            it should not have mutated optimizer state before the point
            where it can raise. A forward/backward pass qualifies, since
            gradients are overwritten on the retry.
        max_attempts: Total attempts, including the first.
        backoff_seconds: Pause before each retry.
        context: Description used in log messages.

    Returns:
        Whatever ``step`` returns.

    Raises:
        OutOfMemoryStepError: If every attempt ran out of memory.
        Exception: Any non-OOM exception raised by ``step``, unchanged.

    Assumptions:
        The caller has not yet applied an optimizer step for this batch.
        Retrying after ``optimizer.step()`` would apply the update twice.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}.")

    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return step()
        except Exception as error:  # noqa: BLE001 - re-raised below
            if not is_out_of_memory(error):
                raise
            last_error = error
            reserved_gb = (
                torch.cuda.memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            logger.warning(
                "CUDA OOM on %s (attempt %d/%d, %.2fGB reserved). "
                "Releasing cache and retrying.",
                context,
                attempt,
                max_attempts,
                reserved_gb,
            )
            release_cuda_memory()
            if attempt < max_attempts:
                time.sleep(backoff_seconds)

    raise OutOfMemoryStepError(
        f"{context} failed to allocate memory on all {max_attempts} "
        f"attempts. Lower dataset.batch_size in configs/data.yaml (raise "
        f"gradient_accumulation_steps in configs/training.yaml to keep "
        f"the effective batch size). If the message mentions HOST "
        f"allocation the shortage is system RAM, not VRAM — lower "
        f"dataset.num_workers and dataset.prefetch_factor instead, since "
        f"pinned batches are page-locked and cannot be swapped."
    ) from last_error
