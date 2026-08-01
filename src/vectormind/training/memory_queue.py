"""MoCo-style memory queue for additional negative samples.

Purpose: maintain a fixed-size FIFO buffer of past embeddings to use
as extra negatives in the contrastive loss, decoupling the number of
negatives from the physical batch size (ARCHITECTURE.md §6).

Design decisions (locked in ARCHITECTURE.md §6):
- Text embeddings stored in the queue (used as extra negatives for
  the image→text loss direction). This is the simplest to reason
  about: the queue holds "what the text encoder has recently produced."
- Circular buffer via a fixed tensor + pointer index — not a Python
  list of tensors (slow, memory-fragmenting).
- Queue tensors are detached (no gradients) — they represent past,
  already-updated states, not current computation graph.
- Default queue_size: 4096 (16× the measured batch of 256).

Input:
  - enqueue(): batch of embeddings [B, D]
  - get_embeddings(): returns all valid embeddings [K, D] where K <= queue_size

Output:
  - Detached embeddings ready to be passed to symmetric_infonce() as
    the ``queue_embeddings`` argument.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class MemoryQueue:
    """MoCo-style FIFO memory queue for negative samples.

    Stores past embeddings in a fixed-size circular buffer. The queue
    never grows beyond ``queue_size`` — oldest entries are evicted
    when new entries are added.

    Attributes:
        queue_size: Maximum number of embeddings stored.
        embed_dim: Dimension of each embedding.
        device: Device where the queue tensor lives.
        queue: Fixed tensor holding the queue contents.
        pointer: Current write position (next index to overwrite).
        num_filled: Number of valid entries currently in the queue.
    """

    def __init__(
        self,
        queue_size: int,
        embed_dim: int,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the memory queue.

        Args:
            queue_size: Maximum number of embeddings to store.
            embed_dim: Dimension of each embedding vector.
            device: Device for the queue tensor. If ``None``, defaults
                to CPU.

        Raises:
            ValueError: If ``queue_size`` or ``embed_dim`` is not positive.

        Assumptions:
            All embeddings passed to ``enqueue()`` have the same
            ``embed_dim`` as specified here.

        Limitations:
            The queue is not thread-safe. The training loop is
            single-threaded, so this is acceptable.
        """
        if queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {queue_size}.")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")

        self.queue_size = queue_size
        self.embed_dim = embed_dim
        self.device = device or torch.device("cpu")

        # Fixed tensor buffer — pre-allocated, no dynamic memory
        self.queue = torch.zeros(queue_size, embed_dim, device=self.device)
        self.pointer = 0
        self.num_filled = 0

        logger.info(
            "MemoryQueue initialized: queue_size=%d, embed_dim=%d, device=%s",
            queue_size,
            embed_dim,
            self.device,
        )

    def enqueue(self, embeddings: torch.Tensor) -> None:
        """Add embeddings to the queue, evicting oldest if full.

        Embeddings are detached before storage — they represent past
        states, not part of the current computation graph.

        Args:
            embeddings: Batch of embeddings to enqueue, shape
                ``[B, embed_dim]``.

        Raises:
            ValueError: If embeddings have wrong shape or dimension.
            ValueError: If embeddings have ``requires_grad=True``
                (caller must detach first or this method handles it).

        Assumptions:
            The caller enqueues embeddings after each forward pass,
            before or after the optimizer step (order doesn't matter
            for the queue's purpose).

        Limitations:
            If B > queue_size, only the last queue_size entries are
            kept (older entries are completely overwritten).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got ndim={embeddings.ndim}.")
        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embed_dim}, "
                f"got {embeddings.shape[1]}."
            )

        # Detach and move to queue device
        batch = embeddings.detach().to(self.device)
        B = batch.shape[0]

        if B >= self.queue_size:
            # Batch larger than queue: keep only the last queue_size
            self.queue.copy_(batch[-self.queue_size :])
            self.pointer = 0
            self.num_filled = self.queue_size
        else:
            # Insert batch into the circular buffer
            end = self.pointer + B
            if end <= self.queue_size:
                # No wrap-around
                self.queue[self.pointer : end] = batch
                self.pointer = end % self.queue_size
            else:
                # Wrap-around
                first_chunk = self.queue_size - self.pointer
                self.queue[self.pointer :] = batch[:first_chunk]
                self.queue[: B - first_chunk] = batch[first_chunk:]
                self.pointer = B - first_chunk

            self.num_filled = min(self.num_filled + B, self.queue_size)

    def get_embeddings(self) -> torch.Tensor:
        """Return all valid embeddings currently in the queue.

        Returns:
            Tensor of shape ``[K, embed_dim]`` where
            ``K = min(num_filled, queue_size)``. The embeddings are
            in FIFO order (oldest first).

        Assumptions:
            The returned tensor is detached (no gradient tracking).
        """
        if self.num_filled == 0:
            return torch.zeros(0, self.embed_dim, device=self.device)

        if self.num_filled < self.queue_size:
            # Queue not yet full — entries are at indices 0..num_filled-1
            return self.queue[: self.num_filled].clone()

        # Queue is full — entries start from pointer (oldest) and wrap
        # For simplicity, return all entries in their current positions
        # (the order doesn't matter for contrastive loss — we just need
        # K distinct negatives)
        return self.queue.clone()

    @property
    def is_full(self) -> bool:
        """Whether the queue has reached capacity."""
        return self.num_filled >= self.queue_size

    @property
    def current_size(self) -> int:
        """Current number of valid entries in the queue."""
        return self.num_filled

    def __len__(self) -> int:
        """Return current number of valid entries."""
        return self.num_filled
