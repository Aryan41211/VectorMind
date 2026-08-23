"""Unit tests for vectormind.training.memory_queue."""

from __future__ import annotations

import pytest
import torch

from vectormind.training.memory_queue import MemoryQueue

# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestMemoryQueueInit:
    """Tests for memory queue initialization."""

    def test_stores_dimensions(self) -> None:
        """queue_size and embed_dim are stored as attributes."""
        q = MemoryQueue(queue_size=100, embed_dim=32)
        assert q.queue_size == 100
        assert q.embed_dim == 32

    def test_initial_size_is_zero(self) -> None:
        """Queue starts empty."""
        q = MemoryQueue(queue_size=100, embed_dim=32)
        assert q.current_size == 0
        assert len(q) == 0
        assert not q.is_full

    def test_queue_tensor_shape(self) -> None:
        """Queue tensor has correct shape."""
        q = MemoryQueue(queue_size=50, embed_dim=16)
        assert q.queue.shape == (50, 16)

    def test_queue_tensor_on_correct_device(self) -> None:
        """Queue tensor is on the specified device."""
        q = MemoryQueue(queue_size=10, embed_dim=8, device=torch.device("cpu"))
        assert q.queue.device == torch.device("cpu")

    def test_zero_queue_size_raises(self) -> None:
        """Zero queue_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            MemoryQueue(queue_size=0, embed_dim=32)

    def test_negative_queue_size_raises(self) -> None:
        """Negative queue_size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            MemoryQueue(queue_size=-1, embed_dim=32)

    def test_zero_embed_dim_raises(self) -> None:
        """Zero embed_dim raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            MemoryQueue(queue_size=100, embed_dim=0)


# ---------------------------------------------------------------------------
# Enqueue tests
# ---------------------------------------------------------------------------


class TestMemoryQueueEnqueue:
    """Tests for enqueue behavior."""

    def test_enqueue_increases_size(self) -> None:
        """Enqueuing embeddings increases current_size."""
        q = MemoryQueue(queue_size=100, embed_dim=16)
        emb = torch.randn(10, 16)
        q.enqueue(emb)
        assert q.current_size == 10

    def test_enqueue_multiple_batches(self) -> None:
        """Multiple enqueue calls accumulate correctly."""
        q = MemoryQueue(queue_size=100, embed_dim=16)
        q.enqueue(torch.randn(5, 16))
        q.enqueue(torch.randn(3, 16))
        assert q.current_size == 8

    def test_enqueue_fills_to_capacity(self) -> None:
        """Queue fills up to queue_size."""
        q = MemoryQueue(queue_size=20, embed_dim=8)
        q.enqueue(torch.randn(15, 8))
        assert q.current_size == 15
        q.enqueue(torch.randn(10, 8))
        assert q.current_size == 20
        assert q.is_full

    def test_enqueue_evicts_oldest(self) -> None:
        """When full, enqueue evicts oldest entries (FIFO)."""
        q = MemoryQueue(queue_size=4, embed_dim=2)

        # Fill queue with identifiable values
        q.enqueue(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        assert q.current_size == 2
        emb1 = q.get_embeddings()
        assert torch.allclose(emb1[0], torch.tensor([1.0, 1.0]))
        assert torch.allclose(emb1[1], torch.tensor([2.0, 2.0]))

        # Enqueue more to reach capacity
        q.enqueue(torch.tensor([[3.0, 3.0], [4.0, 4.0]]))
        assert q.current_size == 4
        emb2 = q.get_embeddings()
        assert torch.allclose(emb2[0], torch.tensor([1.0, 1.0]))
        assert torch.allclose(emb2[3], torch.tensor([4.0, 4.0]))

        # Enqueue one more — should evict oldest (1.0, 1.0)
        q.enqueue(torch.tensor([[5.0, 5.0]]))
        assert q.current_size == 4
        emb3 = q.get_embeddings()
        # Oldest entry [1.0, 1.0] was evicted; remaining entries are
        # [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0] (in storage order)
        all_entries = set()
        for i in range(4):
            all_entries.add(tuple(emb3[i].tolist()))
        assert (1.0, 1.0) not in all_entries  # evicted
        assert (2.0, 2.0) in all_entries
        assert (3.0, 3.0) in all_entries
        assert (4.0, 4.0) in all_entries
        assert (5.0, 5.0) in all_entries

    def test_enqueue_batch_larger_than_queue(self) -> None:
        """Batch larger than queue_size keeps only last queue_size entries."""
        q = MemoryQueue(queue_size=4, embed_dim=2)
        big_batch = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
            ]
        )
        q.enqueue(big_batch)
        assert q.current_size == 4
        emb = q.get_embeddings()
        # Should contain the last 4 entries
        assert torch.allclose(emb[0], torch.tensor([3.0, 3.0]))
        assert torch.allclose(emb[3], torch.tensor([6.0, 6.0]))

    def test_enqueue_wrong_dim_raises(self) -> None:
        """Enqueuing wrong dimension raises ValueError."""
        q = MemoryQueue(queue_size=10, embed_dim=8)
        with pytest.raises(ValueError, match="dim mismatch"):
            q.enqueue(torch.randn(5, 16))

    def test_enqueue_1d_tensor_raises(self) -> None:
        """Enqueuing 1D tensor raises ValueError."""
        q = MemoryQueue(queue_size=10, embed_dim=8)
        with pytest.raises(ValueError, match="2D"):
            q.enqueue(torch.randn(8))

    def test_enqueue_detaches_gradients(self) -> None:
        """Enqueued tensors are detached (no requires_grad)."""
        q = MemoryQueue(queue_size=10, embed_dim=8)
        emb = torch.randn(5, 8, requires_grad=True)
        q.enqueue(emb)
        stored = q.get_embeddings()
        assert not stored.requires_grad


# ---------------------------------------------------------------------------
# Get embeddings tests
# ---------------------------------------------------------------------------


class TestMemoryQueueGetEmbeddings:
    """Tests for get_embeddings behavior."""

    def test_empty_queue_returns_empty(self) -> None:
        """Empty queue returns tensor of shape [0, embed_dim]."""
        q = MemoryQueue(queue_size=10, embed_dim=8)
        emb = q.get_embeddings()
        assert emb.shape == (0, 8)

    def test_partial_queue_returns_correct_size(self) -> None:
        """Partial queue returns exactly current_size entries."""
        q = MemoryQueue(queue_size=100, embed_dim=16)
        q.enqueue(torch.randn(7, 16))
        emb = q.get_embeddings()
        assert emb.shape == (7, 16)

    def test_full_queue_returns_queue_size(self) -> None:
        """Full queue returns queue_size entries."""
        q = MemoryQueue(queue_size=20, embed_dim=8)
        q.enqueue(torch.randn(30, 8))  # more than queue_size
        emb = q.get_embeddings()
        assert emb.shape == (20, 8)

    def test_get_embeddings_is_detached(self) -> None:
        """Returned tensor is detached (no gradient)."""
        q = MemoryQueue(queue_size=10, embed_dim=8)
        q.enqueue(torch.randn(5, 8))
        emb = q.get_embeddings()
        assert not emb.requires_grad

    def test_get_embeddings_is_clone(self) -> None:
        """Returned tensor is a clone (modifying it doesn't affect queue)."""
        q = MemoryQueue(queue_size=10, embed_dim=8)
        q.enqueue(torch.randn(5, 8))
        emb = q.get_embeddings()
        emb.fill_(0)  # modify the returned tensor
        stored = q.get_embeddings()
        assert stored.abs().sum() > 0  # queue unchanged


# ---------------------------------------------------------------------------
# Wrap-around tests
# ---------------------------------------------------------------------------


class TestMemoryQueueWrapAround:
    """Tests for circular buffer wrap-around behavior."""

    def test_wrap_around_preserves_entries(self) -> None:
        """Wrap-around doesn't corrupt existing entries."""
        q = MemoryQueue(queue_size=6, embed_dim=2)

        # Fill partially
        q.enqueue(torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        assert q.pointer == 3

        # Enqueue more that wraps around
        q.enqueue(torch.tensor([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]))
        assert q.pointer == 0  # wrapped back to start
        assert q.current_size == 6

        emb = q.get_embeddings()
        assert emb.shape == (6, 2)

    def test_wrap_around_with_partial_fill(self) -> None:
        """Wrap-around works when queue is partially filled."""
        q = MemoryQueue(queue_size=4, embed_dim=2)

        # Fill to position 3
        q.enqueue(torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        assert q.pointer == 3

        # Enqueue 2 more — wraps around
        q.enqueue(torch.tensor([[4.0, 4.0], [5.0, 5.0]]))
        assert q.pointer == 1
        assert q.current_size == 4

        emb = q.get_embeddings()
        # Should contain all 5 values but only 4 slots, so oldest (1.0) evicted
        all_entries = set()
        for i in range(4):
            all_entries.add(tuple(emb[i].tolist()))
        assert (1.0, 1.0) not in all_entries  # evicted
        assert (2.0, 2.0) in all_entries
        assert (3.0, 3.0) in all_entries
        assert (4.0, 4.0) in all_entries
        assert (5.0, 5.0) in all_entries


# ---------------------------------------------------------------------------
# Size cap tests
# ---------------------------------------------------------------------------


class TestMemoryQueueSizeCap:
    """Tests that queue never exceeds queue_size."""

    def test_never_exceeds_capacity(self) -> None:
        """Queue size never exceeds queue_size regardless of enqueue calls."""
        q = MemoryQueue(queue_size=10, embed_dim=4)
        for _ in range(100):
            q.enqueue(torch.randn(5, 4))
        assert q.current_size <= q.queue_size
        assert q.current_size == 10

    def test_is_full_property(self) -> None:
        """is_full is True only when queue is at capacity."""
        q = MemoryQueue(queue_size=5, embed_dim=4)
        assert not q.is_full
        q.enqueue(torch.randn(3, 4))
        assert not q.is_full
        q.enqueue(torch.randn(3, 4))
        assert q.is_full

    def test_large_enqueue_respects_cap(self) -> None:
        """Single large enqueue respects queue_size."""
        q = MemoryQueue(queue_size=8, embed_dim=4)
        q.enqueue(torch.randn(100, 4))
        assert q.current_size == 8
        assert q.is_full


class TestWarmup:
    """The queue must fill while inactive, then serve a full queue.

    Without a momentum encoder, activating the queue from step 1 swamps
    the gradient with stale negatives — measured as val R@10 stuck at
    chance (0.35%) after two epochs. See MemoryQueue's class docstring.
    """

    DIM = 16

    def _queue(self, active: bool, size: int = 32) -> MemoryQueue:
        return MemoryQueue(queue_size=size, embed_dim=self.DIM, active=active)

    def test_defaults_to_active(self) -> None:
        assert self._queue(active=True).active is True
        assert MemoryQueue(queue_size=8, embed_dim=self.DIM).active is True

    def test_inactive_queue_serves_no_negatives(self) -> None:
        q = self._queue(active=False)
        q.enqueue(torch.randn(8, self.DIM))
        assert q.get_embeddings().shape == (0, self.DIM)

    def test_inactive_queue_still_fills(self) -> None:
        """The whole point of warmup: full the moment it switches on."""
        q = self._queue(active=False)
        q.enqueue(torch.randn(8, self.DIM))
        assert q.current_size == 8
        assert q.get_embeddings().shape[0] == 0

    def test_activation_exposes_what_was_buffered(self) -> None:
        q = self._queue(active=False)
        q.enqueue(torch.randn(8, self.DIM))
        q.enqueue(torch.randn(8, self.DIM))
        assert q.get_embeddings().shape[0] == 0
        q.activate()
        assert q.active is True
        assert q.get_embeddings().shape == (16, self.DIM)

    def test_activation_is_idempotent(self) -> None:
        q = self._queue(active=False)
        q.enqueue(torch.randn(4, self.DIM))
        q.activate()
        q.activate()
        assert q.get_embeddings().shape[0] == 4

    def test_deactivate_hides_contents_without_dropping_them(self) -> None:
        q = self._queue(active=True)
        q.enqueue(torch.randn(4, self.DIM))
        q.deactivate()
        assert q.get_embeddings().shape[0] == 0
        assert q.current_size == 4
        q.activate()
        assert q.get_embeddings().shape[0] == 4

    def test_empty_active_queue_returns_empty(self) -> None:
        q = self._queue(active=True)
        assert q.get_embeddings().shape == (0, self.DIM)

    def test_warmup_can_fill_the_queue_completely(self) -> None:
        q = self._queue(active=False, size=32)
        for _ in range(8):
            q.enqueue(torch.randn(8, self.DIM))
        q.activate()
        assert q.is_full
        assert q.get_embeddings().shape == (32, self.DIM)
