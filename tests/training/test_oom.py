"""Tests for transient CUDA OOM recovery.

These run on CPU: the point is the retry logic, not real allocation
failures, so OOM is simulated. The behaviour being pinned is that a
transient failure is survivable and a genuine one is not silently
swallowed.
"""

from __future__ import annotations

import pytest
import torch

from vectormind.training.oom import (
    OutOfMemoryStepError,
    is_out_of_memory,
    release_cuda_memory,
    run_step_with_oom_retry,
)


def oom_error() -> RuntimeError:
    """A RuntimeError shaped like torch's OOM report."""
    return RuntimeError(
        "CUDA error: out of memory. Tried to allocate 512.00 MiB."
    )


class TestIsOutOfMemory:
    def test_detects_torch_oom_type(self) -> None:
        assert is_out_of_memory(torch.cuda.OutOfMemoryError("CUDA out of memory"))

    def test_detects_runtime_error_by_message(self) -> None:
        """Torch reports OOM as a plain RuntimeError in several paths."""
        assert is_out_of_memory(oom_error())

    def test_detects_accelerator_error_wording(self) -> None:
        # The wording that killed the epoch-5 run.
        assert is_out_of_memory(RuntimeError("CUDA error: out of memory"))

    def test_is_case_insensitive(self) -> None:
        assert is_out_of_memory(RuntimeError("CUDA Error: Out Of Memory"))

    def test_rejects_unrelated_errors(self) -> None:
        assert not is_out_of_memory(ValueError("shape mismatch"))
        assert not is_out_of_memory(RuntimeError("device-side assert triggered"))


class TestRetry:
    def test_returns_the_value_when_the_step_succeeds(self) -> None:
        assert run_step_with_oom_retry(lambda: 42) == 42

    def test_does_not_retry_a_successful_step(self) -> None:
        calls = {"n": 0}

        def step() -> int:
            calls["n"] += 1
            return 1

        run_step_with_oom_retry(step)
        assert calls["n"] == 1

    def test_recovers_from_a_transient_oom(self) -> None:
        """The epoch-5 scenario: one bad allocation, then fine."""
        calls = {"n": 0}

        def step() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise oom_error()
            return "recovered"

        result = run_step_with_oom_retry(step, backoff_seconds=0)
        assert result == "recovered"
        assert calls["n"] == 2

    def test_gives_up_after_max_attempts(self) -> None:
        calls = {"n": 0}

        def step() -> None:
            calls["n"] += 1
            raise oom_error()

        with pytest.raises(OutOfMemoryStepError):
            run_step_with_oom_retry(step, max_attempts=3, backoff_seconds=0)
        assert calls["n"] == 3

    def test_failure_message_says_how_to_fix_it(self) -> None:
        """A real capacity failure must point at the config knobs."""
        with pytest.raises(OutOfMemoryStepError, match="batch_size"):
            run_step_with_oom_retry(
                lambda: (_ for _ in ()).throw(oom_error()),
                max_attempts=1,
                backoff_seconds=0,
            )

    def test_preserves_the_original_error_as_cause(self) -> None:
        original = oom_error()

        def step() -> None:
            raise original

        with pytest.raises(OutOfMemoryStepError) as info:
            run_step_with_oom_retry(step, max_attempts=1, backoff_seconds=0)
        assert info.value.__cause__ is original

    def test_does_not_retry_or_mask_other_errors(self) -> None:
        """A shape bug must surface immediately, not be retried three times."""
        calls = {"n": 0}

        def step() -> None:
            calls["n"] += 1
            raise ValueError("embeddings must be 2D")

        with pytest.raises(ValueError, match="2D"):
            run_step_with_oom_retry(step, backoff_seconds=0)
        assert calls["n"] == 1

    def test_rejects_nonsense_attempt_count(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            run_step_with_oom_retry(lambda: 1, max_attempts=0)

    def test_logs_a_warning_per_oom_attempt(self, caplog) -> None:
        calls = {"n": 0}

        def step() -> int:
            calls["n"] += 1
            if calls["n"] < 3:
                raise oom_error()
            return 0

        with caplog.at_level("WARNING"):
            run_step_with_oom_retry(step, backoff_seconds=0)
        assert sum("CUDA OOM" in r.message for r in caplog.records) == 2


class TestReleaseCudaMemory:
    def test_is_safe_without_cuda(self) -> None:
        """Called from CPU-only test runs and CI, where there is no device."""
        release_cuda_memory()


class TestHostAllocationFailures:
    """Host RAM failures must retry too.

    The project's stated constraint is 6GB of VRAM, but on a 16GB laptop
    shared with a desktop the binding constraint is often system memory.
    A run died at epoch 7 with a cuDNN *host* allocation failure while
    the GPU was fine and 4.5GB of RAM was free — the dataloader's pinned
    buffers are page-locked and cannot be swapped out.
    """

    def test_detects_cudnn_host_allocation_failure(self) -> None:
        # The exact message that killed the epoch-7 run.
        assert is_out_of_memory(
            RuntimeError(
                "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED"
            )
        )

    def test_detects_cpu_allocator_exhaustion(self) -> None:
        assert is_out_of_memory(
            RuntimeError("DefaultCPUAllocator: not enough memory: you tried...")
        )

    def test_detects_plain_memory_error(self) -> None:
        assert is_out_of_memory(MemoryError())

    def test_detects_cudnn_alloc_failed(self) -> None:
        assert is_out_of_memory(RuntimeError("CUDNN_STATUS_ALLOC_FAILED"))

    def test_still_rejects_genuine_cuda_bugs(self) -> None:
        """Broadening detection must not swallow real errors."""
        assert not is_out_of_memory(
            RuntimeError("CUDA error: device-side assert triggered")
        )
        assert not is_out_of_memory(RuntimeError("CUDNN_STATUS_BAD_PARAM"))

    def test_recovers_from_a_host_allocation_failure(self) -> None:
        calls = {"n": 0}

        def step() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED"
                )
            return "recovered"

        assert run_step_with_oom_retry(step, backoff_seconds=0) == "recovered"

    def test_failure_message_distinguishes_host_from_device(self) -> None:
        """A RAM shortage needs different knobs than a VRAM shortage."""
        with pytest.raises(OutOfMemoryStepError, match="num_workers"):
            run_step_with_oom_retry(
                lambda: (_ for _ in ()).throw(MemoryError()),
                max_attempts=1,
                backoff_seconds=0,
            )


class TestRecoveryPathCannotCrash:
    """The recovery path must never be the thing that kills a run.

    empty_cache() and synchronize() can raise when the CUDA context is
    already in a bad state. An exception escaping from there turns a
    retryable OOM into a hard crash and masks the original error — which
    is exactly how a run died at epoch 13, one step after the retry had
    correctly fired.
    """

    def test_survives_empty_cache_raising(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "empty_cache",
            lambda: (_ for _ in ()).throw(RuntimeError("CUDA error: out of memory")),
        )
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        release_cuda_memory()  # must not raise

    def test_survives_synchronize_raising(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(
            torch.cuda,
            "synchronize",
            lambda: (_ for _ in ()).throw(RuntimeError("context is corrupt")),
        )
        release_cuda_memory()  # must not raise

    def test_retry_still_completes_when_cleanup_fails(self, monkeypatch) -> None:
        """The end-to-end property: cleanup failure must not stop a retry."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "empty_cache",
            lambda: (_ for _ in ()).throw(RuntimeError("cannot free")),
        )
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

        calls = {"n": 0}

        def step() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise oom_error()
            return "recovered"

        assert run_step_with_oom_retry(step, backoff_seconds=0) == "recovered"
        assert calls["n"] == 2

    def test_logs_the_cleanup_failure(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "empty_cache",
            lambda: (_ for _ in ()).throw(RuntimeError("cannot free")),
        )
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        with caplog.at_level("WARNING"):
            release_cuda_memory()
        assert any("empty_cache" in r.message for r in caplog.records)
