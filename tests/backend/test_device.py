"""Tests for backend/app.py device resolution (GPU serving path)."""

from __future__ import annotations

import pytest
import torch

from backend.app import resolve_device


class TestResolveDevice:
    """Config-string to torch.device resolution."""

    def test_auto_picks_cpu_when_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """auto falls back to CPU when CUDA is unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto").type == "cpu"

    def test_auto_picks_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """auto uses CUDA when present."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto").type == "cuda"

    def test_auto_default_is_cpu_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting the setting behaves like auto."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device().type == "cpu"

    def test_forced_cpu_ignores_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cpu wins even when CUDA exists."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("cpu").type == "cpu"

    def test_forced_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cuda resolves when CUDA is present."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("cuda").type == "cuda"

    def test_forced_cuda_without_cuda_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cuda on a GPU-less host must fail loudly, not silently run CPU."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="no CUDA device is available"):
            resolve_device("cuda")

    def test_unknown_setting_raises(self) -> None:
        """Anything other than auto/cpu/cuda is rejected."""
        with pytest.raises(ValueError, match="Unknown server.device"):
            resolve_device("tpu")

    def test_case_and_whitespace_handling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The setting is trimmed and case-folded."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("  CPU  ").type == "cpu"
