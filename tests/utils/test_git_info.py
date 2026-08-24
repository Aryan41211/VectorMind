"""Tests for :mod:`vectormind.utils.git_info`.

The point of this module is that it never raises: a checkpoint has to be
written whether or not git can answer. Most of these tests therefore
check the failure paths, which is where the value is.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from vectormind.utils import git_info


class _Completed:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


Handler = Callable[..., _Completed]


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, handler: Handler
) -> list[list[str]]:
    """Replace subprocess.run and record the argument lists it receives."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> _Completed:
        calls.append(list(args))
        return handler(args, **kwargs)

    monkeypatch.setattr(git_info.subprocess, "run", fake_run)
    return calls


class TestCurrentCommit:
    """The SHA recorded beside every checkpoint."""

    def test_returns_the_sha(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sha = "a" * 40
        _patch_run(monkeypatch, lambda args, **kw: _Completed(0, sha + "\n"))
        assert git_info.current_commit() == sha

    def test_returns_none_when_git_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_missing(args: list[str], **kwargs: object) -> _Completed:
            raise FileNotFoundError("git")

        _patch_run(monkeypatch, raise_missing)
        assert git_info.current_commit() is None

    def test_returns_none_outside_a_repository(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_run(
            monkeypatch,
            lambda args, **kw: _Completed(128, "", "not a git repository"),
        )
        assert git_info.current_commit() is None

    def test_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_timeout(args: list[str], **kwargs: object) -> _Completed:
            raise subprocess.TimeoutExpired(cmd="git", timeout=1.0)

        _patch_run(monkeypatch, raise_timeout)
        assert git_info.current_commit() is None

    def test_passes_the_repo_root_as_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}

        def capture(args: list[str], **kwargs: object) -> _Completed:
            seen.update(kwargs)
            return _Completed(0, "b" * 40)

        _patch_run(monkeypatch, capture)
        git_info.current_commit(tmp_path)
        assert seen["cwd"] == str(tmp_path)


class TestIsDirty:
    """Whether the recorded SHA actually describes the trained code."""

    def test_clean_tree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Empty output from `status --porcelain` is a real answer, not a
        # failure to ask — the two were conflated in the first cut.
        _patch_run(monkeypatch, lambda args, **kw: _Completed(0, ""))
        assert git_info.is_dirty() is False

    def test_dirty_tree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(args: list[str], **kwargs: object) -> _Completed:
            if args[1] == "status":
                return _Completed(0, " M src/vectormind/training/train_loop.py")
            return _Completed(0, "c" * 40)

        _patch_run(monkeypatch, handler)
        assert git_info.is_dirty() is True

    def test_unknown_when_git_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_run(monkeypatch, lambda args, **kw: _Completed(128, "", "nope"))
        assert git_info.is_dirty() is None


class TestDescribe:
    """The dict that lands in the checkpoint's metadata block."""

    def test_reports_commit_branch_and_dirtiness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(args: list[str], **kwargs: object) -> _Completed:
            if args[1:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
                return _Completed(0, "main")
            if args[1] == "status":
                return _Completed(0, "")
            return _Completed(0, "d" * 40)

        _patch_run(monkeypatch, handler)
        assert git_info.describe() == {
            "commit": "d" * 40,
            "branch": "main",
            "dirty": False,
        }

    def test_detached_head_reports_no_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # rev-parse --abbrev-ref prints "HEAD" when detached, which is
        # not a branch name and should not be recorded as one.
        def handler(args: list[str], **kwargs: object) -> _Completed:
            if args[1:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
                return _Completed(0, "HEAD")
            if args[1] == "status":
                return _Completed(0, "")
            return _Completed(0, "e" * 40)

        _patch_run(monkeypatch, handler)
        assert git_info.describe()["branch"] is None

    def test_never_raises_when_git_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_missing(args: list[str], **kwargs: object) -> _Completed:
            raise OSError("no git here")

        _patch_run(monkeypatch, raise_missing)
        assert git_info.describe() == {
            "commit": None,
            "branch": None,
            "dirty": None,
        }
