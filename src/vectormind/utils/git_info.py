"""Read the working tree's git state, for checkpoint traceability.

Purpose: ARCHITECTURE.md §12 promises that any checkpoint can be traced
back to the exact code that produced it. A config hash covers the
configuration; this covers the code. Without it, "which commit trained
this?" is answered by matching timestamps against the log, which stops
working the moment two runs share an afternoon.

Every function here fails soft. A checkpoint must still be written when
git is absent, when the tree was exported rather than cloned, or when
the command times out — an untraceable checkpoint is a much smaller
problem than a training run that dies at its first save.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: Seconds to wait for git. Generous for a local rev-parse, short enough
#: that a hung git (a stale index.lock, a network-mounted tree) cannot
#: stall a training run.
GIT_TIMEOUT_SECONDS: float = 5.0


def _run_git(args: list[str], repo_root: Path | None) -> str | None:
    """Run a git command and return its stdout, or None on any failure.

    Args:
        args: Arguments after ``git``.
        repo_root: Directory to run in. Defaults to the process's
            working directory.

    Returns:
        Stripped stdout — possibly the empty string, which is a real
        answer for ``status --porcelain`` — or None if git is missing,
        errored, or timed out. None means "could not ask", never "asked
        and got nothing".
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root) if repo_root is not None else None,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s unavailable: %s", " ".join(args), exc)
        return None

    if completed.returncode != 0:
        logger.debug(
            "git %s failed (%d): %s",
            " ".join(args),
            completed.returncode,
            completed.stderr.strip(),
        )
        return None

    return completed.stdout.strip()


def current_commit(repo_root: Path | None = None) -> str | None:
    """Return the full SHA of HEAD.

    Args:
        repo_root: Directory inside the repository. Defaults to the
            process's working directory.

    Returns:
        The 40-character commit SHA, or None if it cannot be determined.
    """
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    return commit or None


def is_dirty(repo_root: Path | None = None) -> bool | None:
    """Report whether the working tree has uncommitted changes.

    Why it is recorded next to the commit: a SHA alone implies the code
    is recoverable, and a checkpoint trained from a dirty tree is not.
    Saying so is the difference between traceability and the appearance
    of it.

    Args:
        repo_root: Directory inside the repository.

    Returns:
        True if tracked files differ from HEAD, False if the tree is
        clean, None if git could not answer.
    """
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], repo_root)
    if status is None:
        return None
    return bool(status)


def describe(repo_root: Path | None = None) -> dict[str, str | bool | None]:
    """Summarize git state for a metadata block.

    Args:
        repo_root: Directory inside the repository.

    Returns:
        Mapping with ``commit`` (SHA or None), ``branch`` (name, or
        None when detached or unavailable) and ``dirty``.
    """
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    return {
        "commit": current_commit(repo_root),
        # A detached HEAD prints "HEAD", which is not a branch name.
        "branch": branch or None if branch != "HEAD" else None,
        "dirty": is_dirty(repo_root),
    }
