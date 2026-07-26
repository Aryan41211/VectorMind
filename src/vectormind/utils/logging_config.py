"""Logging setup for the VectorMind project.

Purpose: provide one consistent logging configuration so that every
script and module logs through Python's `logging` module (CLAUDE.md
§5) instead of bare `print()` calls, with a uniform format and
optional file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure the root logger for console (and optionally file) output.

    Args:
        log_file: If provided, logs are also written to this file path
            (parent directories are created if missing). If None,
            logs go to stdout only.
        level: Logging level for the root logger (e.g. `logging.INFO`,
            `logging.DEBUG`).

    Returns:
        None. Configures the root logger in place; subsequent calls to
        `logging.getLogger(__name__)` anywhere in the project will use
        this configuration.

    Assumptions:
        Called once, near process startup (e.g. at the top of a script's
        `main()`), before any other module logs.

    Limitations:
        Calling this more than once in the same process will add
        duplicate handlers unless the root logger's handlers are
        cleared first; this function clears existing handlers to
        avoid that, but is not designed for use in long-lived
        multi-entrypoint processes (e.g. a server that reconfigures
        logging per-request).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called more than once.
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
