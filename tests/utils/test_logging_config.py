"""Unit tests for vectormind.utils.logging_config."""

from __future__ import annotations

import logging
from pathlib import Path

from vectormind.utils.logging_config import setup_logging


def test_setup_logging_sets_root_level() -> None:
    setup_logging(log_file=None, level=logging.DEBUG)

    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_console_only_adds_one_handler() -> None:
    setup_logging(log_file=None, level=logging.INFO)

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_setup_logging_with_file_creates_file_and_handler(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "test.log"

    setup_logging(log_file=log_file, level=logging.INFO)
    logging.getLogger(__name__).info("test message")

    assert log_file.exists()
    assert "test message" in log_file.read_text(encoding="utf-8")


def test_setup_logging_called_twice_does_not_duplicate_handlers() -> None:
    setup_logging(log_file=None, level=logging.INFO)
    setup_logging(log_file=None, level=logging.INFO)

    assert len(logging.getLogger().handlers) == 1
