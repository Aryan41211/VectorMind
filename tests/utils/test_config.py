"""Unit tests for vectormind.utils.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vectormind.utils.config import load_config, require_keys


def test_load_config_reads_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(yaml.dump({"a": 1, "b": {"c": 2}}))

    config = load_config(config_path)

    assert config == {"a": 1, "b": {"c": 2}}


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(missing_path)


def test_load_config_accepts_string_path(tmp_path: Path) -> None:
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(yaml.dump({"key": "value"}))

    config = load_config(str(config_path))

    assert config == {"key": "value"}


def test_require_keys_passes_when_all_present() -> None:
    config = {"image_encoder": {}, "text_encoder": {}}

    # Should not raise.
    require_keys(config, ["image_encoder", "text_encoder"])


def test_require_keys_raises_on_missing_key() -> None:
    config = {"image_encoder": {}}

    with pytest.raises(KeyError):
        require_keys(config, ["image_encoder", "text_encoder"])


def test_require_keys_error_message_lists_all_missing_keys() -> None:
    config = {"a": 1}

    with pytest.raises(KeyError, match="b.*c|c.*b"):
        require_keys(config, ["a", "b", "c"])
