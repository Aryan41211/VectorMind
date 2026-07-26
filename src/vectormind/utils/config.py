"""Configuration loading utilities.

Purpose: provide a single, consistent way to load YAML configs so that
no module ever hardcodes hyperparameters, paths, or settings directly
(CLAUDE.md §6). All scripts and modules should obtain their settings
through `load_config`, not by reading files ad hoc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary.

    Args:
        config_path: Path to a `.yaml` config file, e.g.
            ``configs/profiling.yaml``.

    Returns:
        The parsed configuration as a dictionary. Nested keys remain
        nested dictionaries/lists as written in the YAML file.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        yaml.YAMLError: If the file is not valid YAML.

    Assumptions:
        The config file is UTF-8 encoded and contains a single YAML
        document (a mapping at the top level).

    Limitations:
        This function performs no schema validation. Callers are
        responsible for checking that required keys are present and
        of the expected type before use.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. Check that the path is "
            f"correct and relative to the repository root, or an "
            f"absolute path."
        )

    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info("Loaded config from %s", path)
    return config


def require_keys(config: dict[str, Any], required_keys: list[str]) -> None:
    """Assert that all `required_keys` are present in `config`.

    Args:
        config: A config dictionary, typically from `load_config`.
        required_keys: Top-level keys that must be present.

    Raises:
        KeyError: If any key in `required_keys` is missing from `config`,
            naming all missing keys at once (not just the first).

    Assumptions:
        Only checks top-level keys; does not validate nested structure.

    Limitations:
        Does not validate value types — only presence of the key.
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(
            f"Config is missing required key(s): {missing}. "
            f"Present keys: {list(config.keys())}"
        )
