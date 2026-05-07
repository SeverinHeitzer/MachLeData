"""Configuration loading and environment variable expansion helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Global constant for the project root to locate configs relative to the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and expand environment variables.

    Args:
        path: Path to the YAML file, relative to PROJECT_ROOT or absolute.

    Returns:
        Dictionary containing the configuration.
    """
    full_path = Path(path)
    if not full_path.is_absolute():
        full_path = PROJECT_ROOT / path

    if not full_path.exists():
        return {}

    with open(full_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}

    if not isinstance(content, dict):
        raise ValueError(f"Expected a mapping in {path}")

    return expand_env_vars(content)


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in a configuration structure."""
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, str):
        # Handles both $VAR and ${VAR}
        return os.path.expandvars(value)
    return value


def get_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
