"""Local environment loading and config value expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parents[1]
LOCAL_ENV_PATH = REPO_ROOT / ".env.local"


def load_local_env(path: Path = LOCAL_ENV_PATH) -> None:
    """Load simple KEY=VALUE entries from the ignored local environment file."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip("\"'"))


def expand_config_values(value: Any) -> Any:
    """Recursively expand environment variables and `~` in config strings."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_config_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_config_values(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_config_values(child) for key, child in value.items()}
    return value


load_local_env()
