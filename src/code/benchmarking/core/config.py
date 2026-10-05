"""Configuration loading for the benchmark runner."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal shells.
    yaml = None


BENCHMARKING_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BENCHMARKING_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[1]
DEFAULT_CONFIG = BENCHMARKING_DIR / "benchmark.yaml"
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"
os.environ.setdefault("UNMIXMALDI_ROOT", str(PROJECT_ROOT))


def load_local_env(path: Path = LOCAL_ENV_PATH) -> None:
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
    os.environ.setdefault("UNMIXMALDI_ROOT", str(PROJECT_ROOT))


def expand_config_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_config_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_config_values(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_config_values(child) for key, child in value.items()}
    return value


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_at_indent: dict[int, tuple[Any, str]] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            item_text = stripped[2:].strip()
            if not isinstance(parent, list):
                container, key = last_key_at_indent[indent - 2]
                new_list: list[Any] = []
                container[key] = new_list
                stack.append((indent - 2, new_list))
                parent = new_list
            if ":" in item_text:
                item: dict[str, Any] = {}
                key, _, value = item_text.partition(":")
                key = key.strip()
                if value.strip():
                    item[key] = _scalar(value)
                else:
                    child: dict[str, Any] = {}
                    item[key] = child
                    stack.append((indent, item))
                    stack.append((indent + 2, child))
                    parent.append(item)
                    continue
                parent.append(item)
                stack.append((indent, item))
                last_key_at_indent[indent + 2] = (item, key)
                continue
            value = _scalar(item_text)
            parent.append(value)
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        if not isinstance(parent, dict):
            raise TypeError(f"Cannot assign YAML key {key!r} under a list.")
        if value.strip():
            parent[key] = _scalar(value)
            last_key_at_indent[indent] = (parent, key)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            last_key_at_indent[indent] = (parent, key)
            stack.append((indent, child))
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    load_local_env()
    with Path(path).open("r") as handle:
        if yaml is not None:
            data = yaml.safe_load(handle) or {}
        else:
            data = _simple_yaml_load(handle.read())
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return expand_config_values(data)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        if yaml is not None:
            yaml.safe_dump(data, handle, sort_keys=False)
        else:
            json.dump(data, handle, indent=2, default=str)
            handle.write("\n")


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def load_config_block(path: Path, variant: str = "default") -> tuple[dict[str, Any], dict[str, Any]]:
    data = load_yaml(path)
    block = data.get("default", data) if variant == "default" else (data.get("variants") or {}).get(variant)
    if not isinstance(block, dict):
        raise ValueError(f"Config {path} does not define variant {variant!r}.")
    return data, block


def coerce_path_args(
    args: argparse.Namespace,
    path_fields: tuple[str, ...],
    path_list_fields: tuple[str, ...] = (),
) -> None:
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))
    for field in path_list_fields:
        values = getattr(args, field, None)
        if values is not None:
            setattr(args, field, [value if isinstance(value, Path) else Path(value) for value in values])


def load_model_parameters(model_name: str, defaults: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    config = deepcopy(defaults)
    if path is None:
        return config
    config_path = Path(path)
    if not config_path.exists():
        return config
    data = load_yaml(config_path)
    if "model_parameters" in data and isinstance(data["model_parameters"], dict):
        config.update(data["model_parameters"])
        return config
    model_block = data.get(str(model_name).lower(), {})
    if isinstance(model_block, dict):
        params = model_block.get("params", model_block)
        if isinstance(params, dict):
            config.update(params)
    return config
