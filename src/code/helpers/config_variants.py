"""Helpers for YAML configs with optional named variants."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from helpers.local_env import expand_config_values, load_local_env


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    load_local_env()
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return expand_config_values(data)


def load_config_with_variant(config_path: Path, variant: str | None = None) -> dict[str, Any]:
    """Load a plain YAML or resolve a grouped YAML with base/variants."""
    config = load_yaml(config_path)
    variants = config.get("variants")
    if variants is None:
        if variant is not None:
            raise ValueError(f"{config_path} does not define variants, but --variant={variant!r} was provided")
        return config

    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"{config_path} has an empty or invalid variants mapping")
    if variant is None:
        variant = config.get("default_variant")
    if variant is None:
        available = ", ".join(sorted(variants))
        raise ValueError(f"{config_path} requires --variant. Available variants: {available}")
    if variant not in variants:
        available = ", ".join(sorted(variants))
        raise ValueError(f"Unknown variant {variant!r} for {config_path}. Available variants: {available}")

    resolved = deep_merge(config.get("base", {}) or {}, variants[variant] or {})
    resolved["_config_variant"] = {
        "config_path": str(config_path),
        "variant": variant,
        "description": config.get("description"),
    }
    return resolved
