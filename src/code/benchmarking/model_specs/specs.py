"""Thin model specifications for benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


TaskType = Literal["pure", "mixture"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: TaskType
    implementation: str
    config_path: Path
    adapter_backend: Path | None = None
    pure_algorithm: str | None = None
    supports_reference_modes: tuple[str, ...] = ("DRIAMS_MARISMA_PURES",)
    required_config_keys: tuple[str, ...] = ()
    ablation_config_path: Path | None = None
    cli_options: dict[str, str] = field(default_factory=dict)


def active_model_parameters(config: dict[str, Any], variant: str = "default") -> dict[str, Any]:
    block = config.get("default", config) if variant == "default" else (config.get("variants") or {}).get(variant)
    if not isinstance(block, dict):
        raise ValueError(f"Missing model configuration variant {variant!r}.")
    if "model_parameters" in block:
        return dict(block["model_parameters"] or {})
    if "params" in block:
        return dict(block["params"] or {})
    return dict(block)

