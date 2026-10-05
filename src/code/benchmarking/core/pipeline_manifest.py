#!/usr/bin/env python3
"""Preprocessing-grid helpers for benchmark adapters."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "data/mixtures/pipelines_study").exists():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / "data/mixtures/pipelines_study").exists() and (candidate / "outputs").exists():
            return candidate
    return start.parents[3]


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_MATRIX = None


def _slug(*parts: str) -> str:
    return "__".join(str(part).replace("/", "-") for part in parts)


def iter_combinations(matrix: dict[str, Any], include_optional_binning: bool = False):
    preprocessing = matrix["preprocessing"]
    binning = list(preprocessing["binning"].keys())

    keys = [
        "order",
        "variance_stabilization",
        "peak_selection",
        "binning",
        "normalization",
    ]
    values = [
        list(preprocessing["orders"].keys()),
        list(preprocessing["variance_stabilization"].keys()),
        list(preprocessing["peak_selection"].keys()),
        binning,
        list(preprocessing["normalization"].keys()),
    ]
    for parts in itertools.product(*values):
        combo = dict(zip(keys, parts))
        combo["pipeline_id"] = _slug(
            combo["variance_stabilization"],
            combo["peak_selection"],
            combo["binning"],
            combo["normalization"],
        )
        combo["pipeline_slug"] = _slug(combo["order"], combo["pipeline_id"])
        yield combo
