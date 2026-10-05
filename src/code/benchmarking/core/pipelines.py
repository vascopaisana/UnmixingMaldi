"""Canonical final benchmark preprocessing pipeline sets."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path


FULL_PIPELINE = "FULL_PIPELINE"
STEP_ABLATION = "STEP_ABLATION"
FIXED_BINNING = "b9da_avg"
BASELINES = ("snip", "tophat")
TRANSFORMS = ("sqrt", "log2")
NORMALIZATION = "tic"
STEPS = ("S", "BC", "VS", "N")
PIPELINE_PREFIX = "pipeline_grid"


def pipeline_id(order: tuple[str, ...], baseline: str, transform: str) -> str:
    return "{}__{}__{}__{}__{}__{}".format(
        PIPELINE_PREFIX,
        "-".join(order),
        baseline,
        transform,
        FIXED_BINNING,
        NORMALIZATION,
    )


def full_pipeline_ids() -> list[str]:
    return [
        pipeline_id(order, baseline, transform)
        for order in itertools.permutations(STEPS)
        for baseline in BASELINES
        for transform in TRANSFORMS
    ]


def step_ablation_ids() -> list[str]:
    ids: list[str] = []
    for removed in STEPS:
        remaining = tuple(step for step in STEPS if step != removed)
        baselines = BASELINES if "BC" in remaining else ("none",)
        transforms = TRANSFORMS if "VS" in remaining else ("none",)
        for order in itertools.permutations(remaining):
            for baseline in baselines:
                for transform in transforms:
                    ids.append(pipeline_id(order, baseline, transform))
    return ids


def pipeline_ids(experiment: str) -> list[str]:
    if experiment == "full_pipeline":
        ids = full_pipeline_ids()
        expected = 96
    elif experiment == "step_ablation":
        ids = step_ablation_ids()
        expected = 72
    else:
        raise ValueError(f"Unsupported pipeline experiment {experiment!r}.")
    unique = list(dict.fromkeys(ids))
    if len(unique) != expected:
        raise RuntimeError(f"{experiment} generated {len(unique)} pipelines; expected {expected}.")
    return unique


def write_pipeline_manifest(path: Path, ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pipeline_id"])
        writer.writeheader()
        for pipeline in ids:
            writer.writerow({"pipeline_id": pipeline})
    return path


def apply_shard(ids: list[str], shard: dict[str, int] | None) -> list[str]:
    if not shard:
        return ids
    index = int(shard["index"])
    total = int(shard["total"])
    n = len(ids)
    start = (n * (index - 1)) // total
    end = (n * index) // total
    return ids[start:end]
