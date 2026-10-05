#!/usr/bin/env python3
"""Run pure-spectrum and mixture benchmarks from a manifest-first entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from benchmarking.core.config import DEFAULT_CONFIG, load_yaml
from benchmarking.core.datasets import resolve_datasets
from benchmarking.core.outputs import parse_shard
from benchmarking.experiments.mixture.evaluation import run_mixture_model
from benchmarking.experiments.pure.evaluation import run_pure_model
from benchmarking.model_specs.registry import MODEL_REGISTRY


PURE_MODELS = ("random_forest", "svm", "logistic_regression", "knn")
MIXTURE_MODELS = ("rf_cosine", "mahe", "nmf", "yang")


@dataclass(frozen=True)
class ResolvedRun:
    task: str
    experiment: str
    config_name: str
    workers: int
    shard: dict[str, int] | None
    dry_run: bool
    sources: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CONFIG, help="Top-level benchmark execution manifest.")
    parser.add_argument("--task", choices=["pure", "pures", "mixture", "mixtures"], default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Runnable method selection, e.g. rf_cosine, mahe, all, svm, knn.",
    )
    parser.add_argument("--experiment", choices=["full_pipeline", "step_ablation"], default=None)
    parser.add_argument("--workers", type=int, default=None, help="Workers inside each shard process.")
    parser.add_argument("--shard", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", default=None)
    return parser.parse_args()


def _value(cli_value: Any, manifest_value: Any, fallback: Any, key: str, sources: dict[str, str]) -> Any:
    if cli_value is not None:
        sources[key] = "CLI override"
        return cli_value
    if manifest_value is not None:
        sources[key] = "benchmark.yaml"
        return manifest_value
    sources[key] = "framework default"
    return fallback


def _require(value: Any, key: str, task: str | None = None) -> Any:
    if value is None or value == "":
        suffix = f" for task={task!r}" if task else ""
        raise ValueError(f"Missing benchmark.{key}{suffix} in execution manifest. Provide it in benchmark.yaml or via --{key}.")
    return value


def _manifest_run(manifest: dict[str, Any]) -> dict[str, Any]:
    block = manifest.get("benchmark")
    if not isinstance(block, dict):
        raise ValueError("benchmark.yaml requires one benchmark mapping.")
    return block


def _normalize_task(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized == "pures":
        return "pure"
    if normalized == "mixtures":
        return "mixture"
    return normalized


def resolve_run(args: argparse.Namespace, manifest: dict[str, Any], benchmark_block: dict[str, Any]) -> ResolvedRun:
    runtime = manifest.get("runtime") or {}
    sources: dict[str, str] = {
        "benchmark_manifest": str(args.manifest),
    }
    task = _value(args.task, benchmark_block.get("task"), None, "task", sources)
    task = _require(task, "task")
    task = _normalize_task(task)
    experiment = _value(args.experiment, benchmark_block.get("experiment"), None, "experiment", sources)
    experiment = _require(experiment, "experiment", task)
    config_name = _value(args.config, benchmark_block.get("config"), None, "config", sources)
    config_name = _require(config_name, "config", task)
    sources["reference_mode"] = "config YAML" if task == "mixture" else "not applicable"
    manifest_workers = runtime.get("workers_per_shard", runtime.get("workers"))
    workers = int(_value(args.workers, manifest_workers, 1, "workers", sources))
    if args.shard is not None:
        sources["shard"] = "CLI override"
        shard_text = args.shard
    elif runtime.get("shard") is not None:
        sources["shard"] = "benchmark.yaml"
        shard_text = runtime.get("shard")
    else:
        sources["shard"] = "framework default"
        shard_text = None
    dry_run = bool(_value(args.dry_run, runtime.get("dry_run"), False, "dry_run", sources))
    return ResolvedRun(
        task=str(task),
        experiment=str(experiment),
        config_name=str(config_name),
        workers=workers,
        shard=parse_shard(shard_text),
        dry_run=dry_run,
        sources=sources,
    )


def _inject_runtime(config: dict[str, Any], resolved: ResolvedRun) -> dict[str, Any]:
    config.setdefault("runtime", {})["workers"] = resolved.workers
    return config


def _models_for_run(config: dict[str, Any], resolved: ResolvedRun) -> list[str]:
    if resolved.task == "mixture":
        if resolved.config_name == "all":
            return list(MIXTURE_MODELS)
        model = resolved.config_name
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown mixture config/model {model!r}.")
        if MODEL_REGISTRY[model].task != "mixture":
            raise ValueError(f"Config {model!r} is not a mixture model.")
        return [model]

    if resolved.config_name in {"all", "cross_domain"}:
        return ["all"]
    if resolved.config_name not in PURE_MODELS:
        raise ValueError(
            "Pure config must be one of: all, random_forest, svm, logistic_regression, knn."
        )
    return [resolved.config_name]


def _run_one(config: dict[str, Any], datasets: dict[str, Any], resolved: ResolvedRun) -> None:
    config = _inject_runtime(config, resolved)
    if resolved.task == "pure":
        for model in _models_for_run(config, resolved):
            run_pure_model(
                config=config,
                datasets=datasets,
                domain="pure",
                experiment=resolved.experiment,
                model=model,
                dry_run=resolved.dry_run,
                shard=resolved.shard,
                launch_context={
                    "config": resolved.config_name,
                    "sources": resolved.sources,
                },
            )
        return

    for model in _models_for_run(config, resolved):
        launch_context = {"config": resolved.config_name, "sources": resolved.sources}
        run_mixture_model(
            config=config,
            datasets=datasets,
            domain="mixture",
            experiment=resolved.experiment,
            model=model,
            reference_mode=None,
            dry_run=resolved.dry_run,
            shard=resolved.shard,
            launch_context=launch_context,
        )


def _fanout_shards(args: argparse.Namespace, manifest: dict[str, Any]) -> bool:
    runtime = manifest.get("runtime") or {}
    total_shards = int(runtime.get("shards") or 1)
    if total_shards <= 1 or args.shard is not None:
        return False
    workers_per_shard = args.workers
    if workers_per_shard is None:
        workers_per_shard = int(runtime.get("workers_per_shard", runtime.get("workers", 1)))
    dry_run = bool(args.dry_run or runtime.get("dry_run"))
    commands: list[list[str]] = []
    for shard_index in range(1, total_shards + 1):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest",
            str(args.manifest),
            "--shard",
            f"{shard_index}/{total_shards}",
            "--workers",
            str(workers_per_shard),
        ]
        if args.task is not None:
            command.extend(["--task", args.task])
        if args.experiment is not None:
            command.extend(["--experiment", args.experiment])
        if args.config is not None:
            command.extend(["--config", args.config])
        if dry_run:
            command.append("--dry-run")
        commands.append(command)

    print(
        f"Launching {total_shards} shard processes "
        f"with {workers_per_shard} worker(s) per shard.",
        flush=True,
    )
    processes = [subprocess.Popen(command) for command in commands]
    failures = [process.wait() for process in processes]
    failed = [code for code in failures if code]
    if failed:
        raise subprocess.CalledProcessError(failed[0], "benchmark shard fan-out")
    return True


def main() -> None:
    args = parse_args()
    manifest = load_yaml(args.manifest)
    if _fanout_shards(args, manifest):
        return
    datasets = resolve_datasets(manifest)
    resolved = resolve_run(args, manifest, _manifest_run(manifest))
    _run_one(manifest, datasets, resolved)


if __name__ == "__main__":
    main()
