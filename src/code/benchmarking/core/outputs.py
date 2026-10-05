"""Output directories, manifests, and command logging."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import git_commit, write_yaml


def output_dir(base: Path, domain: str, experiment: str, model: str, reference_mode: str | None = None) -> Path:
    if domain == "mixture":
        if reference_mode is None:
            raise ValueError("Mixture output paths require reference_mode.")
        return base / domain / experiment / reference_mode.lower() / model
    return base / domain / experiment / model


def temp_dir(base: Path, domain: str, experiment: str, model: str, reference_mode: str | None = None) -> Path:
    parts = [domain, experiment]
    if reference_mode:
        parts.append(reference_mode.lower())
    parts.append(model)
    return base.joinpath(*parts)


def parse_shard(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    left, sep, right = value.partition("/")
    if sep != "/":
        raise ValueError("--shard must use the form INDEX/TOTAL, for example 1/4.")
    index = int(left)
    total = int(right)
    if total < 1 or index < 1 or index > total:
        raise ValueError("--shard INDEX/TOTAL must satisfy 1 <= INDEX <= TOTAL.")
    return {"index": index, "total": total}


def manifest_base(
    *,
    domain: str,
    experiment: str,
    model: str,
    output_path: Path,
    pipeline_count: int,
    seed: int,
    workers: int,
    datasets: dict[str, Any],
    model_config_path: Path,
    model_config: dict[str, Any],
    variant: str = "default",
    reference_mode: str | None = None,
    shard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "benchmark": {
            "domain": domain,
            "experiment": experiment,
            "model": model,
            "reference_mode": reference_mode,
        },
        "execution": {
            "seed": seed,
            "workers": workers,
            "sharding_enabled": bool(shard),
            "shard": shard,
        },
        "pipeline": {
            "set": "STEP_ABLATION" if experiment == "step_ablation" else "FULL_PIPELINE",
            "count": pipeline_count,
            "binning": "b9da_avg",
        },
        "datasets": datasets,
        "model": {
            "config_path": str(model_config_path),
            "variant": variant,
        },
        "model_config": model_config,
        "output_dir": str(output_path),
    }


def print_summary(manifest: dict[str, Any], dry_run: bool = False) -> None:
    benchmark = manifest["benchmark"]
    execution = manifest["execution"]
    training = manifest["datasets"].get("training", {})
    model = manifest["model"]
    title = f"{benchmark['domain']} benchmark".upper()
    print("=" * 60)
    print(title + (" DRY RUN" if dry_run else ""))
    print("=" * 60)
    print(f"Experiment       : {benchmark['experiment'].upper()}")
    print(f"Model            : {benchmark['model']}")
    if benchmark.get("reference_mode"):
        print(f"Reference mode   : {benchmark['reference_mode']}")
    print(f"Seed             : {execution['seed']}")
    print(f"Pipelines        : {manifest['pipeline']['count']}")
    print(f"Binning          : {manifest['pipeline']['binning']}")
    print("")
    print("Training/reference:")
    print(f"  MARISMa        : {training.get('marisma')}")
    print(f"  DRIAMS-A       : {training.get('driams_a')}")
    print(f"  Total          : {training.get('total')}")
    print(f"  Species        : {training.get('species')}")
    print("")
    print("Execution:")
    print(f"  Workers        : {execution['workers']}")
    shard = execution.get("shard")
    print(f"  Sharding       : {shard['index']}/{shard['total']}" if shard else "  Sharding       : disabled")
    print("")
    print("Model:")
    print(f"  Config         : {model['config_path']}")
    print(f"  Variant        : {model['variant']}")
    print(f"Output           : {manifest['output_dir']}")
    launch = manifest.get("launch") or {}
    if launch:
        print("")
        print("Resolved launch:")
        print(f"  Config         : {launch.get('config')}")
        for key, source in (launch.get("sources") or {}).items():
            print(f"  {key:<15}: {source}")
    print("=" * 60)


def log_dir(base_logs: Path, manifest: dict[str, Any]) -> Path:
    now = datetime.now()
    benchmark = manifest["benchmark"]
    date = now.strftime("%Y-%m-%d")
    run_id = now.strftime("%Y%m%d_%H%M%S")
    experiment_key = f"{benchmark['domain']}_{benchmark['experiment']}"
    model = benchmark["model"]
    variant = manifest["model"].get("variant")
    model_key = model if variant in {None, "default"} else f"{model}_{variant}"
    shard = manifest["execution"].get("shard")
    if shard:
        model_key = f"{model_key}_shard_{int(shard['index']):03d}_of_{int(shard['total']):03d}"
    return base_logs / experiment_key / date / f"{model_key}_{run_id}"


def _write_resolved_config(log, manifest: dict[str, Any]) -> None:
    benchmark = manifest["benchmark"]
    log.write("[RUN CONFIGURATION]\n")
    log.write(f"experiment: {benchmark['domain']}_{benchmark['experiment']}\n")
    log.write(f"model: {benchmark['model']}\n")
    if benchmark.get("reference_mode"):
        log.write(f"reference_mode: {benchmark['reference_mode']}\n")
    log.write(f"variant: {manifest['model'].get('variant', 'default')}\n")
    log.write(f"pipeline_set: {manifest['pipeline']['set']}\n")
    log.write(f"pipeline_count: {manifest['pipeline']['count']}\n")
    log.write(f"binning: {manifest['pipeline']['binning']}\n")
    log.write(f"model_config: {manifest['model']['config_path']}\n")
    launch = manifest.get("launch") or {}
    if launch:
        log.write("sources:\n")
        log.write(f"  selected_config: {launch.get('config')}\n")
        for key, source in (launch.get("sources") or {}).items():
            log.write(f"  {key}: {source}\n")
    log.write("model_parameters:\n")
    for key, value in (manifest.get("active_model_parameters") or {}).items():
        log.write(f"  {key}: {value}\n")
    log.write("runtime:\n")
    log.write(f"  workers: {manifest['execution']['workers']}\n")
    log.write(f"  shard: {manifest['execution'].get('shard')}\n")
    log.write(f"output_dir: {manifest['output_dir']}\n")
    log.write("[EXECUTION]\n")


def run_logged(
    command: list[str],
    cwd: Path,
    output_path: Path,
    log_path_root: Path,
    manifest: dict[str, Any],
    temp_path: Path | None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    log_path_root.mkdir(parents=True, exist_ok=True)
    log_path = log_path_root / "run.log"
    started = time.perf_counter()
    status = "success"
    cleanup_status = "not_needed"
    with log_path.open("w") as log:
        _write_resolved_config(log, manifest)
        log.write("Running: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
        if return_code:
            status = "failed"
            log.write(f"FAILED return_code={return_code}\n")
            raise subprocess.CalledProcessError(return_code, command)
    try:
        if temp_path and temp_path.exists():
            shutil.rmtree(temp_path)
            cleanup_status = "removed"
    finally:
        elapsed = time.perf_counter() - started
        manifest["elapsed_seconds"] = round(elapsed, 3)
        manifest["status"] = status
        manifest["cleanup_status"] = cleanup_status
        with log_path.open("a") as log:
            log.write(f"Status: {status}\n")
            log.write(f"Elapsed seconds: {elapsed:.1f}\n")
            log.write(f"Cleanup: {cleanup_status}\n")
            log.write(f"Results: {output_path}\n")
