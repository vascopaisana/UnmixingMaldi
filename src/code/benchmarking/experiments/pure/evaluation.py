"""Pure benchmark command construction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmarking.core.config import PROJECT_ROOT, load_config_block
from benchmarking.core.datasets import dataset_summary
from benchmarking.core.outputs import log_dir, manifest_base, output_dir, print_summary, run_logged, temp_dir
from benchmarking.core.pipelines import apply_shard, pipeline_ids, write_pipeline_manifest
from benchmarking.model_specs.registry import get_model_spec


PURE_CONFIG_MODELS = {
    "all": "all",
    "random_forest": "rf_multiclass",
    "svm": "svm",
    "logistic_regression": "logistic_l1",
    "knn": "knn",
}


def _shard_slug(shard: dict[str, int] | None) -> str | None:
    if not shard:
        return None
    return f"shard_{int(shard['index']):03d}_of_{int(shard['total']):03d}"


def run_pure_model(
    *,
    config: dict[str, Any],
    datasets: dict[str, Any],
    domain: str,
    experiment: str,
    model: str,
    dry_run: bool,
    shard: dict[str, int] | None = None,
    launch_context: dict[str, Any] | None = None,
) -> None:
    spec = get_model_spec("svm" if model == "all" else model)
    pure_algorithm = "all" if model == "all" else spec.pure_algorithm
    if spec.task != "pure" or spec.adapter_backend is None or pure_algorithm is None:
        raise ValueError(f"Unsupported pure model {model!r}.")
    paths = config["paths"]
    execution = config.get("runtime") or {}
    workers = int(execution.get("workers", 1))
    base_output = Path(paths["output_dir"])
    base_temp = Path(paths["temp_dir"])
    base_logs = Path(paths["logs_dir"])
    output_path = output_dir(base_output, domain, experiment, model)
    run_temp = temp_dir(base_temp, domain, experiment, model)
    shard_slug = _shard_slug(shard)
    if shard_slug:
        run_temp = run_temp / shard_slug
    pipelines = apply_shard(pipeline_ids(experiment), shard)
    _raw_config, model_config = load_config_block(spec.config_path)
    seed = int(model_config.get("seed", 7))
    if model == "all":
        active_params = {
            key: dict(value or {})
            for key, value in (model_config.get("models") or {}).items()
        }
        active_params["seed"] = seed
    else:
        active_params = dict((model_config.get("models") or {}).get(pure_algorithm, {}))
        active_params["seed"] = seed
    manifest = manifest_base(
        domain=domain,
        experiment=experiment,
        model=model,
        output_path=output_path,
        pipeline_count=len(pipelines),
        seed=seed,
        workers=workers,
        datasets=dataset_summary(datasets),
        model_config_path=spec.config_path,
        model_config=model_config,
        shard=shard,
    )
    manifest["active_model_parameters"] = active_params
    if launch_context:
        manifest["launch"] = launch_context
    print_summary(manifest, dry_run=dry_run)
    command = [
        sys.executable,
        str(spec.adapter_backend),
        "--output-dir",
        str(output_path),
        "--pipeline-file",
        str(run_temp / "pipeline_manifest.csv" if shard_slug else output_path / "pipeline_manifest.csv"),
        "--algorithm",
        pure_algorithm,
        "--seed",
        str(seed),
        "--benchmark-binning",
        "b9da_avg",
        "--pipeline-jobs",
        "1",
        "--spectrum-jobs",
        str(workers),
        "--inner-jobs",
        str(workers),
    ]
    if dry_run:
        print("Command:")
        print(" ".join(command))
        print("Active model parameters:")
        for key, value in active_params.items():
            print(f"  {key}: {value}")
        print(f"Log directory pattern: {base_logs / (domain + '_' + experiment) / 'YYYY-MM-DD'}")
        return
    command_manifest = run_temp / "pipeline_manifest.csv" if shard_slug else output_path / "pipeline_manifest.csv"
    pipeline_file = write_pipeline_manifest(command_manifest, pipelines)
    command[command.index(str(command_manifest))] = str(pipeline_file)
    run_logged(command, PROJECT_ROOT, output_path, log_dir(base_logs, manifest), manifest, run_temp)
