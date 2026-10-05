"""Mixture benchmark command construction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmarking.core.config import PROJECT_ROOT, load_config_block
from benchmarking.core.datasets import (
    canonical_reference_mode,
    dataset_summary,
    reference_stage,
    validate_reference_mode,
)
from benchmarking.core.outputs import log_dir, manifest_base, output_dir, print_summary, run_logged, temp_dir
from benchmarking.core.pipelines import apply_shard, pipeline_ids, write_pipeline_manifest
from benchmarking.model_specs.registry import get_model_spec
from benchmarking.model_specs.specs import active_model_parameters


def _model_config_arg(model: str, config_path: Path) -> list[str]:
    if model in {"mahe", "nmf", "yang"}:
        return ["--model-config", str(config_path)]
    return []


def _score_modes(model_config: dict[str, Any], variant: str) -> list[str]:
    if variant == "default":
        return list(model_config.get("score_modes", []))
    return list(model_config.get("score_modes", []))


def _threshold_selection(model_config: dict[str, Any]) -> str | None:
    evaluation = model_config.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return None
    value = evaluation.get("threshold_selection")
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1:
        return str(value[0])
    raise ValueError("evaluation.threshold_selection must be a single metric name.")


def _shard_slug(shard: dict[str, int] | None) -> str | None:
    if not shard:
        return None
    return f"shard_{int(shard['index']):03d}_of_{int(shard['total']):03d}"


def run_mixture_model(
    *,
    config: dict[str, Any],
    datasets: dict[str, Any],
    domain: str,
    experiment: str,
    model: str,
    reference_mode: str | None,
    dry_run: bool,
    variant: str = "default",
    shard: dict[str, int] | None = None,
    launch_context: dict[str, Any] | None = None,
) -> None:
    spec = get_model_spec(model)
    if spec.task != "mixture" or spec.adapter_backend is None:
        raise ValueError(f"Unsupported mixture model {model!r}.")
    paths = config["paths"]
    execution = config.get("runtime") or {}
    workers = int(execution.get("workers", 1))
    base_output = Path(paths["output_dir"])
    base_temp = Path(paths["temp_dir"])
    base_logs = Path(paths["logs_dir"])
    pipelines = apply_shard(pipeline_ids(experiment), shard)
    if variant != "default":
        raise ValueError("Only preprocessing pipeline ablations are supported in the public benchmark runner.")
    raw_config, model_config = load_config_block(spec.config_path)
    reference_cfg = model_config.get("reference_training") or {}
    mode = canonical_reference_mode(reference_mode or reference_cfg.get("mode") or raw_config.get("reference_mode_default"))
    validate_reference_mode(model, mode, datasets)
    if mode not in spec.supports_reference_modes:
        raise ValueError(f"{model} does not support reference mode {mode}.")
    seed = int(model_config.get("seed", 42))
    output_path = output_dir(base_output, domain, experiment, model, mode)
    run_temp = temp_dir(base_temp, domain, experiment, model, mode)
    shard_slug = _shard_slug(shard)
    if shard_slug:
        run_temp = run_temp / shard_slug
    active_params = active_model_parameters({"default": model_config}, "default")
    if "score_modes" in model_config:
        active_params["score_modes"] = list(model_config["score_modes"])
    manifest = manifest_base(
        domain=domain,
        experiment=experiment,
        model=model,
        reference_mode=mode,
        output_path=output_path if variant == "default" else output_path / variant,
        pipeline_count=len(pipelines),
        seed=seed,
        workers=workers,
        datasets=dataset_summary(datasets),
        model_config_path=spec.config_path if variant == "default" else spec.ablation_config_path,
        model_config=model_config,
        variant=variant,
        shard=shard,
    )
    manifest["active_model_parameters"] = active_params
    if launch_context:
        manifest["launch"] = launch_context
    if variant != "default":
        output_path = output_path / variant
        run_temp = run_temp / variant
    print_summary(manifest, dry_run=dry_run)
    command_preview_manifest = (
        run_temp / "pipeline_manifest.csv"
        if shard_slug
        else output_path / "pipeline_manifest.csv"
    )
    command = [
        sys.executable,
        str(spec.adapter_backend),
        "--output-dir",
        str(output_path),
        "--temp-root",
        str(run_temp),
        "--benchmark-stage",
        reference_stage(mode),
        "--pipeline-list",
        str(command_preview_manifest),
        "--preprocess-workers",
        str(workers),
        "--rf-training-pure-roots",
        str(datasets["marisma"]["root"]),
        str(datasets["driams_a"]["root"]),
        "--validation-mixture-roots",
        *[str(path) for path in datasets["experimental_mixtures"]["manifests"]],
    ]
    command.extend(_model_config_arg(model, spec.config_path if variant == "default" else spec.ablation_config_path))
    if str(model).startswith("rf_cosine"):
        if "n_estimators" in active_params:
            command.extend(["--rf-n-estimators", str(active_params["n_estimators"])])
        if "negative_top_k" in active_params:
            command.extend(["--rf-negative-top-k", str(active_params["negative_top_k"])])
        threshold_selection = _threshold_selection(model_config)
        if threshold_selection:
            command.extend(["--threshold-selection", threshold_selection])
        score_modes = list(model_config.get("score_modes", []))
        if score_modes:
            command.extend(["--score-modes", *score_modes])
    if dry_run:
        print("Command:")
        print(" ".join(command))
        print("Active model parameters:")
        for key, value in active_params.items():
            print(f"  {key}: {value}")
        print(f"Log directory pattern: {base_logs / (domain + '_' + experiment) / 'YYYY-MM-DD'}")
        return
    pipeline_file = write_pipeline_manifest(command_preview_manifest, pipelines)
    command[command.index(str(command_preview_manifest))] = str(pipeline_file)
    run_logged(command, PROJECT_ROOT, output_path, log_dir(base_logs, manifest), manifest, run_temp)
