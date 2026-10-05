#!/usr/bin/env python3
"""Evaluate preprocessing pipelines with species-mean fixed-H KL NMF.

This skips the per-species NMF prototype fitting step. For each preprocessing
pipeline it builds one prototype per species by averaging the 16600 pure
training spectra, keeps that H fixed, and infers mixture coefficients with
KL-divergence multiplicative updates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARKING_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = CODE_DIR / "models"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from benchmarking.core.config import coerce_path_args, load_model_parameters  # noqa: E402
from benchmarking.core.metrics import (  # noqa: E402
    dataset_prauc_metrics,
    global_prauc,
    safe_slug,
    species_count_prauc_metrics,
    species_prauc_metrics,
    supported_species_prauc,
)
from benchmarking.core.pipeline_manifest import DEFAULT_MATRIX, PROJECT_ROOT  # noqa: E402
from benchmarking.core.preprocessing import (  # noqa: E402
    build_temporary_h5s_for_pipeline,
    load_mixture_spots,
    load_pure_validation_spots,
    load_top_pipeline_names,
    load_training_data,
    threshold_candidates_from_scores,
)
from benchmarking.model_specs.adapters.rf_cosine import (  # noqa: E402
    DEFAULT_BENCHMARK_STAGE,
    DEFAULT_RF_TRAINING_PURE_ROOTS,
    DEFAULT_VALIDATION_MIXTURE_ROOTS,
    configure_driams_marisma_rf_matrix,
    keep_threshold_selection,
    load_existing_outputs,
    load_pipeline_labels,
    load_pipeline_list,
    load_yaml,
    override_pipeline_binning,
    prefixed_metrics_for_threshold,
    select_pipeline_shard,
    select_pipeline_specs,
    select_one_threshold,
    select_thresholds,
    threshold_selection_from_model_config,
    threshold_metrics,
    write_outputs,
)
from models.nmf import NMFModel  # noqa: E402


DEFAULT_RANKING = None
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/benchmarks/nmf"
DEFAULT_TEMP_ROOT = DEFAULT_OUTPUT_DIR / "_tmp_h5"
DEFAULT_MODEL_CONFIG = BENCHMARKING_DIR / "configs/mixture/nmf.yaml"

NMF_MEAN_PURE_FIXED_H_KL_DEFAULT_CONFIG: dict[str, Any] = {
    "prototype": "species_mean",
    "prototype_scaling": "none",
    "inference_solver": "mu",
    "inference_beta_loss": "kullback-leibler",
    "inference_alpha_W": 1.0e-6,
    "inference_l1_ratio": 0.75,
    "score_mode": "coefficient_raw",
    "normalize_coeffs": False,
    "clip_negative_values": True,
    "max_iter": 1000,
    "tol": 1.0e-5,
}
NMF_MEAN_PURE_FIXED_H_KL_CONFIG: dict[str, Any] = dict(NMF_MEAN_PURE_FIXED_H_KL_DEFAULT_CONFIG)


def fit_nmf_mean_pure_fixed_h_kl(
    spectra: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any] | None = None,
) -> NMFModel:
    profile_ids = list(range(len(labels)))
    profile_to_species = {profile_id: str(labels[profile_id]) for profile_id in profile_ids}
    model = NMFModel(**(config or NMF_MEAN_PURE_FIXED_H_KL_CONFIG))
    model.fit(spectra, profile_ids=profile_ids, profile_to_species=profile_to_species)
    return model


def score_spots_with_nmf(
    matrix: dict[str, Any],
    spots: pd.DataFrame,
    model: NMFModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    species = list(matrix["species"])
    model_species_to_col = {species_id: idx for idx, species_id in enumerate(model.species_ids_)}
    presence_threshold = float(getattr(model, "active_threshold", 0.0))

    score_rows = []
    abundance_rows = []
    truth_rows = []
    diagnostic_rows = []
    for spot_index, row in spots.reset_index(drop=True).iterrows():
        spectrum = np.asarray(row["spectrum"], dtype=np.float64)
        pred = model.predict_scores(spectrum)
        scores = np.asarray(pred["coeffs"], dtype=np.float64)
        aligned_scores = np.zeros(len(species), dtype=np.float64)
        for col, species_name in enumerate(species):
            model_col = model_species_to_col.get(species_name)
            if model_col is not None:
                aligned_scores[col] = scores[model_col]

        abundances = np.asarray([float(row[f"abundance_{name}"]) for name in species], dtype=np.float64)
        score_rows.append(aligned_scores)
        abundance_rows.append(abundances)
        truth_rows.append(abundances > presence_threshold)
        diagnostic_rows.append(
            {
                "spot_index": int(spot_index),
                "rss": float(pred.get("rss", np.nan)),
                "reconstruction_cosine": float(pred.get("reconstruction_cosine", np.nan)),
                "n_active": int(np.sum(aligned_scores > 0)),
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    return (
        np.asarray(truth_rows, dtype=bool),
        np.asarray(score_rows, dtype=np.float64),
        np.asarray(abundance_rows, dtype=np.float64),
        diagnostics,
    )


def base_pipeline_row(spec: dict[str, Any], model: NMFModel, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline": spec["pipeline"],
        "pipeline_id": spec["pipeline"],
        "benchmark_stage": spec.get("benchmark_stage"),
        "model_name": "nmf_mean_pure_fixed_h_kl",
        "nmf_config": json.dumps(config, sort_keys=True),
        "order": spec.get("order"),
        "method_grid_baseline": spec.get("method_grid_baseline", "snip"),
        "variance_stabilization": spec.get("variance_stabilization"),
        "binning": spec.get("binning"),
        "normalization": spec.get("normalization"),
        "peak_selection": spec.get("peak_selection"),
        "n_nmf_prototypes": int(len(model.species_ids_)),
    }


def evaluate_one_pipeline(
    spec: dict[str, Any],
    matrix: dict[str, Any],
    temp_root: Path,
    preprocess_workers: int,
    pipeline_index: int,
    total_pipelines: int,
    model_config: dict[str, Any],
    threshold_selection_metric: str,
) -> dict[str, dict[str, Any]]:
    pipeline_name = spec["pipeline"]
    started = time.perf_counter()
    print(f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: generating temporary H5s", flush=True)
    active_matrix = build_temporary_h5s_for_pipeline(
        spec,
        matrix,
        temp_root / safe_slug(pipeline_name),
        preprocess_workers=preprocess_workers,
    )

    print(f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: loading training data", flush=True)
    spectra, labels, _origins, mz_axis = load_training_data(active_matrix, pipeline_name)
    print(
        f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: "
        f"training spectra={len(spectra)} species={len(np.unique(labels))} bins={spectra.shape[1]}",
        flush=True,
    )
    negative_count = int(np.sum(spectra < 0))
    if negative_count:
        print(
            f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: "
            f"clipping {negative_count} negative training values to zero "
            f"(min={float(np.min(spectra)):.6g}) for KL NMF",
            flush=True,
        )

    print(f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: fitting species-mean fixed-H NMF", flush=True)
    model = fit_nmf_mean_pure_fixed_h_kl(spectra, labels, model_config)

    print(f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: scoring mixtures", flush=True)
    mixture_spots = load_mixture_spots(active_matrix, pipeline_name, mz_axis)
    truth, scores, abundance_matrix, _diagnostics = score_spots_with_nmf(active_matrix, mixture_spots, model)
    threshold_table = threshold_metrics(truth, scores, mixture_spots["mixture_dataset"].astype(str).to_numpy())

    pure_spots = load_pure_validation_spots(active_matrix, pipeline_name, mz_axis)
    if pure_spots.empty:
        pure_truth = pure_scores = pure_abundance_matrix = None
    else:
        pure_truth, pure_scores, pure_abundance_matrix, _pure_diagnostics = score_spots_with_nmf(
            active_matrix,
            pure_spots,
            model,
        )

    base = base_pipeline_row(spec, model, model_config)
    base["mixture_prauc_global"] = global_prauc(truth, scores)
    base["mixture_prauc_supported_species"] = supported_species_prauc(truth, scores)
    datasets = mixture_spots["mixture_dataset"].astype(str).to_numpy()
    base.update(dataset_prauc_metrics(truth, scores, datasets, prefix="mixture"))
    base.update(species_prauc_metrics(truth, scores, active_matrix["species"], prefix="mixture"))
    if "num_species" in mixture_spots.columns:
        base.update(
            species_count_prauc_metrics(
                truth,
                scores,
                mixture_spots["num_species"].astype(int).to_numpy(),
                prefix="mixture",
            )
        )
    if pure_truth is not None and pure_scores is not None:
        base["pure_prauc_global"] = global_prauc(pure_truth, pure_scores)
        base["pure_prauc_supported_species"] = supported_species_prauc(pure_truth, pure_scores)
        constrained_thresholds = threshold_candidates_from_scores(
            np.concatenate([scores.ravel(), pure_scores.ravel()])
        )
        constrained_mixture_table = threshold_metrics(
            truth,
            scores,
            mixture_spots["mixture_dataset"].astype(str).to_numpy(),
            thresholds=constrained_thresholds,
        )
        constrained_pure_table = threshold_metrics(pure_truth, pure_scores, thresholds=constrained_thresholds)
        selected = select_thresholds(threshold_table, constrained_pure_table)
        constrained_selected = select_thresholds(constrained_mixture_table, constrained_pure_table)
        constrained_key = "pure_precision_max_pure_recall_max_mixture_precision_095_or_max_mixture_recall_max"
        if constrained_key in constrained_selected:
            selected[constrained_key] = constrained_selected[constrained_key]
    else:
        selected = select_thresholds(threshold_table)
    selected = select_one_threshold(selected, threshold_selection_metric)

    rows_by_criterion = {}
    for criterion, threshold in selected.items():
        row = dict(base)
        row["selected_threshold"] = float(threshold)
        row["threshold_selection_metric"] = criterion
        row.update(
            prefixed_metrics_for_threshold(
                "mixture",
                threshold,
                truth,
                scores,
                mixture_spots,
                abundance_matrix,
                active_matrix["species"],
                include_abundance_bins=True,
            )
        )
        if pure_truth is not None and pure_scores is not None and pure_abundance_matrix is not None:
            row.update(
                prefixed_metrics_for_threshold(
                    "pure",
                    threshold,
                    pure_truth,
                    pure_scores,
                    pure_spots,
                    pure_abundance_matrix,
                    active_matrix["species"],
                    include_abundance_bins=False,
                )
            )
        rows_by_criterion[criterion] = row

    print(
        f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: done in "
        f"{(time.perf_counter() - started) / 60:.1f} min; "
        f"mixture_prauc_global={base['mixture_prauc_global']:.4f}",
        flush=True,
    )
    return rows_by_criterion


def completed_pipelines(output_dir: Path) -> set[str]:
    path = output_dir / "pipeline_metrics.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=lambda col: col in {"pipeline_id", "pipeline"})
    pipeline_col = "pipeline_id" if "pipeline_id" in frame.columns else "pipeline"
    if pipeline_col not in frame.columns:
        return set()
    return set(frame[pipeline_col].dropna().astype(str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--benchmark-stage", default=DEFAULT_BENCHMARK_STAGE)
    parser.add_argument("--ranking-stage", default="driams_marisma_pures_to_experimental_mixtures")
    parser.add_argument("--pipeline-list", type=Path, default=None)
    parser.add_argument("--pipeline-binning-override", default=None)
    parser.add_argument("--pipeline-shard-index", type=int, default=None)
    parser.add_argument("--pipeline-num-shards", type=int, default=None)
    parser.add_argument("--pipelines", nargs="*", default=None)
    parser.add_argument("--pipeline-labels", type=Path, default=None)
    parser.add_argument("--max-pipelines", type=int, default=None)
    parser.add_argument("--rf-training-pure-roots", nargs="*", type=Path, default=DEFAULT_RF_TRAINING_PURE_ROOTS)
    parser.add_argument(
        "--rf-training-max-spectra-per-species-per-origin",
        default=None,
        help=(
            "Cap per species/origin for pure training spectra. Default is no cap, "
            "which uses the full 16600-spectrum MARISMA/DRIAMS-A pure set."
        ),
    )
    parser.add_argument("--validation-mixture-roots", nargs="*", type=Path, default=DEFAULT_VALIDATION_MIXTURE_ROOTS)
    parser.add_argument("--preprocess-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    args = parser.parse_args()
    coerce_path_args(
        args,
        ("matrix", "ranking", "pipeline_list", "output_dir", "temp_root", "pipeline_labels", "model_config"),
        ("rf_training_pure_roots", "validation_mixture_roots"),
    )
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)

    matrix = configure_driams_marisma_rf_matrix(load_yaml(args.matrix), args)

    if args.pipelines:
        pipeline_names = args.pipelines
    elif args.pipeline_list is not None:
        pipeline_names = load_pipeline_list(args.pipeline_list)
    elif args.ranking is not None:
        pipeline_names = load_top_pipeline_names(args.ranking, args.ranking_stage, args.max_pipelines)
    else:
        raise ValueError("Provide --pipeline-list, --pipelines, or --ranking.")
    pipeline_names = override_pipeline_binning(pipeline_names, args.pipeline_binning_override)
    pipeline_names = select_pipeline_shard(pipeline_names, args.pipeline_shard_index, args.pipeline_num_shards)
    if args.resume:
        done = completed_pipelines(args.output_dir)
        pipeline_names = [name for name in pipeline_names if name not in done]

    specs = select_pipeline_specs(matrix, pipeline_names)
    labels = load_pipeline_labels(args.pipeline_labels)
    threshold_selection_metric = threshold_selection_from_model_config(args.model_config)
    rows_by_criterion = keep_threshold_selection(load_existing_outputs(args.output_dir), threshold_selection_metric)
    errors = []
    model_config = load_model_parameters("nmf", NMF_MEAN_PURE_FIXED_H_KL_DEFAULT_CONFIG, args.model_config)

    print(f"NMF config: {json.dumps(model_config, sort_keys=True)}", flush=True)
    print(f"Pipelines to run: {len(specs)} | preprocess_workers={args.preprocess_workers}", flush=True)
    for index, spec in enumerate(specs, start=1):
        pipeline_name = str(spec["pipeline"])
        label = labels.get(pipeline_name)
        print(
            f"[{index}/{len(specs)}] evaluating {pipeline_name}" + (f" | {label}" if label else ""),
            flush=True,
        )
        pipeline_temp_root = args.temp_root / safe_slug(pipeline_name)
        try:
            rows = evaluate_one_pipeline(
                spec,
                matrix,
                args.temp_root,
                args.preprocess_workers,
                index,
                len(specs),
                model_config,
                threshold_selection_metric,
            )
        except Exception as exc:
            if pipeline_temp_root.exists() and not args.keep_temp:
                shutil.rmtree(pipeline_temp_root)
            if not args.continue_on_error:
                raise
            print(f"[{index}/{len(specs)}] FAILED {pipeline_name}: {exc!r}", flush=True)
            errors.append({"pipeline": pipeline_name, "error": repr(exc)})
            continue
        if pipeline_temp_root.exists() and not args.keep_temp:
            shutil.rmtree(pipeline_temp_root)
        for criterion, row in rows.items():
            rows_by_criterion[criterion].append(row)
        write_outputs(rows_by_criterion, args.output_dir)

    if errors:
        pd.DataFrame(errors).to_csv(args.output_dir / "pipeline_evaluation_errors.csv", index=False)
    write_outputs(rows_by_criterion, args.output_dir)


if __name__ == "__main__":
    main()
