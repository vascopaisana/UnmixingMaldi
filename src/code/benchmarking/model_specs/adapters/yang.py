#!/usr/bin/env python3
"""Evaluate preprocessing pipelines with the optimized Yang baseline."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

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
    canonicalize_output_rows,
    configure_driams_marisma_rf_matrix,
    keep_threshold_selection,
    load_existing_outputs,
    load_pipeline_labels,
    load_pipeline_list,
    load_yaml,
    ordered_columns,
    override_pipeline_binning,
    prefixed_metrics_for_threshold,
    select_pipeline_shard,
    select_pipeline_specs,
    select_one_threshold,
    select_thresholds,
    threshold_selection_from_model_config,
    threshold_metrics,
)
from models.yang import YangModel  # noqa: E402


DEFAULT_RANKING = None
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/benchmarks/yang"
DEFAULT_TEMP_ROOT = DEFAULT_OUTPUT_DIR / "_tmp_h5"
DEFAULT_MODEL_CONFIG = BENCHMARKING_DIR / "configs/mixture/yang.yaml"

YANG_DEFAULT_CONFIG: dict[str, Any] = {
    "normalize_coeffs": False,
    "max_species": 6,
    "max_candidates": 7,
    "score_mode": "coefficient_x_fit_cosine",
    "use_resampling": True,
    "keep_fraction": 0.15,
    "block_size_bins": 45,
    "n_resamples": 100,
    "resample_aggregation": "mean",
    "reselect_candidates_per_resample": True,
    "reselect_profiles_per_resample": True,
    "random_state": 42,
    "presence_threshold": 0.0,
}
YANG_CONFIG: dict[str, Any] = dict(YANG_DEFAULT_CONFIG)

_WORKER_MODEL: YangModel | None = None
_WORKER_SPECIES: list[str] | None = None
_WORKER_MODEL_SPECIES_TO_COL: dict[Any, int] | None = None
_WORKER_PRESENCE_THRESHOLD: float = 0.0


def _init_yang_score_worker() -> None:
    threadpool_limits(limits=1)


def _score_yang_spot_payload(
    payload: tuple[int, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if _WORKER_MODEL is None or _WORKER_SPECIES is None or _WORKER_MODEL_SPECIES_TO_COL is None:
        raise RuntimeError("Yang scoring worker was not initialised.")

    spot_index, spectrum, abundances = payload
    _active_species, _active_scores, info = _WORKER_MODEL.predict(np.asarray(spectrum, dtype=np.float64))
    scores = np.asarray(info.get("score_vector"), dtype=np.float64)
    aligned_scores = np.zeros(len(_WORKER_SPECIES), dtype=np.float64)
    for col, species_name in enumerate(_WORKER_SPECIES):
        model_col = _WORKER_MODEL_SPECIES_TO_COL.get(species_name)
        if model_col is not None:
            aligned_scores[col] = scores[model_col]

    diagnostics = {
        "spot_index": int(spot_index),
        "fit_cosine": float(info.get("fit_cosine", np.nan)),
        "n_active": int(np.sum(aligned_scores > 0)),
        "n_candidate_species": int(len(info.get("candidate_species_ids", []))),
    }
    abundances = np.asarray(abundances, dtype=np.float64)
    return (
        abundances > _WORKER_PRESENCE_THRESHOLD,
        aligned_scores,
        abundances,
        diagnostics,
    )


def fit_yang(spectra: np.ndarray, labels: np.ndarray, config: dict[str, Any] | None = None) -> YangModel:
    ref_ids = list(range(len(labels)))
    id_to_species = {ref_id: str(labels[ref_id]) for ref_id in ref_ids}
    species_names = {str(label): str(label) for label in np.unique(labels)}
    model = YangModel(**(config or YANG_CONFIG))
    model.fit(
        np.asarray(spectra, dtype=np.float64),
        ref_ids=ref_ids,
        id_to_species=id_to_species,
        species_names=species_names,
    )
    return model


def score_spots_with_yang(
    matrix: dict[str, Any],
    spots: pd.DataFrame,
    model: YangModel,
    score_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    species = list(matrix["species"])
    model_species_to_col = {species_id: idx for idx, species_id in enumerate(model.species_ids_)}
    presence_threshold = float(YANG_CONFIG.get("presence_threshold", 0.0))

    spots = spots.reset_index(drop=True)
    payloads = [
        (
            int(spot_index),
            np.asarray(row["spectrum"], dtype=np.float64),
            np.asarray([float(row[f"abundance_{name}"]) for name in species], dtype=np.float64),
        )
        for spot_index, row in spots.iterrows()
    ]

    global _WORKER_MODEL
    global _WORKER_SPECIES
    global _WORKER_MODEL_SPECIES_TO_COL
    global _WORKER_PRESENCE_THRESHOLD
    _WORKER_MODEL = model
    _WORKER_SPECIES = species
    _WORKER_MODEL_SPECIES_TO_COL = model_species_to_col
    _WORKER_PRESENCE_THRESHOLD = presence_threshold

    workers = min(max(1, int(score_workers)), len(payloads) or 1)
    if workers > 1 and len(payloads) > 1:
        context = mp.get_context("fork")
        with context.Pool(processes=workers, initializer=_init_yang_score_worker) as pool:
            results = pool.map(_score_yang_spot_payload, payloads)
    else:
        _init_yang_score_worker()
        results = [
            _score_yang_spot_payload(payload)
            for payload in payloads
        ]

    truth_rows = [item[0] for item in results]
    score_rows = [item[1] for item in results]
    abundance_rows = [item[2] for item in results]
    diagnostic_rows = [item[3] for item in results]

    diagnostics = pd.DataFrame(diagnostic_rows)
    return (
        np.asarray(truth_rows, dtype=bool),
        np.asarray(score_rows, dtype=np.float64),
        np.asarray(abundance_rows, dtype=np.float64),
        diagnostics,
    )


def base_pipeline_row(spec: dict[str, Any], model: YangModel, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline": spec["pipeline"],
        "pipeline_id": spec["pipeline"],
        "benchmark_stage": spec.get("benchmark_stage"),
        "model_name": "yang_mean_screen_best_profile_window_jackknife",
        "yang_config": json.dumps(config, sort_keys=True),
        "order": spec.get("order"),
        "method_grid_baseline": spec.get("method_grid_baseline", "snip"),
        "variance_stabilization": spec.get("variance_stabilization"),
        "binning": spec.get("binning"),
        "normalization": spec.get("normalization"),
        "peak_selection": spec.get("peak_selection"),
        "n_yang_reference_profiles": int(len(model.ref_ids_)),
        "n_yang_species": int(len(model.species_ids_)),
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

    print(f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: fitting Yang", flush=True)
    model = fit_yang(spectra, labels, model_config)

    print(f"[{pipeline_index}/{total_pipelines}] {pipeline_name}: scoring mixtures", flush=True)
    mixture_spots = load_mixture_spots(active_matrix, pipeline_name, mz_axis)
    truth, scores, abundance_matrix, _diagnostics = score_spots_with_yang(
        active_matrix,
        mixture_spots,
        model,
        score_workers=preprocess_workers,
    )
    threshold_table = threshold_metrics(truth, scores, mixture_spots["mixture_dataset"].astype(str).to_numpy())

    pure_spots = load_pure_validation_spots(active_matrix, pipeline_name, mz_axis)
    if pure_spots.empty:
        pure_truth = pure_scores = pure_abundance_matrix = None
    else:
        pure_truth, pure_scores, pure_abundance_matrix, _pure_diagnostics = score_spots_with_yang(
            active_matrix,
            pure_spots,
            model,
            score_workers=preprocess_workers,
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
    try:
        frame = pd.read_csv(path, usecols=lambda col: col in {"pipeline_id", "pipeline"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Could not read completed pipelines from {path}: {exc!r}", flush=True)
        return set()
    pipeline_col = "pipeline_id" if "pipeline_id" in frame.columns else "pipeline"
    if pipeline_col not in frame.columns:
        return set()
    return set(frame[pipeline_col].dropna().astype(str))


def merge_rows_by_pipeline(
    old_rows_by_criterion: dict[str, list[dict[str, Any]]],
    new_rows_by_criterion: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    criteria = set(old_rows_by_criterion) | set(new_rows_by_criterion)
    for criterion in criteria:
        rows = [
            *old_rows_by_criterion.get(criterion, []),
            *new_rows_by_criterion.get(criterion, []),
        ]
        by_pipeline: dict[str, dict[str, Any]] = {}
        unkeyed = []
        for row in rows:
            pipeline_id = row.get("pipeline_id", row.get("pipeline"))
            if pipeline_id is None or pd.isna(pipeline_id):
                unkeyed.append(row)
            else:
                by_pipeline[str(pipeline_id)] = row
        merged[criterion] = [*unkeyed, *by_pipeline.values()]
    return merged


def load_existing_outputs_safe(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    combined = output_dir / "pipeline_metrics.csv"
    if combined.exists() and combined.stat().st_size > 0:
        try:
            frame = pd.read_csv(combined)
            if "threshold_selection_metric" in frame.columns:
                return {
                    str(criterion): rows.to_dict("records")
                    for criterion, rows in frame.groupby("threshold_selection_metric", dropna=False)
                }
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            print(f"Could not read existing combined output {combined}: {exc!r}", flush=True)

    filenames = {
        "best_exact_match": "pipeline_evaluation_best_exact_match.csv",
        "best_precision": "pipeline_evaluation_best_precision.csv",
        "best_recall": "pipeline_evaluation_best_recall.csv",
        "best_f1": "pipeline_evaluation_best_f1.csv",
    }
    rows_by_criterion: dict[str, list[dict[str, Any]]] = {}
    for criterion, filename in filenames.items():
        path = output_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            rows_by_criterion[criterion] = []
            continue
        try:
            rows_by_criterion[criterion] = pd.read_csv(path).to_dict("records")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            backup_path = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
            shutil.copy2(path, backup_path)
            print(
                f"WARNING: could not read {path}: {exc!r}. "
                f"Copied it to {backup_path} and continuing with empty rows for {criterion}.",
                flush=True,
            )
            rows_by_criterion[criterion] = []
    return rows_by_criterion


def write_outputs_atomic(rows_by_criterion: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_criterion = canonicalize_output_rows(rows_by_criterion)
    columns = ordered_columns(rows_by_criterion)
    frames = [
        pd.DataFrame(rows)
        for rows in rows_by_criterion.values()
        if rows
    ]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if {"pipeline_id", "threshold_selection_metric"}.issubset(combined.columns):
        combined = combined.drop_duplicates(["pipeline_id", "threshold_selection_metric"], keep="last")
    combined = combined.reindex(columns=columns)
    combined.to_csv(output_dir / "pipeline_metrics.csv", index=False)

    count_columns = [
        column
        for column in combined.columns
        if column in {"tp", "fp", "fn", "tn"}
        or column.endswith(
            (
                "_tp",
                "_fp",
                "_fn",
                "_tn",
                "_n_spots",
                "_n_samples",
                "_n_occurrences",
                "_n_positive",
                "_n_negative",
            )
        )
    ]
    if count_columns:
        id_columns = [
            column
            for column in (
                "pipeline_id",
                "train_dataset",
                "order",
                "baseline_correction",
                "variance_stabilization",
                "smoothing_method",
                "smoothing_window",
                "binning",
                "normalization",
                "threshold_selection_metric",
                "selected_threshold",
            )
            if column in combined.columns
        ]
        combined.loc[:, [*id_columns, *[column for column in count_columns if column not in id_columns]]].to_csv(
            output_dir / "pipeline_counts.csv",
            index=False,
        )


def write_outputs_with_lock(
    rows_by_criterion: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".pipeline_evaluation.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        current_rows = load_existing_outputs_safe(output_dir)
        merged_rows = merge_rows_by_pipeline(current_rows, rows_by_criterion)
        write_outputs_atomic(merged_rows, output_dir)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return merged_rows


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
    rows_by_criterion = keep_threshold_selection(
        load_existing_outputs_safe(args.output_dir),
        threshold_selection_metric,
    )
    errors = []
    model_config = load_model_parameters("yang", YANG_DEFAULT_CONFIG, args.model_config)

    print(f"Yang config: {json.dumps(model_config, sort_keys=True)}", flush=True)
    print(
        f"Pipelines to run: {len(specs)} | preprocess_workers={args.preprocess_workers} "
        f"| score_workers={args.preprocess_workers}",
        flush=True,
    )
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
            rows_by_criterion.setdefault(criterion, []).append(row)
        rows_by_criterion = write_outputs_with_lock(rows_by_criterion, args.output_dir)

    if errors:
        pd.DataFrame(errors).to_csv(args.output_dir / "pipeline_evaluation_errors.csv", index=False)
    write_outputs_with_lock(rows_by_criterion, args.output_dir)


if __name__ == "__main__":
    main()
