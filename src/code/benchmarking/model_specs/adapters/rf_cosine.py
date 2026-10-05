#!/usr/bin/env python3
"""Evaluate MALDI preprocessing pipelines with RF weighting and similarity metrics."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

LOCAL_CODE_DIR = Path(__file__).resolve().parents[3]
if str(LOCAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE_DIR))

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
from data_generation.species_mapping import load_global_species_to_id

from benchmarking.core.config import coerce_path_args, expand_config_values, load_local_env
from benchmarking.core.metrics import (
    ABUNDANCE_BINS,
    ABUNDANCE_DATASET_SLUGS,
    ABUNDANCE_METRIC_NAMES,
    TARGET_MIXTURE_SPECIES,
    MIXTURE_DATASET_SLUGS,
    RF_HOLDOUT_METRIC_NAMES,
    SPECIES_COUNT_VALUES,
    SUBSET_METRIC_NAMES,
    binary_species_metrics,
    dataset_prauc_metrics,
    global_prauc,
    ordered_metric_columns,
    safe_slug,
    species_count_prauc_metrics,
    species_prauc_metrics,
    supported_species_prauc,
    threshold_subset_metrics,
)
from benchmarking.core.pipeline_manifest import DEFAULT_MATRIX, PROJECT_ROOT, iter_combinations
from benchmarking.core.preprocessing_backend import (
    ALL_METHOD_GRID_ORDERS,
    DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD,
    DEFAULT_RF_MIN_SAMPLES_PER_BINARY_CLASS,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_RF_RANDOM_STATE,
    METHOD_GRID_ORDERS,
    apply_benchmark_stage,
    build_temporary_h5s_for_pipeline,
    load_h5_records,
    load_top_pipeline_names,
    load_training_data,
    metadata_source_prefix_from_root,
    normalize_species_name,
    baseline_adjusted_species_scores,
    parse_vector,
    pipeline_h5_path,
    pure_score_matrix_from_means,
    safe_metric,
    species_means,
    split_indices,
    spot_key,
    threshold_candidates_from_scores,
    weighted_cosine,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
DEFAULT_RANKING = None
EXTRACTED_SOURCE_PREFIX = "raw_spectra"
DEFAULT_BENCHMARK_STAGE = "driams_marisma_pures_to_experimental_mixtures"
DEFAULT_RF_TRAINING_PURE_ROOTS = [
    PROJECT_ROOT / "data/raw/MARISMA",
    PROJECT_ROOT / "data/raw/DRIAMS-A",
]
DEFAULT_VALIDATION_MIXTURE_ROOTS = [
    PROJECT_ROOT / "data/manifests/mixtures/local_mixture_set_a.csv",
    PROJECT_ROOT / "data/manifests/mixtures/local_mixture_set_b.csv",
    PROJECT_ROOT / "data/manifests/mixtures/local_mixture_set_c.csv",
]
REDUNDANT_OUTPUT_COLUMNS = {
    "pipeline",
    "pipeline_slug",
    "baseline_method",
    "peak_picking_method",
    "binning_method",
    "normalization_method",
    "method_grid_family",
    "method_grid_baseline",
    "method_grid_order",
    "threshold",
    "rf_balanced_accuracy_mean",
    "rf_f1_mean",
    "rf_average_precision_mean",
    "rf_roc_auc_mean",
}
EXCLUDED_OUTPUT_PREFIXES = (
    "mixture_abundance_local_mixture_set_b_",
    "pure_dataset_",
    "pure_species_",
)
VALID_SCORE_MODES = {
    "cosine",
    "rf_weighted_cosine",
    "rf_weighted_cosine_baseline_adjusted",
}
DEFAULT_THRESHOLD_SELECTION_METRIC = (
    "pure_precision_max_pure_recall_max_mixture_precision_095_or_max_mixture_recall_max"
)
VALID_THRESHOLD_SELECTION_METRICS = {
    "best_exact_match",
    "best_precision",
    "best_f1",
    "best_recall",
    DEFAULT_THRESHOLD_SELECTION_METRIC,
}
PIPELINE_GRID_PREFIX = "pipeline_grid"


def strip_pipeline_prefix(pipeline_id: str) -> str:
    return str(pipeline_id).removeprefix(f"{PIPELINE_GRID_PREFIX}__")


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    load_local_env()
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return expand_config_values(data)


def normalize_threshold_selection(value: Any, *, source: str = "threshold_selection") -> str:
    if value is None:
        return DEFAULT_THRESHOLD_SELECTION_METRIC
    if isinstance(value, str):
        metric = value
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"{source} must name exactly one threshold-selection metric. "
                f"Got {len(value)} values: {list(value)!r}"
            )
        metric = str(value[0])
    else:
        raise TypeError(f"{source} must be a string or single-item list.")
    metric = metric.strip()
    if metric not in VALID_THRESHOLD_SELECTION_METRICS:
        raise ValueError(
            f"Unknown {source}={metric!r}. Valid values: {sorted(VALID_THRESHOLD_SELECTION_METRICS)}"
        )
    return metric


def threshold_selection_from_model_config(config_path: Path | None) -> str:
    if config_path is None:
        return DEFAULT_THRESHOLD_SELECTION_METRIC
    data = load_yaml(config_path)
    block = data.get("default", data)
    evaluation = block.get("evaluation", {}) if isinstance(block, dict) else {}
    value = evaluation.get("threshold_selection") if isinstance(evaluation, dict) else None
    return normalize_threshold_selection(value, source=f"{config_path}: evaluation.threshold_selection")


def select_one_threshold(selected: dict[str, float], metric: str) -> dict[str, float]:
    if metric not in selected:
        raise ValueError(
            f"Threshold-selection metric {metric!r} was requested but could not be computed. "
            f"Available metrics: {sorted(selected)}"
        )
    return {metric: float(selected[metric])}


def _criterion_from_key(key: str) -> str:
    return str(key).rsplit("::", 1)[-1]


def keep_threshold_selection(
    rows_by_criterion: dict[str, list[dict[str, Any]]],
    metric: str,
) -> dict[str, list[dict[str, Any]]]:
    kept = {
        key: rows
        for key, rows in rows_by_criterion.items()
        if _criterion_from_key(key) == metric
    }
    return kept or {metric: []}


def load_pipeline_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        pipeline_col = "pipeline_id" if "pipeline_id" in frame.columns else "pipeline"
        if pipeline_col not in frame.columns:
            raise ValueError(f"{path} must contain pipeline_id or pipeline")
        return frame[pipeline_col].dropna().astype(str).tolist()
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def override_pipeline_binning(pipeline_names: list[str], binning: str | None) -> list[str]:
    if not binning:
        return pipeline_names
    rewritten = []
    for pipeline_name in pipeline_names:
        parts = str(pipeline_name).split("__")
        if len(parts) == 6 and parts[4].endswith("_avg"):
            parts[4] = binning
            rewritten.append("__".join(parts))
        else:
            rewritten.append(str(pipeline_name))
    return list(dict.fromkeys(rewritten))


def select_pipeline_shard(pipeline_names: list[str], shard_index: int | None, num_shards: int | None) -> list[str]:
    if shard_index is None and num_shards is None:
        return pipeline_names
    if shard_index is None or num_shards is None:
        raise ValueError("pipeline_shard_index and pipeline_num_shards must be set together.")
    if num_shards < 1:
        raise ValueError("pipeline_num_shards must be >= 1.")
    if shard_index < 1 or shard_index > num_shards:
        raise ValueError("pipeline_shard_index must be between 1 and pipeline_num_shards.")
    n = len(pipeline_names)
    start = (n * (shard_index - 1)) // num_shards
    end = (n * shard_index) // num_shards
    return pipeline_names[start:end]


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def abundance_percent_values(abundance: np.ndarray) -> np.ndarray:
    abundance = np.asarray(abundance, dtype=float)
    finite = abundance[np.isfinite(abundance)]
    if finite.size and np.nanmax(finite) <= 1.0:
        return abundance * 100.0
    return abundance


def std_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.std(finite, ddof=0)) if finite else float("nan")


def multilabel_metrics(truth: np.ndarray, pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    return threshold_subset_metrics(truth, pred)


def parse_species_label(value: Any) -> str | None:
    return normalize_species_name(value)


def discover_spectrum_paths(sample_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for sample in sample_dir.iterdir():
        if sample.is_dir():
            bruker_dir = sample / "1SLin"
            if (bruker_dir / "fid").exists():
                paths.append(bruker_dir)
                continue
            text_files = sorted(
                child
                for child in sample.iterdir()
                if child.is_file() and child.suffix.lower() in {".txt", ".tsv", ".csv"}
            )
            paths.extend(text_files)
        elif sample.is_file() and sample.suffix.lower() in {".txt", ".tsv", ".csv"}:
            paths.append(sample)
    seen: set[str] = set()
    result = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def origin_from_source_root(source_root: Path) -> str:
    name = safe_slug(source_root.name)
    if name == "marisma":
        return "raw_spectra_marisma_unzipped"
    if name == "driams_a":
        return "raw_spectra_driams_a_unzipped"
    if name.startswith("driams"):
        return f"raw_spectra_{name}_unzipped"
    return f"raw_spectra_{name}_unzipped"


def write_extracted_pure_metadata(
    source_roots: list[Path],
    output_root: Path,
    species: list[str],
) -> dict[str, str] | None:
    species_to_id = load_global_species_to_id()
    keep_species = set(species)
    rows = []
    for source_root in source_roots:
        if not source_root.exists():
            continue
        origin = origin_from_source_root(source_root)
        for species_dir in source_root.iterdir():
            if not species_dir.is_dir():
                continue
            species = parse_species_label(species_dir.name)
            if species not in keep_species:
                continue
            species_id = species_to_id.get(species)
            if species_id is None:
                continue
            for spectrum_path in discover_spectrum_paths(species_dir):
                rows.append(
                    {
                        "file_path": str(spectrum_path),
                        "species_names": json.dumps([species]),
                        "species_ids": json.dumps([species_id]),
                        "abundances": json.dumps([100.0]),
                        "num_species": 1,
                        "origin": origin,
                        "mixture_name": spectrum_path.parent.name,
                        "sample_folder": str(spectrum_path),
                    }
                )
    if not rows:
        return None

    metadata_dir = output_root / "_generated_metadata" / EXTRACTED_SOURCE_PREFIX
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{EXTRACTED_SOURCE_PREFIX}.csv"
    pd.DataFrame(rows).drop_duplicates("file_path").to_csv(metadata_path, index=False)
    try:
        metadata_source_dir = str(metadata_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        metadata_source_dir = str(metadata_dir)
    return {
        "prefix": EXTRACTED_SOURCE_PREFIX,
        "dir": metadata_source_dir,
        "metadata_csv": str(metadata_path),
    }


def discover_extracted_species(source_roots: list[Path]) -> list[str]:
    species_names: set[str] = set()
    for source_root in source_roots:
        if not source_root.exists():
            continue
        for species_dir in source_root.iterdir():
            if not species_dir.is_dir():
                continue
            species = parse_species_label(species_dir.name)
            if species:
                species_names.add(species)
    return sorted(species_names)


def configure_driams_marisma_rf_matrix(matrix: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    matrix = copy.deepcopy(matrix)
    matrix.setdefault("project", {})
    matrix.setdefault("datasets", {})
    training_cfg = matrix.setdefault("training", {})
    if str(args.benchmark_stage).upper() == "LOCAL_EXPERIMENTAL_PURES":
        matrix["project"]["benchmark_stage"] = "LOCAL_EXPERIMENTAL_PURES"
        matrix["datasets"]["training_origin_prefixes"] = []
        matrix["datasets"]["additional_pure_training_sources"] = []
        matrix["datasets"]["include_mixture_pures_in_training"] = True
        matrix["datasets"]["validation_mixture_roots"] = [str(path) for path in args.validation_mixture_roots]
        matrix["datasets"]["mixture_prefixes"] = [
            metadata_source_prefix_from_root(path)
            for path in args.validation_mixture_roots
        ]
        if getattr(args, "rf_n_estimators", None) is not None:
            training_cfg["n_estimators"] = int(args.rf_n_estimators)
        if getattr(args, "rf_negative_top_k", None) is not None:
            training_cfg["negative_top_k"] = int(args.rf_negative_top_k)
        return matrix

    pure_roots = list(args.rf_training_pure_roots)
    extracted_species = discover_extracted_species(pure_roots)
    if not extracted_species:
        raise RuntimeError(f"No species directories found in {pure_roots}")
    matrix["species"] = extracted_species
    matrix["project"]["benchmark_stage"] = args.benchmark_stage
    matrix["datasets"]["training_origin_prefixes"] = []
    matrix["datasets"]["additional_pure_training_sources"] = []
    matrix["datasets"]["include_mixture_pures_in_training"] = False
    matrix["datasets"]["validation_mixture_roots"] = [str(path) for path in args.validation_mixture_roots]
    matrix["datasets"]["mixture_prefixes"] = [
        metadata_source_prefix_from_root(path)
        for path in args.validation_mixture_roots
    ]
    max_per_species = args.rf_training_max_spectra_per_species_per_origin
    if max_per_species != "matrix":
        training_cfg["max_spectra_per_species_per_origin"] = (
            None
            if max_per_species is None or str(max_per_species).lower() in {"none", "null", "all"}
            else int(max_per_species)
        )
    if getattr(args, "rf_n_estimators", None) is not None:
        training_cfg["n_estimators"] = int(args.rf_n_estimators)
    if getattr(args, "rf_negative_top_k", None) is not None:
        training_cfg["negative_top_k"] = int(args.rf_negative_top_k)
    source = write_extracted_pure_metadata(pure_roots, args.output_dir, matrix["species"])
    if source is None:
        raise RuntimeError(f"No pure spectra found in {pure_roots}")
    matrix["datasets"]["additional_pure_training_sources"] = [source]
    return matrix


def spec_from_pipeline_name(pipeline_name: str, matrix: dict[str, Any]) -> dict[str, Any] | None:
    stage = matrix["project"].get("benchmark_stage", "custom")
    parts = pipeline_name.split("__")
    if parts and parts[0] == PIPELINE_GRID_PREFIX and len(parts) == 6:
        order_name, baseline, transform, binning, normalization = parts[1:]
        order_map = ALL_METHOD_GRID_ORDERS if order_name in ALL_METHOD_GRID_ORDERS else METHOD_GRID_ORDERS
        order_steps = order_map.get(order_name)
        if order_steps is None:
            step_aliases = {"VS": "T", "S": "S", "BC": "B", "N": "N"}
            raw_steps = order_name.split("-")
            valid_partial_order = (
                1 <= len(raw_steps) <= 4
                and len(set(raw_steps)) == len(raw_steps)
                and all(step in step_aliases for step in raw_steps)
            )
            if not valid_partial_order:
                return None
            order_steps = [step_aliases[step] for step in raw_steps]
        if order_steps is None:
            return None
        return {
            "pipeline": pipeline_name,
            "pipeline_id": pipeline_name,
            "pipeline_slug": pipeline_name,
            "benchmark_stage": stage,
            "order": order_name,
            "method_grid_order": order_name,
            "method_grid_baseline": baseline,
            "variance_stabilization": transform,
            "peak_selection": "none",
            "binning": binning,
            "normalization": normalization,
        }
    return None


def select_pipeline_specs(matrix: dict[str, Any], pipeline_names: list[str]) -> list[dict[str, Any]]:
    spec_by_pipeline: dict[str, dict[str, Any]] = {}
    for combo in iter_combinations(matrix):
        pipeline = str(combo["pipeline_slug"])
        spec_by_pipeline[pipeline] = {"pipeline": pipeline, "benchmark_stage": matrix["project"].get("benchmark_stage", "custom"), **combo}
    specs = []
    missing = []
    for name in pipeline_names:
        if name in spec_by_pipeline:
            specs.append(spec_by_pipeline[name])
            continue
        parsed = spec_from_pipeline_name(name, matrix)
        if parsed is not None:
            specs.append(parsed)
            continue
        missing.append(name)
    if missing:
        raise KeyError(f"Selected pipelines are not present in the benchmark matrix: {missing[:10]}")
    return specs


def load_pipeline_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    pipeline_col = "pipeline_id" if "pipeline_id" in frame.columns else "pipeline"
    if pipeline_col not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"Pipeline labels file must contain {pipeline_col!r} and 'label': {path}")
    return dict(zip(frame[pipeline_col].astype(str), frame["label"].astype(str)))


def load_evaluation_spots(matrix: dict[str, Any], pipeline_name: str, mz_axis: np.ndarray) -> pd.DataFrame:
    cfg = matrix["datasets"]
    base_dir = PROJECT_ROOT / cfg["pipeline_study_dir"]
    species = list(matrix["species"])
    rows = []
    for prefix in cfg.get("mixture_prefixes", []):
        h5_path = pipeline_h5_path(base_dir, prefix, pipeline_name)
        if not h5_path.exists():
            continue
        spectra, _, mz, metadata, _ = load_h5_records(h5_path, pure_only=False)
        if len(mz) != len(mz_axis) or not np.allclose(mz, mz_axis):
            raise ValueError(f"Mismatched m/z axis in evaluation file {h5_path}")
        metadata = metadata.copy()
        if "num_species" in metadata.columns:
            mixture_mask = metadata["num_species"].astype(int).gt(1).to_numpy()
            if not mixture_mask.any():
                continue
            metadata = metadata.loc[mixture_mask].reset_index(drop=True)
            spectra = spectra[mixture_mask]
        metadata["_spectrum_row"] = list(spectra)
        metadata["_row"] = np.arange(len(metadata))
        if "mixture_name" not in metadata.columns:
            metadata["mixture_name"] = metadata.get("sample_folder", metadata["_row"]).astype(str)
        metadata["spot_key"] = metadata.get("sample_folder", metadata["_row"]).map(spot_key)
        metadata["composition_key"] = [
            "|".join(sorted(str(x) for x in parse_vector(names))) + "::" + "|".join(str(x) for x in parse_vector(abundances))
            for names, abundances in zip(metadata["species_names"], metadata["abundances"])
        ]
        group_cols = ["composition_key", "mixture_name", "spot_key"]
        for _, group in metadata.groupby(group_cols, sort=False):
            first = group.iloc[0]
            names = [normalize_species_name(x) for x in parse_vector(first.get("species_names", []))]
            abundances = [float(x) for x in parse_vector(first.get("abundances", []))]
            if not abundances:
                abundances = [1.0] * len(names)
            abundance_by_species = {name: 0.0 for name in species}
            for name, abundance in zip(names, abundances):
                if name in abundance_by_species:
                    abundance_by_species[name] += abundance
            rows.append(
                {
                    "pipeline": pipeline_name,
                    "mixture_dataset": prefix,
                    "mixture_name": first.get("mixture_name", ""),
                    "spot_key": first.get("spot_key", ""),
                    "n_technical_replicates": int(len(group)),
                    "num_species": int(sum(value > 0 for value in abundance_by_species.values())),
                    "spectrum": np.vstack(group["_spectrum_row"].to_list()).mean(axis=0),
                    **{f"abundance_{name}": abundance_by_species[name] for name in species},
                }
            )
    if not rows:
        raise FileNotFoundError(f"No evaluation H5 files found for pipeline {pipeline_name}")
    return pd.DataFrame(rows)


def load_pure_validation_spots(matrix: dict[str, Any], pipeline_name: str, mz_axis: np.ndarray) -> pd.DataFrame:
    cfg = matrix["datasets"]
    base_dir = PROJECT_ROOT / cfg["pipeline_study_dir"]
    species = list(matrix["species"])
    rows = []
    for prefix in cfg.get("mixture_prefixes", []):
        h5_path = pipeline_h5_path(base_dir, prefix, pipeline_name)
        if not h5_path.exists():
            continue
        spectra, labels, mz, metadata, _ = load_h5_records(h5_path, pure_only=False)
        if len(mz) != len(mz_axis) or not np.allclose(mz, mz_axis):
            raise ValueError(f"Mismatched m/z axis in pure validation file {h5_path}")
        metadata = metadata.copy()
        if "num_species" not in metadata.columns:
            continue
        pure_mask = metadata["num_species"].astype(int).eq(1).to_numpy()
        if not pure_mask.any():
            continue
        metadata = metadata.loc[pure_mask].reset_index(drop=True)
        spectra = spectra[pure_mask]
        metadata["_spectrum_row"] = list(spectra)
        metadata["_row"] = np.arange(len(metadata))
        if "mixture_name" not in metadata.columns:
            metadata["mixture_name"] = metadata.get("sample_folder", metadata["_row"]).astype(str)
        metadata["spot_key"] = metadata.get("sample_folder", metadata["_row"]).map(spot_key)
        metadata["composition_key"] = [
            "|".join(sorted(str(x) for x in parse_vector(names))) + "::" + "|".join(str(x) for x in parse_vector(abundances))
            for names, abundances in zip(metadata["species_names"], metadata["abundances"])
        ]
        group_cols = ["composition_key", "mixture_name", "spot_key"]
        for _, group in metadata.groupby(group_cols, sort=False):
            first = group.iloc[0]
            names = [normalize_species_name(x) for x in parse_vector(first.get("species_names", []))]
            abundances = [float(x) for x in parse_vector(first.get("abundances", []))]
            if not abundances:
                abundances = [100.0] * len(names)
            abundance_by_species = {name: 0.0 for name in species}
            for name, abundance in zip(names, abundances):
                if name in abundance_by_species:
                    abundance_by_species[name] += abundance
            rows.append(
                {
                    "pipeline": pipeline_name,
                    "pure_dataset": prefix,
                    "mixture_dataset": prefix,
                    "mixture_name": first.get("mixture_name", ""),
                    "spot_key": first.get("spot_key", ""),
                    "n_technical_replicates": int(len(group)),
                    "num_species": 1,
                    "spectrum": np.vstack(group["_spectrum_row"].to_list()).mean(axis=0),
                    **{f"abundance_{name}": abundance_by_species[name] for name in species},
                }
            )
    return pd.DataFrame(rows)


def score_evaluation_spots(
    matrix: dict[str, Any],
    spots: pd.DataFrame,
    means: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    pure_score_matrix: np.ndarray,
    score_mode: str = "rf_weighted_cosine_baseline_adjusted",
    classifiers: dict[str, RandomForestClassifier] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    species = list(matrix["species"])
    spectra = np.vstack([np.asarray(value, dtype=float) for value in spots["spectrum"].to_list()])
    abundance_np = spots[[f"abundance_{target}" for target in species]].to_numpy(dtype=float)
    presence_threshold = float(
        matrix.get("evaluation", {}).get("abundance_presence_threshold", DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD)
    )
    truth_np = abundance_np > presence_threshold
    raw_scores_np = np.zeros((len(spots), len(species)), dtype=float)

    spectra_norm2 = np.sum(spectra * spectra, axis=1)
    for target_idx, target in enumerate(species):
        mean = means.get(target)
        if mean is None:
            continue
        if score_mode == "cosine":
            denom = np.sqrt(np.sum(mean * mean)) * np.sqrt(spectra_norm2)
            raw_scores_np[:, target_idx] = np.divide(
                spectra @ mean,
                denom,
                out=np.zeros(len(spectra), dtype=float),
                where=denom > 0,
            )
        else:
            weight = weights.get(target)
            if weight is None:
                continue
            denom = np.sqrt(np.sum(weight * mean * mean)) * np.sqrt((spectra * spectra) @ weight)
            raw_scores_np[:, target_idx] = np.divide(
                spectra @ (weight * mean),
                denom,
                out=np.zeros(len(spectra), dtype=float),
                where=denom > 0,
            )
    datasets_np = spots["mixture_dataset"].astype(str).to_numpy(dtype=object)
    if score_mode != "rf_weighted_cosine_baseline_adjusted":
        return truth_np, raw_scores_np, abundance_np
    adjusted, _, _ = baseline_adjusted_species_scores(
        raw_scores_np,
        abundance_np,
        datasets_np,
        pure_score_matrix,
        matrix,
    )
    return truth_np, adjusted, abundance_np


def threshold_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    datasets: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    datasets_np = np.asarray(datasets, dtype=object) if datasets is not None else None
    rows = []
    candidates = threshold_candidates_from_scores(scores.ravel()) if thresholds is None else np.asarray(thresholds, dtype=float)
    for threshold in candidates:
        if not np.isfinite(threshold):
            continue
        pred = scores >= float(threshold)
        row = {"threshold": float(threshold), **multilabel_metrics(truth, pred)}
        if datasets_np is not None:
            dataset_exact_values = []
            dataset_precision_values = []
            dataset_f1_values = []
            for dataset in sorted(set(datasets_np.tolist())):
                mask = datasets_np == dataset
                if not mask.any():
                    continue
                dataset_metrics = multilabel_metrics(truth[mask], pred[mask])
                dataset_exact_values.append(float(dataset_metrics["exact_match"]))
                dataset_precision_values.append(float(dataset_metrics["precision"]))
                dataset_f1_values.append(float(dataset_metrics["f1"]))
            row["mean_dataset_exact_match"] = mean_or_nan(dataset_exact_values)
            row["mean_dataset_precision"] = mean_or_nan(dataset_precision_values)
            row["mean_dataset_f1"] = mean_or_nan(dataset_f1_values)
        rows.append(row)
    return pd.DataFrame(rows)


def select_thresholds(metrics: pd.DataFrame, pure_metrics: pd.DataFrame | None = None) -> dict[str, float]:
    frame = metrics.copy()
    for col in ("exact_match", "precision", "recall", "f1", "mean_dataset_exact_match", "mean_dataset_precision", "mean_dataset_f1"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    exact_col = "mean_dataset_exact_match" if "mean_dataset_exact_match" in frame.columns else "exact_match"
    precision_col = "mean_dataset_precision" if "mean_dataset_precision" in frame.columns else "precision"
    f1_col = "mean_dataset_f1" if "mean_dataset_f1" in frame.columns else "f1"
    recall_col = "recall"
    selections = {}
    selections["best_exact_match"] = frame.sort_values(
        [exact_col, precision_col, f1_col, "threshold"],
        ascending=[False, False, False, True],
        na_position="last",
    ).iloc[0]["threshold"]
    selections["best_precision"] = frame.sort_values(
        [precision_col, f1_col, exact_col, "threshold"],
        ascending=[False, False, False, True],
        na_position="last",
    ).iloc[0]["threshold"]
    selections["best_f1"] = frame.sort_values(
        [f1_col, exact_col, precision_col, "threshold"],
        ascending=[False, False, False, True],
        na_position="last",
    ).iloc[0]["threshold"]
    selections["best_recall"] = frame.sort_values(
        [recall_col, precision_col, f1_col, exact_col, "threshold"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).iloc[0]["threshold"]
    if pure_metrics is not None and not pure_metrics.empty:
        pure_frame = pure_metrics.copy()
        for col in ("precision", "recall", "f1", "exact_match"):
            if col in pure_frame.columns:
                pure_frame[col] = pd.to_numeric(pure_frame[col], errors="coerce")
        max_pure_precision = pure_frame["precision"].max()
        pure_candidates = pure_frame.loc[pure_frame["precision"].eq(max_pure_precision)].copy()
        max_pure_recall = pure_candidates["recall"].max()
        pure_candidates = pure_candidates.loc[pure_candidates["recall"].eq(max_pure_recall)].copy()
        constrained = frame.merge(
            pure_candidates[["threshold", "precision", "recall"]],
            on="threshold",
            how="inner",
            suffixes=("", "_pure"),
        )
        if not constrained.empty:
            mixture_candidates = constrained.loc[constrained[precision_col].ge(0.95)].copy()
            if mixture_candidates.empty:
                max_mixture_precision = constrained[precision_col].max()
                mixture_candidates = constrained.loc[constrained[precision_col].eq(max_mixture_precision)].copy()
            selections["pure_precision_max_pure_recall_max_mixture_precision_095_or_max_mixture_recall_max"] = mixture_candidates.sort_values(
                [recall_col, exact_col, "threshold"],
                ascending=[False, False, True],
                na_position="last",
            ).iloc[0]["threshold"]
    return {key: float(value) for key, value in selections.items()}


def pair_sample_indices(n: int, max_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total = n * (n - 1) // 2
    if total <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    if total <= max_pairs:
        pairs = np.asarray([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    else:
        chosen = rng.choice(total, size=max_pairs, replace=False)
        pairs = []
        for flat in chosen:
            remaining = int(flat)
            for i in range(n - 1):
                row_count = n - i - 1
                if remaining < row_count:
                    pairs.append((i, i + 1 + remaining))
                    break
                remaining -= row_count
        pairs = np.asarray(pairs, dtype=int)
    return pairs[:, 0], pairs[:, 1]


def cosine_pair_values(vectors: np.ndarray, weights: np.ndarray | None, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) < 2:
        return np.asarray([], dtype=float)
    i, j = pair_sample_indices(len(vectors), max_pairs, rng)
    if len(i) == 0:
        return np.asarray([], dtype=float)
    if weights is None:
        norms = np.linalg.norm(vectors, axis=1)
        denom = norms[i] * norms[j]
        num = np.sum(vectors[i] * vectors[j], axis=1)
    else:
        w = np.asarray(weights, dtype=float)
        denom = np.sqrt(np.sum(w * vectors[i] * vectors[i], axis=1)) * np.sqrt(np.sum(w * vectors[j] * vectors[j], axis=1))
        num = np.sum(w * vectors[i] * vectors[j], axis=1)
    return np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=denom > 0)


def cosine_value(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is not None:
        return weighted_cosine(a, b, weights)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def top_cosine_negative_species(
    spectra: np.ndarray,
    labels: np.ndarray,
    species: list[str],
    top_k: int | None,
) -> dict[str, set[str]]:
    if top_k is None or int(top_k) <= 0:
        return {name: {other for other in species if other != name} for name in species}
    means = species_means(spectra, labels, species)
    negative_sets: dict[str, set[str]] = {}
    for focal in species:
        focal_mean = means.get(focal)
        if focal_mean is None:
            negative_sets[focal] = set()
            continue
        scored = [
            (cosine_value(focal_mean, means[other], None), other)
            for other in species
            if other != focal and other in means
        ]
        scored = [(score, other) for score, other in scored if np.isfinite(score)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        negative_sets[focal] = {other for _, other in scored[: int(top_k)]}
    return negative_sets


def similarity_summary(
    spectra: np.ndarray,
    labels: np.ndarray,
    species: list[str],
    weights: dict[str, np.ndarray] | None,
    max_pairs: int,
    random_state: int,
    prefix: str = "",
    workers: int = 1,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    means = {name: spectra[labels == name].mean(axis=0) for name in species if np.any(labels == name)}
    focal_species = [name for name in species if name in means]
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=len(focal_species), dtype=np.uint32)

    def summarize_focal(item: tuple[str, int]) -> tuple[float, float, float]:
        focal, seed = item
        if focal not in means:
            return float("nan"), float("nan"), float("nan")
        focal_rng = np.random.default_rng(int(seed))
        focal_weights = weights.get(focal) if weights is not None else None
        focal_vectors = spectra[labels == focal]
        pair_values = cosine_pair_values(focal_vectors, focal_weights, max_pairs, focal_rng)
        intra = float(np.nanmean(pair_values)) if pair_values.size else float("nan")
        inter_values = [
            cosine_value(means[focal], means[other], focal_weights)
            for other in species
            if other != focal and other in means
        ]
        inter_values = [value for value in inter_values if np.isfinite(value)]
        if inter_values:
            return intra, float(np.mean(inter_values)), float(np.max(inter_values))
        return intra, float("nan"), float("nan")

    tasks = list(zip(focal_species, seeds.tolist()))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            focal_summaries = list(executor.map(summarize_focal, tasks))
    else:
        focal_summaries = [summarize_focal(task) for task in tasks]

    intra_by_species = [row[0] for row in focal_summaries if np.isfinite(row[0])]
    inter_by_species = [row[1] for row in focal_summaries if np.isfinite(row[1])]
    closest_by_species = [row[2] for row in focal_summaries if np.isfinite(row[2])]
    return {
        f"{prefix}intra_similarity_mean": mean_or_nan(intra_by_species),
        f"{prefix}intra_similarity_std": std_or_nan(intra_by_species),
        f"{prefix}inter_similarity_mean": mean_or_nan(inter_by_species),
        f"{prefix}inter_similarity_std": std_or_nan(inter_by_species),
        f"{prefix}closest_inter_similarity_mean": mean_or_nan(closest_by_species),
        f"{prefix}closest_inter_similarity_std": std_or_nan(closest_by_species),
    }


def pipeline_parameter_row(spec: dict[str, Any]) -> dict[str, Any]:
    pipeline_id = strip_pipeline_prefix(str(spec["pipeline"]))
    return {
        "pipeline_id": pipeline_id,
        "pipeline_parameters": json.dumps(spec, sort_keys=True, default=str),
        "variance_stabilization": spec.get("variance_stabilization"),
        "smoothing_method": "savgol",
        "smoothing_window": 21,
        "baseline_correction": spec.get("method_grid_baseline", "snip"),
        "binning": spec.get("binning"),
        "normalization": spec.get("normalization"),
        "order": spec.get("order"),
        "train_dataset": spec.get("benchmark_stage"),
    }


def metrics_for_threshold(
    base_row: dict[str, Any],
    threshold: float,
    truth: np.ndarray,
    scores: np.ndarray,
    spots: pd.DataFrame,
    abundance_matrix: np.ndarray,
    species: list[str],
) -> dict[str, Any]:
    pred = scores >= float(threshold)
    species_index = {name: idx for idx, name in enumerate(species)}
    target_species = [name for name in TARGET_MIXTURE_SPECIES if name in species_index]
    row = dict(base_row)
    row["selected_threshold"] = float(threshold)
    row["threshold"] = float(threshold)
    global_metrics = multilabel_metrics(truth, pred)
    for key, value in global_metrics.items():
        row[f"global_{key}"] = value
    for key in ("exact_match", "precision", "recall", "f1", "mean_predicted_species", "n_samples"):
        global_key = f"global_{key}"
        if global_key in row:
            row[key] = row[global_key]

    datasets = spots["mixture_dataset"].astype(str).to_numpy()
    for dataset in sorted(set(datasets)):
        mask = datasets == dataset
        for key, value in multilabel_metrics(truth[mask], pred[mask]).items():
            row[f"dataset_{safe_slug(dataset)}_{key}"] = value

    species_counts = spots["num_species"].astype(int).to_numpy()
    for count in sorted(set(species_counts.tolist())):
        mask = species_counts == count
        for key, value in multilabel_metrics(truth[mask], pred[mask]).items():
            row[f"species_count_{count}_{key}"] = value

    for species_name in target_species:
        idx = species_index[species_name]
        metrics = binary_species_metrics(truth[:, idx], pred[:, idx])
        for key, value in metrics.items():
            row[f"species_{safe_slug(species_name)}_{key}"] = value

    for dataset in sorted(set(datasets)):
        if dataset == "local_mixture_set_b":
            continue
        dataset_mask = datasets == dataset
        for low, high in ABUNDANCE_BINS:
            occurrence_mask = np.zeros_like(truth, dtype=bool)
            for species_name in target_species:
                species_idx = species_index[species_name]
                abundance = abundance_percent_values(abundance_matrix[:, species_idx])
                occurrence_mask[:, species_idx] = dataset_mask & (abundance >= low) & (abundance < high) & truth[:, species_idx]
            n_occurrences = int(np.sum(occurrence_mask))
            tp = int(np.sum(occurrence_mask & pred))
            fn = int(np.sum(occurrence_mask & ~pred))
            recall = tp / n_occurrences if n_occurrences else float("nan")
            score_mean = float(np.mean(scores[occurrence_mask])) if n_occurrences else float("nan")
            prefix = f"abundance_{safe_slug(dataset)}_{low}_{high}"
            row[f"{prefix}_n_occurrences"] = n_occurrences
            row[f"{prefix}_tp"] = tp
            row[f"{prefix}_fn"] = fn
            row[f"{prefix}_recall"] = recall
            row[f"{prefix}_score_mean"] = score_mean
    return row


def prefixed_metrics_for_threshold(
    prefix: str,
    threshold: float,
    truth: np.ndarray,
    scores: np.ndarray,
    spots: pd.DataFrame,
    abundance_matrix: np.ndarray,
    species: list[str],
    include_abundance_bins: bool,
) -> dict[str, Any]:
    pred = scores >= float(threshold)
    species_index = {name: idx for idx, name in enumerate(species)}
    target_species = [name for name in TARGET_MIXTURE_SPECIES if name in species_index]
    row: dict[str, Any] = {}

    for key, value in multilabel_metrics(truth, pred).items():
        row[f"{prefix}_global_{key}"] = value

    dataset_col = "pure_dataset" if prefix == "pure" and "pure_dataset" in spots.columns else "mixture_dataset"
    datasets = spots[dataset_col].astype(str).to_numpy()
    for dataset in sorted(set(datasets)):
        mask = datasets == dataset
        for key, value in multilabel_metrics(truth[mask], pred[mask]).items():
            row[f"{prefix}_dataset_{safe_slug(dataset)}_{key}"] = value

    if prefix != "pure":
        species_counts = spots["num_species"].astype(int).to_numpy()
        for count in sorted(set(species_counts.tolist())):
            mask = species_counts == count
            for key, value in multilabel_metrics(truth[mask], pred[mask]).items():
                row[f"{prefix}_species_count_{count}_{key}"] = value

    for species_name in target_species:
        idx = species_index[species_name]
        for key, value in binary_species_metrics(truth[:, idx], pred[:, idx]).items():
            row[f"{prefix}_species_{safe_slug(species_name)}_{key}"] = value

    if include_abundance_bins:
        mixture_datasets = spots["mixture_dataset"].astype(str).to_numpy()
        for dataset in sorted(set(mixture_datasets)):
            if dataset == "local_mixture_set_b":
                continue
            dataset_mask = mixture_datasets == dataset
            for low, high in ABUNDANCE_BINS:
                occurrence_mask = np.zeros_like(truth, dtype=bool)
                for species_name in target_species:
                    species_idx = species_index[species_name]
                    abundance = abundance_percent_values(abundance_matrix[:, species_idx])
                    occurrence_mask[:, species_idx] = dataset_mask & (abundance >= low) & (abundance < high) & truth[:, species_idx]
                n_occurrences = int(np.sum(occurrence_mask))
                tp = int(np.sum(occurrence_mask & pred))
                fn = int(np.sum(occurrence_mask & ~pred))
                recall = tp / n_occurrences if n_occurrences else float("nan")
                score_mean = float(np.mean(scores[occurrence_mask])) if n_occurrences else float("nan")
                out_prefix = f"{prefix}_abundance_{safe_slug(dataset)}_{low}_{high}"
                row[f"{out_prefix}_n_occurrences"] = n_occurrences
                row[f"{out_prefix}_tp"] = tp
                row[f"{out_prefix}_fn"] = fn
                row[f"{out_prefix}_recall"] = recall
                row[f"{out_prefix}_score_mean"] = score_mean
    return row


def train_forests_with_progress(
    spectra: np.ndarray,
    labels: np.ndarray,
    origins: np.ndarray,
    mz_axis: np.ndarray,
    matrix: dict[str, Any],
    pipeline_name: str,
    pipeline_index: int,
    total_pipelines: int,
) -> tuple[dict[str, np.ndarray], dict[str, RandomForestClassifier], list[dict[str, Any]]]:
    training_cfg = matrix.get("training", {})
    species = list(matrix["species"])
    random_state = int(training_cfg.get("random_state", DEFAULT_RF_RANDOM_STATE))
    n_estimators = int(training_cfg.get("n_estimators", DEFAULT_RF_N_ESTIMATORS))
    min_class = int(training_cfg.get("min_samples_per_binary_class", DEFAULT_RF_MIN_SAMPLES_PER_BINARY_CLASS))
    negative_top_k = training_cfg.get("negative_top_k", None)
    negative_top_k = None if negative_top_k in (None, "none", "null", "all") else int(negative_top_k)
    negative_species = top_cosine_negative_species(spectra, labels, species, negative_top_k)
    weights: dict[str, np.ndarray] = {}
    classifiers: dict[str, RandomForestClassifier] = {}
    rf_summary_rows: list[dict[str, Any]] = []
    total_species = len(species)
    started = time.perf_counter()
    print(
        f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: "
        f"training {total_species} one-vs-rest RFs "
        f"({n_estimators} trees each, sklearn n_jobs=-1)",
        flush=True,
    )

    for species_index, species_name in enumerate(species, start=1):
        species_started = time.perf_counter()
        train_mask = (labels == species_name) | np.isin(labels, list(negative_species.get(species_name, set())))
        train_spectra = spectra[train_mask]
        train_origins = origins[train_mask]
        y = (labels[train_mask] == species_name).astype(int)
        counts = np.bincount(y, minlength=2)
        print(
            f"[pipeline {pipeline_index}/{total_pipelines}] "
            f"[RF {species_index}/{total_species}] start {species_name}: "
            f"positive={int(counts[1])}, negative={int(counts[0])}, "
            f"negative_species={len(negative_species.get(species_name, set()))}",
            flush=True,
        )
        if np.any(counts < min_class):
            weights[species_name] = np.ones(spectra.shape[1], dtype=np.float32)
            rf_summary_rows.append(
                {
                    "pipeline": pipeline_name,
                    "species": species_name,
                    "n_positive": int(counts[1]),
                    "n_negative": int(counts[0]),
                    "negative_top_k": negative_top_k,
                    "n_negative_species": int(len(negative_species.get(species_name, set()))),
                    "n_valid_folds": 0,
                    "balanced_accuracy": float("nan"),
                    "f1": float("nan"),
                    "average_precision": float("nan"),
                    "roc_auc": float("nan"),
                    "status": f"skipped_class_count_below_{min_class}",
                }
            )
            print(
                f"[pipeline {pipeline_index}/{total_pipelines}] "
                f"[RF {species_index}/{total_species}] skipped {species_name}: "
                f"class count below {min_class}",
                flush=True,
            )
            continue

        fold_importances = []
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(split_indices(y, train_origins, random_state), start=1):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state + fold_idx,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(train_spectra[train_idx], y[train_idx])
            pred = clf.predict(train_spectra[test_idx])
            proba = clf.predict_proba(train_spectra[test_idx])[:, 1]
            fold_importances.append(clf.feature_importances_)
            fold_metrics.append(
                {
                    "balanced_accuracy": safe_metric(balanced_accuracy_score, y[test_idx], pred),
                    "f1": safe_metric(f1_score, y[test_idx], pred),
                    "average_precision": safe_metric(average_precision_score, y[test_idx], proba),
                    "roc_auc": safe_metric(roc_auc_score, y[test_idx], proba),
                }
            )

        final_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        final_clf.fit(train_spectra, y)
        classifiers[species_name] = final_clf

        if fold_importances:
            mean_importance = np.mean(fold_importances, axis=0)
        else:
            mean_importance = final_clf.feature_importances_

        if not np.any(mean_importance > 0):
            mean_importance = np.ones_like(mean_importance)
        weights[species_name] = mean_importance.astype(np.float32)

        if fold_metrics:
            metric_means = {
                key: float(np.nanmean([row[key] for row in fold_metrics]))
                for key in RF_HOLDOUT_METRIC_NAMES
            }
            metric_text = ", ".join(
                f"{key}={value:.4f}"
                for key, value in metric_means.items()
            )
        else:
            metric_means = {
                "balanced_accuracy": float("nan"),
                "f1": float("nan"),
                "average_precision": float("nan"),
                "roc_auc": float("nan"),
            }
            metric_text = "holdout_metrics=nan"
        rf_summary_rows.append(
            {
                "pipeline": pipeline_name,
                "species": species_name,
                "n_positive": int(counts[1]),
                "n_negative": int(counts[0]),
                "negative_top_k": negative_top_k,
                "n_negative_species": int(len(negative_species.get(species_name, set()))),
                "n_valid_folds": int(len(fold_metrics)),
                **metric_means,
                "status": "ok" if fold_metrics else "trained_without_valid_holdout_fold",
            }
        )
        elapsed = time.perf_counter() - species_started
        print(
            f"[pipeline {pipeline_index}/{total_pipelines}] "
            f"[RF {species_index}/{total_species}] done {species_name} "
            f"in {elapsed:.1f}s; {metric_text}",
            flush=True,
        )

    elapsed_total = time.perf_counter() - started
    print(
        f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: "
        f"finished RF training in {elapsed_total:.1f}s",
        flush=True,
    )
    return weights, classifiers, rf_summary_rows


def evaluate_one_pipeline(
    spec: dict[str, Any],
    matrix: dict[str, Any],
    temp_root: Path,
    preprocess_workers: int,
    max_pairs: int,
    similarity_workers: int,
    score_modes: list[str],
    threshold_selection_metric: str,
    pipeline_index: int,
    total_pipelines: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pipeline_name = spec["pipeline"]
    pipeline_started = time.perf_counter()
    print(
        f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: generating temporary H5s",
        flush=True,
    )
    active_matrix = build_temporary_h5s_for_pipeline(
        spec,
        matrix,
        temp_root / safe_slug(pipeline_name),
        preprocess_workers=preprocess_workers,
    )
    print(
        f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: loading training data",
        flush=True,
    )
    spectra, labels, origins, mz_axis = load_training_data(active_matrix, pipeline_name)
    print(
        f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: "
        f"loaded {len(spectra)} training spectra, {len(np.unique(labels))} species, "
        f"{len(np.unique(origins))} origins",
        flush=True,
    )
    weights, classifiers, rf_summary_rows = train_forests_with_progress(
        spectra,
        labels,
        origins,
        mz_axis,
        active_matrix,
        pipeline_name,
        pipeline_index,
        total_pipelines,
    )
    print(
        f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: scoring mixtures",
        flush=True,
    )
    means = species_means(spectra, labels, active_matrix["species"])
    pure_score_matrix = pure_score_matrix_from_means(means, weights, active_matrix["species"])
    spots = load_evaluation_spots(active_matrix, pipeline_name, mz_axis)
    pure_spots = load_pure_validation_spots(active_matrix, pipeline_name, mz_axis)

    base_row = pipeline_parameter_row(spec)
    rf_summary_frame = pd.DataFrame(rf_summary_rows)
    if not rf_summary_frame.empty:
        for metric_name in RF_HOLDOUT_METRIC_NAMES:
            if metric_name in rf_summary_frame.columns:
                metric_mean = mean_or_nan(
                    pd.to_numeric(rf_summary_frame[metric_name], errors="coerce").tolist()
                )
                base_row[f"rf_{metric_name}_mean"] = metric_mean
                base_row[f"rf_holdout_{metric_name}_mean"] = metric_mean
    rows_by_criterion = {}
    for score_mode in score_modes:
        mode_started = time.perf_counter()
        print(
            f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: "
            f"score_mode={score_mode} scoring",
            flush=True,
        )
        truth, scores, abundance_matrix = score_evaluation_spots(
            active_matrix,
            spots,
            means,
            weights,
            pure_score_matrix,
            score_mode=score_mode,
            classifiers=classifiers,
        )
        if pure_spots.empty:
            pure_truth = pure_scores = pure_abundance_matrix = None
        else:
            pure_truth, pure_scores, pure_abundance_matrix = score_evaluation_spots(
                active_matrix,
                pure_spots,
                means,
                weights,
                pure_score_matrix,
                score_mode=score_mode,
                classifiers=classifiers,
            )
        print(
            f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: "
            f"score_mode={score_mode} threshold selection",
            flush=True,
        )
        threshold_table = threshold_metrics(truth, scores, spots["mixture_dataset"].astype(str).to_numpy())
        if pure_truth is not None and pure_scores is not None:
            constrained_thresholds = threshold_candidates_from_scores(
                np.concatenate([scores.ravel(), pure_scores.ravel()])
            )
            constrained_mixture_table = threshold_metrics(
                truth,
                scores,
                spots["mixture_dataset"].astype(str).to_numpy(),
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

        mode_base_row = dict(base_row)
        mode_base_row["score_mode"] = score_mode
        mode_base_row["mixture_prauc_global"] = global_prauc(truth, scores)
        mode_base_row["mixture_prauc_supported_species"] = supported_species_prauc(truth, scores)
        mode_base_row.update(dataset_prauc_metrics(truth, scores, spots["mixture_dataset"].astype(str).to_numpy(), prefix="mixture"))
        if "num_species" in spots.columns:
            mode_base_row.update(
                species_count_prauc_metrics(
                    truth,
                    scores,
                    spots["num_species"].astype(int).to_numpy(),
                    prefix="mixture",
                )
            )
        mode_base_row.update(species_prauc_metrics(truth, scores, active_matrix["species"], prefix="mixture"))
        if pure_truth is not None and pure_scores is not None:
            mode_base_row["pure_prauc_global"] = global_prauc(pure_truth, pure_scores)
            mode_base_row["pure_prauc_supported_species"] = supported_species_prauc(pure_truth, pure_scores)

        for criterion, threshold in selected.items():
            row = dict(mode_base_row)
            row["selected_threshold"] = float(threshold)
            row.update(
                prefixed_metrics_for_threshold(
                    "mixture",
                    threshold,
                    truth,
                    scores,
                    spots,
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
            key = f"{score_mode}::{criterion}"
            row["threshold_selection_metric"] = criterion
            rows_by_criterion[key] = row
        print(
            f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: "
            f"score_mode={score_mode} done in {time.perf_counter() - mode_started:.1f}s",
            flush=True,
        )
    print(
        f"[pipeline {pipeline_index}/{total_pipelines}] {pipeline_name}: "
        f"finished pipeline in {time.perf_counter() - pipeline_started:.1f}s",
        flush=True,
    )
    return rows_by_criterion, rf_summary_rows


def ordered_columns(rows_by_criterion: dict[str, list[dict[str, Any]]]) -> list[str]:
    all_cols: set[str] = set()
    for rows in rows_by_criterion.values():
        for row in rows:
            all_cols.update(row)
    return ordered_metric_columns(all_cols)


def canonicalize_output_rows(rows_by_criterion: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    similarity_columns = {
        "intra_similarity_mean",
        "intra_similarity_std",
        "inter_similarity_mean",
        "inter_similarity_std",
        "closest_inter_similarity_mean",
        "closest_inter_similarity_std",
        "weighted_intra_similarity_mean",
        "weighted_intra_similarity_std",
        "weighted_inter_similarity_mean",
        "weighted_inter_similarity_std",
        "weighted_closest_inter_similarity_mean",
        "weighted_closest_inter_similarity_std",
    }
    canonical: dict[str, list[dict[str, Any]]] = {}
    for criterion, rows in rows_by_criterion.items():
        canonical_rows = []
        for row in rows:
            out = dict(row)
            for column in list(out):
                if column in similarity_columns:
                    out[f"similarity_{column}"] = out.pop(column)
            for column in REDUNDANT_OUTPUT_COLUMNS:
                out.pop(column, None)
            for column in list(out):
                if (
                    column.startswith(EXCLUDED_OUTPUT_PREFIXES)
                    or column.endswith("_species_accuracy")
                    or column.endswith("_specificity")
                    or column.endswith("_false_positive_rate")
                ):
                    out.pop(column, None)
            canonical_rows.append(out)
        canonical[criterion] = canonical_rows
    return canonical


def write_outputs(rows_by_criterion: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".pipeline_outputs.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = load_existing_outputs(output_dir)
        merged = {
            criterion: [*existing.get(criterion, []), *rows]
            for criterion, rows in rows_by_criterion.items()
        }
        for criterion, rows in existing.items():
            merged.setdefault(criterion, rows)
        _write_outputs_unlocked(merged, output_dir)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_outputs_unlocked(rows_by_criterion: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    rows_by_criterion = canonicalize_output_rows(rows_by_criterion)
    columns = ordered_columns(rows_by_criterion)
    frames = []
    for rows in rows_by_criterion.values():
        if rows:
            frames.append(pd.DataFrame(rows))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if {"pipeline_id", "score_mode", "threshold_selection_metric"}.issubset(frame.columns):
        frame = frame.drop_duplicates(["pipeline_id", "score_mode", "threshold_selection_metric"], keep="last")
    elif {"pipeline_id", "threshold_selection_metric"}.issubset(frame.columns):
        frame = frame.drop_duplicates(["pipeline_id", "threshold_selection_metric"], keep="last")
    frame = frame.reindex(columns=columns)

    id_columns = [
        column
        for column in (
            "pipeline_id",
            "train_dataset",
            "score_mode",
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
        if column in frame.columns
    ]
    similarity_columns = [column for column in frame.columns if column.startswith("similarity_")]
    count_columns = [
        column
        for column in frame.columns
        if column in {"tp", "fp", "fn", "tn"}
        or column.endswith((
            "_tp",
            "_fp",
            "_fn",
            "_tn",
            "_n_spots",
            "_n_samples",
            "_n_occurrences",
            "_n_positive",
            "_n_negative",
        ))
    ]

    metric_columns = [
        column
        for column in frame.columns
        if column not in set(similarity_columns) | set(count_columns)
    ]
    frame.loc[:, metric_columns].to_csv(output_dir / "pipeline_metrics.csv", index=False)

    if count_columns:
        frame.loc[:, [*id_columns, *[column for column in count_columns if column not in id_columns]]].to_csv(
            output_dir / "pipeline_counts.csv",
            index=False,
        )
    if similarity_columns:
        similarity_frame = frame.loc[
            :,
            [
                *[column for column in id_columns if column not in {"score_mode", "threshold_selection_metric", "selected_threshold"}],
                *similarity_columns,
            ],
        ].drop_duplicates()
        if "pipeline_id" in similarity_frame.columns:
            similarity_frame = similarity_frame.drop_duplicates("pipeline_id", keep="last")
        similarity_frame.to_csv(output_dir / "similarity_metrics.csv", index=False)


def load_existing_outputs(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_criterion: dict[str, list[dict[str, Any]]] = {}
    combined = output_dir / "pipeline_metrics.csv"
    if combined.exists():
        frame = pd.read_csv(combined)
        for criterion in (
            "best_exact_match",
            "best_precision",
            "best_f1",
            "best_recall",
            "pure_precision_max_pure_recall_max_mixture_precision_095_or_max_mixture_recall_max",
        ):
            if "threshold_selection_metric" in frame.columns:
                rows_by_criterion[criterion] = frame.loc[
                    frame["threshold_selection_metric"].astype(str).eq(criterion)
                ].to_dict("records")
            else:
                rows_by_criterion[criterion] = []
        return rows_by_criterion
    for criterion in (
        "best_exact_match",
        "best_precision",
        "best_f1",
        "best_recall",
        "pure_precision_max_pure_recall_max_mixture_precision_095_or_max_mixture_recall_max",
    ):
        rows_by_criterion[criterion] = []
    return rows_by_criterion


def write_rf_holdout_metrics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".pipeline_rf_holdout_metrics.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _write_rf_holdout_metrics_unlocked([*load_existing_rf_holdout_metrics(output_dir), *rows], output_dir)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_rf_holdout_metrics_unlocked(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = [
        "pipeline",
        "species",
        "n_positive",
        "n_negative",
        "negative_top_k",
        "n_negative_species",
        "n_valid_folds",
        "balanced_accuracy",
        "f1",
        "average_precision",
        "roc_auc",
        "status",
    ]
    frame = pd.DataFrame(rows)
    if {"pipeline", "species"}.issubset(frame.columns):
        frame = frame.drop_duplicates(["pipeline", "species"], keep="last")
    frame = frame.reindex(columns=columns)
    frame.to_csv(output_dir / "pipeline_rf_holdout_metrics.csv", index=False)


def load_existing_rf_holdout_metrics(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "pipeline_rf_holdout_metrics.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def completed_pipeline_ids(
    rows_by_criterion: dict[str, list[dict[str, Any]]],
    score_modes: list[str],
) -> set[str]:
    by_pipeline: dict[str, set[str]] = {}
    for rows in rows_by_criterion.values():
        for row in rows:
            pipeline_id = row.get("pipeline_id")
            score_mode = row.get("score_mode")
            if pipeline_id is None or score_mode is None or pd.isna(pipeline_id) or pd.isna(score_mode):
                continue
            by_pipeline.setdefault(str(pipeline_id), set()).add(str(score_mode))
    required = set(score_modes)
    return {pipeline_id for pipeline_id, present in by_pipeline.items() if required.issubset(present)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp-root", type=Path, default=PROJECT_ROOT / "outputs/rf_preprocessing_benchmark/_tmp_rf_mixture_eval")
    parser.add_argument("--benchmark-stage", default=DEFAULT_BENCHMARK_STAGE)
    parser.add_argument("--rf-training-pure-roots", nargs="*", type=Path, default=DEFAULT_RF_TRAINING_PURE_ROOTS)
    parser.add_argument("--rf-training-max-spectra-per-species-per-origin", default="matrix")
    parser.add_argument("--rf-n-estimators", type=int, default=None)
    parser.add_argument("--rf-negative-top-k", type=int, default=None)
    parser.add_argument(
        "--score-modes",
        nargs="*",
        default=[
            "rf_weighted_cosine_baseline_adjusted",
            "rf_weighted_cosine",
            "cosine",
        ],
        help="Score variants to evaluate.",
    )
    parser.add_argument("--validation-mixture-roots", nargs="*", type=Path, default=DEFAULT_VALIDATION_MIXTURE_ROOTS)
    parser.add_argument(
        "--max-pipelines",
        type=int,
        default=None,
        help="Optional cap after ranking. Default evaluates every pipeline in the ranking/stage.",
    )
    parser.add_argument("--ranking-stage", default="local_mixtures")
    parser.add_argument("--pipeline-list", type=Path, default=None)
    parser.add_argument("--pipeline-binning-override", default=None)
    parser.add_argument("--pipeline-shard-index", type=int, default=None)
    parser.add_argument("--pipeline-num-shards", type=int, default=None)
    parser.add_argument("--pipelines", nargs="*", default=None)
    parser.add_argument(
        "--pipeline-labels",
        type=Path,
        default=None,
        help="Optional CSV with pipeline_id,label columns to print richer progress labels.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Worker processes for temporary H5 preprocessing. Defaults to all CPUs; use 1 to disable preprocessing parallelism.",
    )
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument(
        "--similarity-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Worker threads for per-species similarity summaries. Use 1 to disable similarity parallelism.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--threshold-selection",
        default=DEFAULT_THRESHOLD_SELECTION_METRIC,
        help=(
            "Single threshold-selection metric to keep in pipeline_metrics.csv. "
            f"Default: {DEFAULT_THRESHOLD_SELECTION_METRIC}."
        ),
    )
    args = parser.parse_args()
    coerce_path_args(
        args,
        ("matrix", "ranking", "pipeline_list", "output_dir", "temp_root", "pipeline_labels"),
        ("rf_training_pure_roots", "validation_mixture_roots"),
    )
    if args.similarity_workers < 1:
        raise ValueError("--similarity-workers must be >= 1")
    unknown_score_modes = sorted(set(args.score_modes) - VALID_SCORE_MODES)
    if unknown_score_modes:
        raise ValueError(f"Unknown --score-modes values: {unknown_score_modes}. Valid values: {sorted(VALID_SCORE_MODES)}")
    args.threshold_selection = normalize_threshold_selection(args.threshold_selection, source="--threshold-selection")

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
    specs = select_pipeline_specs(matrix, pipeline_names)
    pipeline_labels = load_pipeline_labels(args.pipeline_labels)

    rows_by_criterion = keep_threshold_selection(load_existing_outputs(args.output_dir), args.threshold_selection)
    completed = completed_pipeline_ids(rows_by_criterion, list(dict.fromkeys(args.score_modes)))
    errors = []
    rf_holdout_rows: list[dict[str, Any]] = load_existing_rf_holdout_metrics(args.output_dir)
    for index, spec in enumerate(specs, start=1):
        pipeline_temp_root = args.temp_root / safe_slug(spec["pipeline"])
        short_pipeline_id = pipeline_parameter_row(spec)["pipeline_id"]
        if short_pipeline_id in completed:
            print(f"[{index}/{len(specs)}] skipping completed {spec['pipeline']}", flush=True)
            continue
        label = pipeline_labels.get(str(spec["pipeline"]))
        if label:
            print(f"[{index}/{len(specs)}] evaluating {spec['pipeline']} | {label}", flush=True)
        else:
            print(f"[{index}/{len(specs)}] evaluating {spec['pipeline']}", flush=True)
        try:
            rows, rf_rows = evaluate_one_pipeline(
                spec,
                matrix,
                args.temp_root,
                args.preprocess_workers,
                args.max_pairs,
                args.similarity_workers,
                list(dict.fromkeys(args.score_modes)),
                args.threshold_selection,
                index,
                len(specs),
            )
        except Exception as exc:
            if pipeline_temp_root.exists():
                shutil.rmtree(pipeline_temp_root)
            if not args.continue_on_error:
                raise
            errors.append({"pipeline": spec["pipeline"], "error": repr(exc)})
            continue
        if pipeline_temp_root.exists():
            shutil.rmtree(pipeline_temp_root)
        for criterion, row in rows.items():
            rows_by_criterion.setdefault(criterion, []).append(row)
        rf_holdout_rows.extend(rf_rows)
        write_outputs(rows_by_criterion, args.output_dir)
        write_rf_holdout_metrics(rf_holdout_rows, args.output_dir)

    if errors:
        pd.DataFrame(errors).to_csv(args.output_dir / "pipeline_evaluation_errors.csv", index=False)
    write_outputs(rows_by_criterion, args.output_dir)
    write_rf_holdout_metrics(rf_holdout_rows, args.output_dir)


if __name__ == "__main__":
    main()
