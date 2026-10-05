#!/usr/bin/env python3
"""Run the MALDI RF weighted-cosine benchmark for preprocessed H5 pipelines."""

from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

LOCAL_CODE_DIR = Path(__file__).resolve().parents[2]
if str(LOCAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE_DIR))

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit

from benchmarking.core.metrics import (
    ABUNDANCE_BINS,
    ABUNDANCE_DATASET_SLUGS,
    ABUNDANCE_METRIC_NAMES,
    TARGET_MIXTURE_SPECIES,
    MIXTURE_DATASET_SLUGS,
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
)
from benchmarking.core.pipeline_manifest import DEFAULT_MATRIX, PROJECT_ROOT, iter_combinations

from helpers.SpectrumObject import SpectrumObject
from helpers import preprocess as preprocess_module
from models.rf_cosine import (
    DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD,
    DEFAULT_CROSS_SPECIES_BASELINE_METHOD,
    DEFAULT_RF_MIN_SAMPLES_PER_BINARY_CLASS,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_RF_RANDOM_STATE,
    DEFAULT_SUBTRACT_CROSS_SPECIES_TRAINING_BASELINE,
)
from data_generation.species_mapping import load_global_species_names, map_species_names_to_global_ids


SPECIES_LABELS = {
    "candida_albicans": "MALDI albicans",
    "candida_glabrata": "MALDI glabrata",
    "candida_parapsilosis": "MALDI parapsilosis",
    "candida_tropicalis": "MALDI tropicalis",
}
DEFAULT_MAX_SPECTRA_PER_SPECIES_PER_ORIGIN = None
GENERATION_MODE = 3
REFERENCE_PREPROCESSING_NAME = "reference"
_SPECIES_VOCAB_CACHE: list[str] | None = None
_METADATA_SOURCE_CACHE: dict[tuple[str, bool], pd.DataFrame] = {}
_INVALID_REFERENCE_FILE_CACHE: dict[tuple[str, str, str], set[str]] = {}
_WORKER_PREPROCESSOR = None
METHOD_GRID_ORDERS = {
    "VS-S-BC-N": ["T", "S", "B", "N"],
    "S-VS-BC-N": ["S", "T", "B", "N"],
    "S-BC-VS-N": ["S", "B", "T", "N"],
    "VS-S-N-BC": ["T", "S", "N", "B"],
    "VS-BC-S-N": ["T", "B", "S", "N"],
    "S-BC-N-VS": ["S", "B", "N", "T"],
}
ALL_METHOD_GRID_ORDERS = {
    "-".join(order): ["T" if step == "VS" else "S" if step == "S" else "B" if step == "BC" else "N" for step in order]
    for order in itertools.permutations(("VS", "S", "BC", "N"))
}
METHOD_GRID_BASELINES = ("snip", "tophat")
METHOD_GRID_TRANSFORMS = ("sqrt", "log2")
METHOD_GRID_BINNINGS = ("b3da_avg", "b6da_avg", "b9da_avg", "b2000ppm_avg")
PIPELINE_GRID_PREFIX = "pipeline_grid"
ABUNDANCE_BINS = [(low, low + 10) for low in range(10, 100, 10)]
MIXTURE_DATASET_SLUGS = (
    "local_mixture_set_a",
    "local_mixture_set_b",
    "local_mixture_set_c",
)
REDUNDANT_OUTPUT_COLUMNS = {
    "pipeline",
    "pipeline_slug",
    "baseline_method",
    "peak_picking_method",
    "binning_method",
    "normalization_method",
    "method_grid_order",
}
PIPELINE_EVALUATION_FILENAMES = {
    "best_exact_match": "pipeline_evaluation_best_exact_match.csv",
    "best_precision": "pipeline_evaluation_best_precision.csv",
    "best_f1": "pipeline_evaluation_best_f1.csv",
}
MEAN_DATASET_THRESHOLD_SELECTIONS = {
    "best_exact_match": "mean_dataset_exact_match",
    "best_precision": "mean_dataset_precision",
    "best_f1": "mean_dataset_f1",
}
CANONICAL_FINAL_METRICS = PIPELINE_EVALUATION_FILENAMES["best_f1"]
CANONICAL_RF_HOLDOUT_METRICS = "pipeline_rf_holdout_metrics.csv"
CANONICAL_ERRORS = "pipeline_evaluation_errors.csv"


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return data


def build_preprocessing_config(combo: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    pipeline_name = str(combo.get("pipeline", ""))
    if pipeline_name.startswith(f"{PIPELINE_GRID_PREFIX}__"):
        baseline_method = str(combo.get("method_grid_baseline", "snip")).lower()
        if baseline_method == "snip":
            baseline_params = {"method": "SNIP", "snip_n_iter": 20}
        elif baseline_method == "tophat":
            baseline_params = {"method": "TopHat", "tophat_window": 150}
        elif baseline_method == "convexhull":
            baseline_params = {"method": "ConvexHull"}
        elif baseline_method == "none":
            baseline_params = None
        else:
            raise ValueError(f"Unknown method_grid baseline: {baseline_method}")

        binning_name = str(combo["binning"])
        if binning_name == "b3da_avg":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 3, "aggregation": "mean"}
        elif binning_name == "b3da_sum":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 3, "aggregation": "sum"}
        elif binning_name == "b3da_max":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 3, "aggregation": "max"}
        elif binning_name == "b6da_avg":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 6, "aggregation": "mean"}
        elif binning_name == "b6da_sum":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 6, "aggregation": "sum"}
        elif binning_name == "b6da_max":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 6, "aggregation": "max"}
        elif binning_name == "b9da_avg":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 9, "aggregation": "mean"}
        elif binning_name == "b9da_sum":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 9, "aggregation": "sum"}
        elif binning_name == "b9da_max":
            binning_step = "Binner"
            binning_params = {"start": 2000, "stop": 20000, "step": 9, "aggregation": "max"}
        elif binning_name == "b2000ppm_avg":
            binning_step = "AdaptiveBinner"
            binning_params = {"start": 2000, "stop": 20000, "ppm_tolerance": 2000, "aggregation": "mean"}
        elif binning_name == "b2000ppm_sum":
            binning_step = "AdaptiveBinner"
            binning_params = {"start": 2000, "stop": 20000, "ppm_tolerance": 2000, "aggregation": "sum"}
        elif binning_name == "b2000ppm_max":
            binning_step = "AdaptiveBinner"
            binning_params = {"start": 2000, "stop": 20000, "ppm_tolerance": 2000, "aggregation": "max"}
        else:
            raise ValueError(f"Unknown method_grid binning: {binning_name}")

        pipeline = ["Trimmer"]
        params = {"Trimmer": {"min": 2000, "max": 20000}}
        for step in str(combo["method_grid_order"]).split("-"):
            if step == "VS":
                pipeline.append("VarStabilizer")
                params["VarStabilizer"] = {"method": combo["variance_stabilization"]}
            elif step == "S":
                pipeline.append("Smoother")
                params["Smoother"] = {"halfwindow": 10, "polyorder": 3}
            elif step == "BC":
                if baseline_params is None:
                    raise ValueError(f"method_grid_order contains BC but baseline is none for {pipeline_name}")
                pipeline.append("BaselineCorrecter")
                params["BaselineCorrecter"] = baseline_params
            elif step == "N":
                pipeline.append("Normalizer")
                params["Normalizer"] = {"sum": 1}
            else:
                raise ValueError(f"Unknown method_grid step: {step}")
        pipeline.append(binning_step)
        params[binning_step] = binning_params
        return {"pipeline": pipeline, "params": params}
    preprocessing = matrix["preprocessing"]
    fixed_key_by_step = {
        "Smoother": "smoothing",
        "BaselineCorrecter": "baseline",
        "Trimmer": "trimming",
    }
    pipeline = []
    params = {}
    for alias in preprocessing["orders"][combo["order"]]:
        if alias == "PeakSelection":
            spec = preprocessing["peak_selection"][combo["peak_selection"]]
            if spec is None:
                continue
        elif alias == "VarStabilizer":
            spec = preprocessing["variance_stabilization"][combo["variance_stabilization"]]
        elif alias == "Binning":
            spec = preprocessing["binning"][combo["binning"]]
        elif alias == "Normalization":
            spec = preprocessing["normalization"][combo["normalization"]]
        elif alias in fixed_key_by_step:
            spec = preprocessing["fixed"][fixed_key_by_step[alias]]
        else:
            raise KeyError(f"Unknown preprocessing alias: {alias}")
        step = spec["step"]
        pipeline.append(step)
        params[step] = spec.get("params", {})
    return {"pipeline": pipeline, "params": params}


def build_preprocessor(preprocess_cfg: dict[str, Any]):
    steps = []
    for step_name in preprocess_cfg["pipeline"]:
        cls = getattr(preprocess_module, step_name, None)
        if cls is None:
            raise ImportError(f"Preprocessing class '{step_name}' not found.")
        steps.append(cls(**preprocess_cfg["params"].get(step_name, {})))
    return preprocess_module.SequentialPreprocessor(*steps) if steps else None


def normalize_species_name(value: Any) -> str | None:
    value = parse_single_value(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value).strip().replace(" ", "_").lower()


def parse_single_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) else None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except ValueError:
        pass
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return parsed[0] if len(parsed) else None
        return parsed
    except (SyntaxError, ValueError):
        return text


def parse_vector(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, np.ndarray):
            return parsed.tolist()
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    except (SyntaxError, ValueError):
        inner = text.strip("[]")
        if not inner:
            return []
        return [part for part in inner.replace(",", " ").split() if part]


def species_names_from_ids(species_ids: np.ndarray, species_vocab: list[str]) -> np.ndarray:
    labels = []
    for species_id in species_ids:
        sid = int(species_id)
        labels.append(str(species_vocab[sid]) if 0 <= sid < len(species_vocab) else f"species_{sid}")
    return np.asarray(labels, dtype=object)


def default_origin_from_path(h5_path: Path) -> str:
    name = h5_path.stem
    for suffix in ("_pipeline5.1", "_pipeline5"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def load_h5_records(h5_path: Path, pure_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    csv_path = h5_path.with_suffix(".csv")
    with h5py.File(h5_path, "r") as handle:
        mz_axis = np.asarray(handle["unstructured/mz_axis"][:], dtype=np.float32)
        num_species = np.asarray(handle["0/num_species"][:], dtype=int)
        indices = np.where(num_species == 1)[0] if pure_only else np.arange(len(num_species))
        intensities = handle["0/intensity"]
        if len(indices):
            spectra = np.vstack([np.asarray(intensities[int(idx)], dtype=np.float32) for idx in indices])
        else:
            spectra = np.empty((0, len(mz_axis)), dtype=np.float32)
        config = json.loads(handle["unstructured/config_json"][0].decode())
        species_vocab = config.get("species_vocab", [])
        first_species_ids = np.asarray([int(np.asarray(handle["0/species_ids"][int(idx)])[0]) for idx in indices])

    if csv_path.exists():
        full_meta = pd.read_csv(csv_path)
        metadata = full_meta.iloc[indices].reset_index(drop=True)
    else:
        metadata = pd.DataFrame({"sample_index": indices})
    metadata["source_h5"] = h5_path.name
    if "origin" not in metadata.columns:
        metadata["origin"] = default_origin_from_path(h5_path)

    labels = species_names_from_ids(first_species_ids, species_vocab)
    if "species_names" in metadata.columns:
        csv_labels = metadata["species_names"].map(metadata_species_label).to_numpy(dtype=object)
        missing = pd.isna(csv_labels)
        if not missing.all():
            labels[~missing] = csv_labels[~missing]
    else:
        metadata["species_names"] = labels
    metadata["species_label"] = labels
    return spectra, labels.astype(str), mz_axis, metadata, first_species_ids


def pipeline_h5_path(base_dir: Path, prefix: str, pipeline_name: str) -> Path:
    return base_dir / f"{prefix}_{pipeline_name}.h5"


def load_species_vocab_from_reference() -> list[str]:
    raise RuntimeError("Species vocab must be provided by the active benchmark matrix.")


def set_h5torch_attrs(dataset, mode: str, filled_to: int, logical_shape, dtype_name: str) -> None:
    dataset.attrs["mode"] = mode
    dataset.attrs["filled_to"] = int(filled_to)
    dataset.attrs["shape"] = np.asarray(logical_shape, dtype=np.int64)
    dataset.attrs["dtypes"] = np.asarray([dtype_name, dtype_name], dtype=object)


def save_records_to_h5(records: list[dict[str, Any]], output_path: Path, species_vocab: list[str], preprocess_cfg: dict[str, Any]) -> None:
    if not records:
        raise RuntimeError(f"No records to save for {output_path}")
    reference_mz = np.asarray(records[0]["mz"], dtype=np.float32)
    for idx, record in enumerate(records):
        mz = np.asarray(record["mz"], dtype=np.float32)
        intensity = np.asarray(record["intensity"], dtype=np.float32)
        if mz.shape != reference_mz.shape or not np.allclose(mz, reference_mz):
            raise ValueError(f"Non-identical m/z axis while saving {output_path}, record {idx}")
        if not np.isfinite(intensity).all():
            raise ValueError(f"Non-finite intensity while saving {output_path}, record {idx}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        central = handle.create_dataset("central", data=np.arange(len(records), dtype=np.int64))
        set_h5torch_attrs(central, "N-D", len(records), [len(records)], "int64")
        unstructured = handle.create_group("unstructured")
        zero = handle.create_group("0")
        mz_axis = unstructured.create_dataset("mz_axis", data=reference_mz.astype(np.float32))
        set_h5torch_attrs(mz_axis, "N-D", reference_mz.shape[0], [reference_mz.shape[0]], "float32")

        intensity_values = np.empty(len(records), dtype=object)
        species_ids = np.empty(len(records), dtype=object)
        abundances = np.empty(len(records), dtype=object)
        sample_indices = np.empty(len(records), dtype=object)
        for idx, record in enumerate(records):
            intensity_values[idx] = np.asarray(record["intensity"], dtype=np.float32)
            species_ids[idx] = np.asarray(record["species_ids"], dtype=np.int32)
            abundances[idx] = np.asarray(record["abundances"], dtype=np.float32)
            sample_indices[idx] = np.asarray(record["sample_indices"], dtype=np.int32)

        ds = zero.create_dataset("intensity", data=intensity_values, dtype=h5py.vlen_dtype(np.float32))
        set_h5torch_attrs(ds, "vlen", len(records), [len(records)], "float32")
        ds = zero.create_dataset("species_ids", data=species_ids, dtype=h5py.vlen_dtype(np.int32))
        set_h5torch_attrs(ds, "vlen", len(records), [len(records)], "int32")
        ds = zero.create_dataset("abundances", data=abundances, dtype=h5py.vlen_dtype(np.float32))
        set_h5torch_attrs(ds, "vlen", len(records), [len(records)], "float32")
        ds = zero.create_dataset("sample_indices", data=sample_indices, dtype=h5py.vlen_dtype(np.int32))
        set_h5torch_attrs(ds, "vlen", len(records), [len(records)], "int32")

        values = np.asarray([record["num_species"] for record in records], dtype=np.int32)
        ds = zero.create_dataset("num_species", data=values)
        set_h5torch_attrs(ds, "N-D", len(records), [len(records)], str(np.dtype(np.int32)))

        config_json = json.dumps(
            {
                "species_vocab": species_vocab,
                "preprocessing": preprocess_cfg,
            }
        )
        unstructured.create_dataset("config_json", data=np.asarray([config_json], dtype=object), dtype=h5py.string_dtype("utf-8"))


def metadata_source_csv(prefix: str, candida: bool = False) -> Path:
    if candida:
        return PROJECT_ROOT / "data/mixtures/pipelines_study/raw_spectra" / f"{prefix}.csv"
    return PROJECT_ROOT / "data/manifests/mixtures" / f"{prefix}.csv"


def metadata_source_csv_in_dir(prefix: str, source_dir: str | Path) -> Path:
    return PROJECT_ROOT / source_dir / f"{prefix}.csv"


def metadata_source_prefix_from_root(path: str | Path) -> str:
    return Path(path).stem


def resolve_metadata_source_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metadata_source_root(path: str | Path) -> pd.DataFrame:
    path = resolve_metadata_source_root(path)
    key = (str(path), False)
    if key in _METADATA_SOURCE_CACHE:
        return _METADATA_SOURCE_CACHE[key].copy()
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata source CSV: {path}")
    metadata = pd.read_csv(path)
    _METADATA_SOURCE_CACHE[key] = metadata
    return metadata.copy()


def mixture_metadata_sources(matrix: dict[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    roots = matrix["datasets"].get("validation_mixture_roots")
    if roots:
        sources = []
        for root in roots:
            metadata_root = root["metadata_csv"] if isinstance(root, dict) else root
            name = root.get("name") if isinstance(root, dict) else None
            prefix = str(name or metadata_source_prefix_from_root(metadata_root))
            sources.append((prefix, load_metadata_source_root(metadata_root)))
        return sources
    return [
        (prefix, load_metadata_source(prefix, candida=False))
        for prefix in matrix["datasets"].get("mixture_prefixes", [])
    ]


def load_metadata_source(prefix: str, candida: bool = False) -> pd.DataFrame:
    key = (prefix, candida)
    if key in _METADATA_SOURCE_CACHE:
        return _METADATA_SOURCE_CACHE[key].copy()
    path = metadata_source_csv(prefix, candida=candida)
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata source CSV: {path}")
    metadata = pd.read_csv(path)
    _METADATA_SOURCE_CACHE[key] = metadata
    return metadata.copy()


def load_metadata_source_from_dir(prefix: str, source_dir: str | Path) -> pd.DataFrame:
    key = (prefix, str(source_dir))
    if key in _METADATA_SOURCE_CACHE:
        return _METADATA_SOURCE_CACHE[key].copy()
    path = metadata_source_csv_in_dir(prefix, source_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata source CSV: {path}")
    metadata = pd.read_csv(path)
    _METADATA_SOURCE_CACHE[key] = metadata
    return metadata.copy()


def load_metadata_source_from_config(source: dict[str, Any]) -> pd.DataFrame:
    if source.get("metadata_csv"):
        path = resolve_metadata_source_root(source["metadata_csv"])
        key = (str(path), False)
        if key in _METADATA_SOURCE_CACHE:
            return _METADATA_SOURCE_CACHE[key].copy()
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata source CSV: {path}")
        metadata = pd.read_csv(path)
        _METADATA_SOURCE_CACHE[key] = metadata
        return metadata.copy()
    return load_metadata_source_from_dir(source["prefix"], source["dir"])


def metadata_species_label(value: Any) -> str | None:
    values = parse_vector(value)
    if not values:
        return None
    return normalize_species_name(values[0])


def add_benchmark_species_ids(metadata: pd.DataFrame, species_vocab: list[str]) -> pd.DataFrame:
    if "species_names" not in metadata.columns:
        raise ValueError("Metadata must include species_names to write benchmark H5 species_ids.")
    remapped = metadata.copy()
    mapped_ids = []
    for row_idx, value in enumerate(remapped["species_names"]):
        names = [normalize_species_name(name) for name in parse_vector(value)]
        try:
            ids = map_species_names_to_global_ids(names)
        except ValueError as exc:
            raise ValueError(
                "Metadata species_names contain species outside the global species mapping "
                f"at row {row_idx}: {names}"
            ) from exc
        mapped_ids.append(json.dumps(ids))
    remapped["_benchmark_species_ids"] = mapped_ids
    return remapped


def benchmark_metadata_for_csv(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata.drop(columns=[column for column in ("_benchmark_species_ids",) if column in metadata.columns])


def sampled_candida_metadata_by_prefix(matrix: dict[str, Any]) -> dict[str, pd.DataFrame]:
    training_cfg = matrix.get("training", {})
    max_per_species = training_cfg.get(
        "max_spectra_per_species_per_origin",
        DEFAULT_MAX_SPECTRA_PER_SPECIES_PER_ORIGIN,
    )
    random_state = int(training_cfg.get("random_state", DEFAULT_RF_RANDOM_STATE))
    frames = []
    for prefix in matrix["datasets"]["training_origin_prefixes"]:
        metadata = load_metadata_source(prefix, candida=True)
        metadata["_source_prefix"] = prefix
        metadata["_species_for_sampling"] = metadata["species_names"].map(metadata_species_label)
        frames.append(metadata)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    if max_per_species is None:
        sampled = combined
    else:
        rng = np.random.default_rng(random_state)
        keep = []
        for _, idx in combined.groupby(["_source_prefix", "_species_for_sampling"], dropna=False).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            if len(idx_arr) > max_per_species:
                idx_arr = rng.choice(idx_arr, size=max_per_species, replace=False)
            keep.extend(idx_arr.tolist())
        sampled = combined.iloc[np.asarray(sorted(keep), dtype=int)].copy()
    result = {}
    for prefix in matrix["datasets"]["training_origin_prefixes"]:
        frame = sampled.loc[sampled["_source_prefix"].eq(prefix)].copy()
        frame = frame.drop(columns=["_source_prefix", "_species_for_sampling"])
        result[prefix] = frame.reset_index(drop=True)
    return result


def apply_benchmark_stage(matrix: dict[str, Any], stage_name: str | None) -> dict[str, Any]:
    matrix = copy.deepcopy(matrix)
    project = matrix.setdefault("project", {})
    matrix.setdefault("datasets", {})
    if not stage_name:
        project["benchmark_stage"] = "custom"
        return matrix
    stages = matrix.get("benchmark_stages", {})
    if not stages:
        project["benchmark_stage"] = stage_name
        return matrix
    if stage_name not in stages:
        available = ", ".join(stages) if stages else "none"
        raise KeyError(f"Unknown benchmark stage '{stage_name}'. Available stages: {available}")
    stage = stages[stage_name]
    datasets = matrix["datasets"]
    for key in (
        "training_origin_prefixes",
        "additional_pure_training_sources",
        "mixture_prefixes",
        "include_mixture_pures_in_training",
    ):
        if key in stage:
            datasets[key] = copy.deepcopy(stage[key])
    project["benchmark_stage"] = stage_name
    return matrix


def sampled_additional_pure_metadata(matrix: dict[str, Any]) -> dict[str, pd.DataFrame]:
    training_cfg = matrix.get("training", {})
    max_per_species = training_cfg.get(
        "max_spectra_per_species_per_origin",
        DEFAULT_MAX_SPECTRA_PER_SPECIES_PER_ORIGIN,
    )
    random_state = int(training_cfg.get("random_state", DEFAULT_RF_RANDOM_STATE))
    result = {}
    for source in matrix["datasets"].get("additional_pure_training_sources", []):
        prefix = source["prefix"]
        metadata = load_metadata_source_from_config(source)
        invalid_paths = invalid_reference_file_paths(prefix, source["dir"])
        if invalid_paths and "file_path" in metadata.columns:
            metadata = metadata.loc[~metadata["file_path"].astype(str).isin(invalid_paths)].copy()
        metadata = metadata.reset_index(drop=True)
        metadata["_species_for_sampling"] = metadata["species_names"].map(metadata_species_label)
        if max_per_species is not None:
            rng = np.random.default_rng(random_state)
            keep = []
            for _, idx in metadata.groupby("_species_for_sampling", dropna=False).groups.items():
                idx_arr = np.asarray(list(idx), dtype=int)
                if len(idx_arr) > max_per_species:
                    idx_arr = rng.choice(idx_arr, size=max_per_species, replace=False)
                keep.extend(idx_arr.tolist())
            metadata = metadata.iloc[np.asarray(sorted(keep), dtype=int)].copy()
        result[prefix] = metadata.drop(columns=["_species_for_sampling"]).reset_index(drop=True)
    return result


def invalid_reference_file_paths(prefix: str, source_dir: str | Path, reference_pipeline: str = REFERENCE_PREPROCESSING_NAME) -> set[str]:
    """Return source file paths whose reference H5 pure spectra are unusable."""
    key = (prefix, str(source_dir), reference_pipeline)
    if key in _INVALID_REFERENCE_FILE_CACHE:
        return set(_INVALID_REFERENCE_FILE_CACHE[key])

    h5_path = PROJECT_ROOT / source_dir / f"{prefix}_{reference_pipeline}.h5"
    if not h5_path.exists():
        _INVALID_REFERENCE_FILE_CACHE[key] = set()
        return set()

    spectra, _, _, metadata, _ = load_h5_records(h5_path, pure_only=True)
    if "file_path" not in metadata.columns or spectra.size == 0:
        _INVALID_REFERENCE_FILE_CACHE[key] = set()
        return set()

    finite_mask = np.all(np.isfinite(spectra), axis=1)
    signal_mask = np.nansum(np.abs(spectra), axis=1) > 0
    invalid = set(metadata.loc[~(finite_mask & signal_mask), "file_path"].astype(str))
    _INVALID_REFERENCE_FILE_CACHE[key] = invalid
    return set(invalid)


def init_preprocess_worker(preprocess_cfg: dict[str, Any]) -> None:
    global _WORKER_PREPROCESSOR
    _WORKER_PREPROCESSOR = build_preprocessor(preprocess_cfg)


def preprocess_row_dict(row: dict[str, Any], preprocessor) -> dict[str, Any]:
    spectrum = SpectrumObject.load(str(row["file_path"]))
    if preprocessor is not None:
        spectrum = preprocessor(spectrum)
    species_ids = np.asarray([int(x) for x in parse_vector(row["_benchmark_species_ids"])], dtype=np.int32)
    abundances = np.asarray([float(x) for x in parse_vector(row["abundances"])], dtype=np.float32)
    return {
        "mz": np.asarray(spectrum.mz, dtype=np.float32),
        "intensity": np.asarray(spectrum.intensity, dtype=np.float32),
        "species_ids": species_ids,
        "abundances": abundances,
        "sample_indices": np.full(len(species_ids), -1, dtype=np.int32),
        "num_species": int(row["num_species"]),
    }


def preprocess_row_worker(row: dict[str, Any]) -> dict[str, Any]:
    return preprocess_row_dict(row, _WORKER_PREPROCESSOR)


def preprocess_metadata_rows(
    metadata: pd.DataFrame,
    preprocessor=None,
    executor: ProcessPoolExecutor | None = None,
    chunksize: int = 16,
) -> list[dict[str, Any]]:
    rows = metadata.to_dict("records")
    if executor is not None:
        return list(executor.map(preprocess_row_worker, rows, chunksize=chunksize))
    return [preprocess_row_dict(row, preprocessor) for row in rows]


def build_temporary_h5s_for_pipeline(
    spec: dict[str, Any],
    matrix: dict[str, Any],
    temp_root: Path,
    preprocess_workers: int = 1,
) -> dict[str, Any]:
    pipeline_name = spec["pipeline"]
    preprocess_cfg = build_preprocessing_config(spec, matrix)
    preprocessor = None if preprocess_workers > 1 else build_preprocessor(preprocess_cfg)
    species_vocab = load_global_species_names()
    temp_root.mkdir(parents=True, exist_ok=True)
    mixture_dir = temp_root / "pipelines_study"
    candida_dir = mixture_dir / "raw_spectra"
    pure_sources = list(matrix["datasets"].get("additional_pure_training_sources", []))
    mixture_sources = mixture_metadata_sources(matrix)

    def cached_pair_exists(path: Path) -> bool:
        csv_path = path.with_suffix(".csv")
        return path.exists() and csv_path.exists()

    sampled_candida_metadata = sampled_candida_metadata_by_prefix(matrix)
    sampled_extra_metadata = sampled_additional_pure_metadata(
        {**matrix, "datasets": {**matrix["datasets"], "additional_pure_training_sources": pure_sources}}
    )
    missing_work = False
    for prefix in matrix["datasets"]["training_origin_prefixes"]:
        metadata = sampled_candida_metadata[prefix]
        if metadata.empty:
            continue
        if not cached_pair_exists(candida_dir / f"{prefix}_{pipeline_name}.h5"):
            missing_work = True
            break
    if not missing_work:
        for source in pure_sources:
            prefix = source["prefix"]
            metadata = sampled_extra_metadata.get(prefix, pd.DataFrame())
            if metadata.empty:
                continue
            output_dir = temp_root / "additional_pure_training_sources" / safe_slug(prefix)
            if not cached_pair_exists(output_dir / f"{prefix}_{pipeline_name}.h5"):
                missing_work = True
                break
    if not missing_work:
        for prefix, metadata in mixture_sources:
            if metadata.empty:
                continue
            if not cached_pair_exists(mixture_dir / f"{prefix}_{pipeline_name}.h5"):
                missing_work = True
                break

    executor = (
        ProcessPoolExecutor(
            max_workers=preprocess_workers,
            initializer=init_preprocess_worker,
            initargs=(preprocess_cfg,),
        )
        if missing_work and preprocess_workers > 1
        else None
    )

    try:
        for prefix in matrix["datasets"]["training_origin_prefixes"]:
            metadata = sampled_candida_metadata[prefix]
            if metadata.empty:
                continue
            output_h5 = candida_dir / f"{prefix}_{pipeline_name}.h5"
            if cached_pair_exists(output_h5):
                continue
            metadata = add_benchmark_species_ids(metadata, species_vocab)
            records = preprocess_metadata_rows(metadata, preprocessor, executor=executor)
            save_records_to_h5(records, output_h5, species_vocab, preprocess_cfg)
            benchmark_metadata_for_csv(metadata).to_csv(output_h5.with_suffix(".csv"), index=False)

        for source in pure_sources:
            prefix = source["prefix"]
            metadata = sampled_extra_metadata.get(prefix, pd.DataFrame())
            if metadata.empty:
                continue
            output_dir = temp_root / "additional_pure_training_sources" / safe_slug(prefix)
            output_h5 = output_dir / f"{prefix}_{pipeline_name}.h5"
            if cached_pair_exists(output_h5):
                continue
            metadata = add_benchmark_species_ids(metadata, species_vocab)
            records = preprocess_metadata_rows(metadata, preprocessor, executor=executor)
            save_records_to_h5(records, output_h5, species_vocab, preprocess_cfg)
            benchmark_metadata_for_csv(metadata).to_csv(output_h5.with_suffix(".csv"), index=False)

        for prefix, metadata in mixture_sources:
            output_h5 = mixture_dir / f"{prefix}_{pipeline_name}.h5"
            if cached_pair_exists(output_h5):
                continue
            metadata = add_benchmark_species_ids(metadata, species_vocab)
            records = preprocess_metadata_rows(metadata, preprocessor, executor=executor)
            save_records_to_h5(records, output_h5, species_vocab, preprocess_cfg)
            benchmark_metadata_for_csv(metadata).to_csv(output_h5.with_suffix(".csv"), index=False)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    temp_matrix = copy.deepcopy(matrix)
    temp_matrix["datasets"]["pipeline_study_dir"] = str(mixture_dir)
    temp_matrix["datasets"]["raw_spectra_dir"] = str(candida_dir)
    temp_matrix["datasets"]["mixture_prefixes"] = [prefix for prefix, _metadata in mixture_sources]
    temp_matrix["datasets"].pop("validation_mixture_roots", None)
    temp_matrix["datasets"]["additional_pure_training_sources"] = [
        {**source, "dir": str(temp_root / "additional_pure_training_sources" / safe_slug(source["prefix"]))}
        for source in matrix["datasets"].get("additional_pure_training_sources", [])
    ]
    return temp_matrix


def load_training_data(matrix: dict[str, Any], pipeline_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = matrix["datasets"]
    candida_dir = PROJECT_ROOT / cfg["raw_spectra_dir"]
    mixture_dir = PROJECT_ROOT / cfg["pipeline_study_dir"]
    aggregate = cfg.get("aggregate_training_origins", {})
    species_keep = set(matrix["species"])
    loaded = []
    mz_axis = None

    training_sources = [
        (candida_dir, prefix)
        for prefix in cfg["training_origin_prefixes"]
    ]
    training_sources.extend(
        (PROJECT_ROOT / source["dir"], source["prefix"])
        for source in cfg.get("additional_pure_training_sources", [])
    )
    if cfg.get("include_mixture_pures_in_training", True):
        training_sources.extend((mixture_dir, prefix) for prefix in cfg.get("mixture_prefixes", []))

    for base_dir, prefix in training_sources:
        path = pipeline_h5_path(base_dir, prefix, pipeline_name)
        if not path.exists():
            continue
        spectra, labels, mz, metadata, _ = load_h5_records(path, pure_only=True)
        mask = np.asarray([label in species_keep for label in labels])
        if not mask.any():
            continue
        if mz_axis is None:
            mz_axis = mz
        elif len(mz_axis) != len(mz) or not np.allclose(mz_axis, mz):
            raise ValueError(f"Mismatched m/z axis in {path}")
        origins = metadata["origin"].astype(str).map(lambda x: aggregate.get(x, x)).to_numpy()
        loaded.append((spectra[mask], labels[mask], origins[mask]))
    if not loaded:
        raise FileNotFoundError(f"No training H5 files found for pipeline '{pipeline_name}'")
    return (
        np.vstack([item[0] for item in loaded]),
        np.concatenate([item[1] for item in loaded]),
        np.concatenate([item[2] for item in loaded]),
        mz_axis,
    )


def cap_training_data(
    spectra: np.ndarray,
    labels: np.ndarray,
    origins: np.ndarray,
    max_per_species_origin: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_per_species_origin is None:
        return spectra, labels, origins
    rng = np.random.default_rng(random_state)
    keep: list[int] = []
    frame = pd.DataFrame({"species": labels, "origin": origins})
    for _, idx in frame.groupby(["origin", "species"]).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) > max_per_species_origin:
            idx_arr = rng.choice(idx_arr, size=max_per_species_origin, replace=False)
        keep.extend(idx_arr.tolist())
    keep_idx = np.asarray(sorted(keep), dtype=int)
    return spectra[keep_idx], labels[keep_idx], origins[keep_idx]


def split_indices(y: np.ndarray, groups: np.ndarray, random_state: int):
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        splitter = LeaveOneGroupOut()
        yield from splitter.split(np.zeros_like(y), y, groups)
        return
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    yield from splitter.split(np.zeros_like(y), y)


def safe_metric(metric_fn, y_true, y_pred_or_score) -> float:
    try:
        return float(metric_fn(y_true, y_pred_or_score))
    except ValueError:
        return float("nan")


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def std_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.std(finite, ddof=0)) if finite else float("nan")


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


def train_forests(
    spectra: np.ndarray,
    labels: np.ndarray,
    origins: np.ndarray,
    mz_axis: np.ndarray,
    matrix: dict[str, Any],
    pipeline_name: str,
    collect_importances: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    training_cfg = matrix.get("training", {})
    species = matrix["species"]
    benchmark_stage = matrix.get("project", {}).get("benchmark_stage", "custom")
    random_state = int(training_cfg.get("random_state", DEFAULT_RF_RANDOM_STATE))
    n_estimators = int(training_cfg.get("n_estimators", DEFAULT_RF_N_ESTIMATORS))
    min_class = int(training_cfg.get("min_samples_per_binary_class", DEFAULT_RF_MIN_SAMPLES_PER_BINARY_CLASS))
    weights: dict[str, np.ndarray] = {}
    metric_rows = []
    importance_rows = []

    for species_name in species:
        y = (labels == species_name).astype(int)
        counts = np.bincount(y, minlength=2)
        if np.any(counts < min_class):
            weights[species_name] = np.ones(spectra.shape[1], dtype=np.float32)
            continue

        fold_importances = []
        for fold_idx, (train_idx, test_idx) in enumerate(split_indices(y, origins, random_state), start=1):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state + fold_idx,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(spectra[train_idx], y[train_idx])
            pred = clf.predict(spectra[test_idx])
            proba = clf.predict_proba(spectra[test_idx])[:, 1]
            fold_importances.append(clf.feature_importances_)
            metric_rows.append(
                {
                    "benchmark_stage": benchmark_stage,
                    "pipeline": pipeline_name,
                    "species": species_name,
                    "fold": fold_idx,
                    "test_origin": ",".join(sorted(set(origins[test_idx]))),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "balanced_accuracy": safe_metric(balanced_accuracy_score, y[test_idx], pred),
                    "f1": safe_metric(f1_score, y[test_idx], pred),
                    "average_precision": safe_metric(average_precision_score, y[test_idx], proba),
                    "roc_auc": safe_metric(roc_auc_score, y[test_idx], proba),
                }
            )

        if fold_importances:
            mean_importance = np.mean(fold_importances, axis=0)
            std_importance = np.std(fold_importances, axis=0)
        else:
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(spectra, y)
            mean_importance = clf.feature_importances_
            std_importance = np.zeros_like(mean_importance)

        if not np.any(mean_importance > 0):
            mean_importance = np.ones_like(mean_importance)
        weights[species_name] = mean_importance.astype(np.float32)
        if collect_importances:
            total = float(mean_importance.sum())
            ranked = np.argsort(-mean_importance)
            ranks = np.empty_like(ranked)
            ranks[ranked] = np.arange(1, len(ranked) + 1)
            for idx, (mz, imp, std) in enumerate(zip(mz_axis, mean_importance, std_importance)):
                importance_rows.append(
                    {
                        "pipeline": pipeline_name,
                        "species": species_name,
                        "bin_index": idx,
                        "mz": float(mz),
                        "importance_mean": float(imp),
                        "importance_std": float(std),
                        "importance_fraction": float(imp / total) if total > 0 else 0.0,
                        "rank": int(ranks[idx]),
                    }
                )

    return weights, pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def summarize_rf_holdout(rf_metrics: pd.DataFrame, pipeline_name: str) -> pd.DataFrame:
    if rf_metrics.empty:
        return pd.DataFrame([{"pipeline": pipeline_name}])
    metric_cols = ["balanced_accuracy", "f1", "average_precision", "roc_auc"]
    group_cols = ["pipeline", "species"]
    if "benchmark_stage" in rf_metrics.columns:
        group_cols = ["benchmark_stage", *group_cols]
    summary = (
        rf_metrics.groupby(group_cols, dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .rename(columns={col: f"rf_holdout_{col}_mean" for col in metric_cols})
    )
    overall = {"pipeline": pipeline_name, "species": "ALL_SPECIES_MEAN"}
    if "benchmark_stage" in rf_metrics.columns:
        overall["benchmark_stage"] = str(rf_metrics["benchmark_stage"].iloc[0])
    for col in metric_cols:
        overall[f"rf_holdout_{col}_mean"] = float(rf_metrics[col].mean())
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


def weighted_cosine(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    numerator = float(np.sum(weights * a * b))
    denom = float(np.sqrt(np.sum(weights * a * a)) * np.sqrt(np.sum(weights * b * b)))
    return numerator / denom if denom > 0 else 0.0


def species_means(spectra: np.ndarray, labels: np.ndarray, species: list[str]) -> dict[str, np.ndarray]:
    return {name: spectra[labels == name].mean(axis=0) for name in species if np.any(labels == name)}


def training_cross_species_baseline(means: dict[str, np.ndarray], weights: dict[str, np.ndarray], species: list[str]) -> dict[tuple[str, str], float]:
    baseline = {}
    for target in species:
        for interferer in species:
            if target == interferer or target not in means or interferer not in means:
                baseline[(target, interferer)] = 0.0
            else:
                baseline[(target, interferer)] = weighted_cosine(means[target], means[interferer], weights[target])
    return baseline


def pure_score_matrix_from_means(
    means: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    species: list[str],
) -> np.ndarray:
    scores = np.zeros((len(species), len(species)), dtype=float)
    for column_idx, column_species in enumerate(species):
        for target_idx, target_species in enumerate(species):
            if target_species in means and column_species in means:
                scores[column_idx, target_idx] = weighted_cosine(
                    means[target_species],
                    means[column_species],
                    weights[target_species],
                )
    return scores


def estimate_mixture_abundances_from_scores(
    raw_score_matrix: np.ndarray,
) -> np.ndarray:
    """Estimate per-sample species proportions from weighted-cosine evidence only."""
    evidence = np.clip(np.asarray(raw_score_matrix, dtype=float), 0.0, None)
    totals = evidence.sum(axis=1, keepdims=True)
    estimated = np.divide(evidence, totals, out=np.zeros_like(evidence), where=totals > 0)
    return estimated


def normalize_abundance_rows(abundance_matrix: np.ndarray) -> np.ndarray:
    abundance_matrix = np.clip(np.asarray(abundance_matrix, dtype=float), 0.0, None)
    totals = abundance_matrix.sum(axis=1, keepdims=True)
    return np.divide(abundance_matrix, totals, out=np.zeros_like(abundance_matrix), where=totals > 0)


def abundance_percent_values(abundance: np.ndarray) -> np.ndarray:
    abundance = np.asarray(abundance, dtype=float)
    finite = abundance[np.isfinite(abundance)]
    if finite.size and np.nanmax(finite) <= 1.0:
        return abundance * 100.0
    return abundance


def baseline_adjusted_species_scores(
    raw_score_matrix: np.ndarray,
    known_abundance_matrix: np.ndarray,
    datasets: np.ndarray,
    pure_score_matrix: np.ndarray,
    matrix: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract pure-derived cross-species baseline using score-estimated abundances for every mixture."""
    species = matrix["species"]
    evaluation_cfg = matrix.get("evaluation", {})
    abundance_threshold = float(
        evaluation_cfg.get("abundance_presence_threshold", DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD)
    )

    estimated_abundance = estimate_mixture_abundances_from_scores(raw_score_matrix)
    apply_abundance = estimated_abundance

    pure_abundance = np.eye(len(species), dtype=float)
    fit_scores = pure_score_matrix
    fit_abundance = pure_abundance
    apply_for_prediction = apply_abundance
    predicted_baseline = np.zeros_like(raw_score_matrix, dtype=float)

    for target_idx, _target in enumerate(species):
        interferer_indices = [idx for idx in range(len(species)) if idx != target_idx]
        target_absent_fit = (
            (fit_abundance[:, target_idx] <= abundance_threshold)
            & np.isfinite(fit_scores[:, target_idx])
        )
        design = fit_abundance[target_absent_fit][:, interferer_indices]
        observed = fit_scores[target_absent_fit, target_idx]
        if design.size == 0 or observed.size == 0:
            coefficients = np.zeros(len(interferer_indices), dtype=float)
        else:
            coefficients, _ = nnls(design, observed)
        predicted_baseline[:, target_idx] = apply_for_prediction[:, interferer_indices] @ coefficients

    adjusted = np.clip(raw_score_matrix - predicted_baseline, 0.0, None)
    return adjusted, apply_abundance, predicted_baseline


def threshold_candidates_from_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    values = np.unique(values[np.isfinite(values)])
    if values.size == 0:
        return np.asarray([np.nan], dtype=float)
    candidates = [max(0.0, float(values[0]) - 1e-12)]
    if values.size > 1:
        candidates.extend(((values[:-1] + values[1:]) / 2.0).tolist())
    candidates.append(float(values[-1]) + 1e-12)
    return np.asarray(candidates, dtype=float)


def exact_match_metrics_at_threshold(truth_matrix: np.ndarray, score_matrix: np.ndarray, threshold: float) -> dict[str, Any]:
    truth_matrix = np.asarray(truth_matrix, dtype=bool)
    score_matrix = np.asarray(score_matrix, dtype=float)
    finite_row_mask = np.all(np.isfinite(score_matrix), axis=1)
    truth_matrix = truth_matrix[finite_row_mask]
    score_matrix = score_matrix[finite_row_mask]
    if score_matrix.size == 0:
        return {
            "exact_match": float("nan"),
            "species_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mean_predicted_species": float("nan"),
            "mean_true_species": float("nan"),
            "n_spots": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    pred = score_matrix >= float(threshold)
    tp = int(np.sum(pred & truth_matrix))
    tn = int(np.sum(~pred & ~truth_matrix))
    fp = int(np.sum(pred & ~truth_matrix))
    fn = int(np.sum(~pred & truth_matrix))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else float("nan")
    return {
        "exact_match": float(np.mean(np.all(pred == truth_matrix, axis=1))),
        "species_accuracy": float((tp + tn) / truth_matrix.size),
        "precision": float(precision) if np.isfinite(precision) else float("nan"),
        "recall": float(recall) if np.isfinite(recall) else float("nan"),
        "f1": float(f1) if np.isfinite(f1) else float("nan"),
        "mean_predicted_species": float(np.mean(pred.sum(axis=1))),
        "mean_true_species": float(np.mean(truth_matrix.sum(axis=1))),
        "n_spots": int(truth_matrix.shape[0]),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def spot_key(sample_folder: Any) -> str:
    parts = [part for part in str(sample_folder).replace("\\", "/").split("/") if part]
    for part in parts:
        if re.fullmatch(r"\d+_[A-H]\d+", part):
            return part
    return str(sample_folder)


def canonical_composition_key(species_names: Any, abundances: Any) -> str:
    names = [normalize_species_name(x) for x in parse_vector(species_names)]
    values = [float(x) for x in parse_vector(abundances)]
    parts = [
        (name, abundance)
        for name, abundance in zip(names, values)
        if name is not None
    ]
    parts.sort(key=lambda item: item[0])
    return "|".join(f"{name}:{abundance:.12g}" for name, abundance in parts)


def load_mixture_spots(matrix: dict[str, Any], pipeline_name: str, mz_axis: np.ndarray) -> pd.DataFrame:
    cfg = matrix["datasets"]
    base_dir = PROJECT_ROOT / cfg["pipeline_study_dir"]
    species_keep = matrix["species"]
    frames = []
    for prefix in cfg.get("mixture_prefixes", []):
        path = pipeline_h5_path(base_dir, prefix, pipeline_name)
        if not path.exists():
            continue
        spectra, _, mz, metadata, _ = load_h5_records(path, pure_only=False)
        if len(mz) != len(mz_axis) or not np.allclose(mz, mz_axis):
            raise ValueError(f"Mismatched m/z axis in mixture file {path}")
        metadata = metadata.copy()
        mixture_mask = metadata["num_species"].astype(int).gt(1).to_numpy()
        if not mixture_mask.any():
            continue
        metadata = metadata.loc[mixture_mask].reset_index(drop=True)
        spectra = spectra[mixture_mask]
        metadata["row_index"] = np.arange(len(metadata))
        metadata["spot_key"] = metadata.get("sample_folder", metadata["row_index"]).map(spot_key)
        metadata["composition_key"] = [
            canonical_composition_key(species_names, abundances)
            for species_names, abundances in zip(metadata["species_names"], metadata["abundances"])
        ]
        metadata["mixture_dataset"] = prefix
        metadata["_spectrum_row"] = list(spectra)
        frames.append(metadata)
    if not frames:
        raise FileNotFoundError(f"No mixture H5 files found for pipeline '{pipeline_name}'")

    all_rows = pd.concat(frames, ignore_index=True)
    spot_rows = []
    grouping_cols = ["mixture_dataset", "composition_key", "mixture_name", "spot_key"]
    for (dataset, composition_key, mixture_name, spot), group in all_rows.groupby(grouping_cols, sort=False):
        spectra_stack = np.vstack(group["_spectrum_row"].to_list())
        first = group.iloc[0]
        names = [normalize_species_name(x) for x in parse_vector(first.get("species_names", []))]
        abundances_raw = parse_vector(first.get("abundances", []))
        abundances = [float(x) for x in abundances_raw] if abundances_raw else [1.0] * len(names)
        abundance_by_species = {name: 0.0 for name in species_keep}
        for name, abundance in zip(names, abundances):
            if name in abundance_by_species:
                abundance_by_species[name] += float(abundance)
        spot_rows.append(
            {
                "pipeline": pipeline_name,
                "mixture_dataset": dataset,
                "composition_key": composition_key,
                "spot_key": spot,
                "mixture_name": mixture_name,
                "evaluation_unit": "dataset_composition_spot",
                "n_technical_replicates": int(len(group)),
                "num_species": int(sum(value > 0 for value in abundance_by_species.values())),
                "expected_species": "|".join([name for name in species_keep if abundance_by_species[name] > 0]),
                "spectrum": spectra_stack.mean(axis=0),
                **{f"abundance_{name}": abundance_by_species[name] for name in species_keep},
            }
        )
    return pd.DataFrame(spot_rows)


def load_pure_validation_spots(matrix: dict[str, Any], pipeline_name: str, mz_axis: np.ndarray) -> pd.DataFrame:
    cfg = matrix["datasets"]
    base_dir = PROJECT_ROOT / cfg["pipeline_study_dir"]
    species_keep = matrix["species"]
    rows = []
    for prefix in cfg.get("mixture_prefixes", []):
        path = pipeline_h5_path(base_dir, prefix, pipeline_name)
        if not path.exists():
            continue
        spectra, _, mz, metadata, _ = load_h5_records(path, pure_only=False)
        if len(mz) != len(mz_axis) or not np.allclose(mz, mz_axis):
            raise ValueError(f"Mismatched m/z axis in pure validation file {path}")
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
            abundance_by_species = {name: 0.0 for name in species_keep}
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
                    **{f"abundance_{name}": abundance_by_species[name] for name in species_keep},
                }
            )
    return pd.DataFrame(rows)


def score_evaluation_spots(
    matrix: dict[str, Any],
    spots: pd.DataFrame,
    means: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    pure_score_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    species = list(matrix["species"])
    raw_scores = []
    abundances = []
    truth = []
    for _, row in spots.iterrows():
        spectrum = np.asarray(row["spectrum"], dtype=float)
        raw_scores.append(
            [
                weighted_cosine(means[target], spectrum, weights[target])
                if target in means and target in weights
                else 0.0
                for target in species
            ]
        )
        abundances.append([float(row[f"abundance_{target}"]) for target in species])
        presence_threshold = float(
            matrix.get("evaluation", {}).get("abundance_presence_threshold", DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD)
        )
        truth.append([float(row[f"abundance_{target}"]) > presence_threshold for target in species])
    raw_scores_np = np.asarray(raw_scores, dtype=float)
    abundance_np = np.asarray(abundances, dtype=float)
    datasets_np = spots["mixture_dataset"].astype(str).to_numpy(dtype=object)
    adjusted, _, _ = baseline_adjusted_species_scores(
        raw_scores_np,
        abundance_np,
        datasets_np,
        pure_score_matrix,
        matrix,
    )
    return np.asarray(truth, dtype=bool), adjusted, abundance_np


def evaluate_mixtures(
    pipeline_name: str,
    mixture_spots: pd.DataFrame,
    means: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    pure_score_matrix: np.ndarray,
    matrix: dict[str, Any],
    collect_diagnostics: bool = False,
    threshold_selection: str = "exact_match",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    species = matrix["species"]
    evaluation_cfg = matrix.get("evaluation", {})
    subtract_baseline = bool(
        evaluation_cfg.get(
            "subtract_cross_species_training_baseline",
            DEFAULT_SUBTRACT_CROSS_SPECIES_TRAINING_BASELINE,
        )
    )
    presence_threshold = float(
        evaluation_cfg.get("abundance_presence_threshold", DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD)
    )
    raw_score_matrix = []
    known_abundance_matrix = []
    truth_matrix = []
    datasets = []

    for _, row in mixture_spots.iterrows():
        spectrum = row["spectrum"]
        raw_scores = []
        truth = []
        for target in species:
            raw = weighted_cosine(means[target], spectrum, weights[target]) if target in means else 0.0
            raw_scores.append(raw)
            truth.append(float(row[f"abundance_{target}"]) > presence_threshold)
        raw_score_matrix.append(raw_scores)
        known_abundance_matrix.append([float(row[f"abundance_{target}"]) for target in species])
        truth_matrix.append(truth)
        datasets.append(str(row["mixture_dataset"]))

    raw_score_matrix_np = np.asarray(raw_score_matrix, dtype=float)
    abundance_matrix_np = np.asarray(known_abundance_matrix, dtype=float)
    datasets_np = np.asarray(datasets, dtype=object)
    baseline_abundance_matrix = normalize_abundance_rows(abundance_matrix_np)
    predicted_baseline_matrix = np.zeros_like(raw_score_matrix_np, dtype=float)
    score_matrix_np = raw_score_matrix_np.copy()
    if subtract_baseline:
        baseline_method = evaluation_cfg.get("baseline_method", DEFAULT_CROSS_SPECIES_BASELINE_METHOD)
        if baseline_method == "nonnegative_ridge":
            score_matrix_np, baseline_abundance_matrix, predicted_baseline_matrix = baseline_adjusted_species_scores(
                raw_score_matrix_np,
                abundance_matrix_np,
                datasets_np,
                pure_score_matrix,
                matrix,
            )
        else:
            baseline_abundance_matrix = estimate_mixture_abundances_from_scores(raw_score_matrix_np)
            for target_idx, target in enumerate(species):
                predicted_baseline = np.zeros(score_matrix_np.shape[0], dtype=float)
                for interferer_idx, interferer in enumerate(species):
                    if interferer == target:
                        continue
                    predicted_baseline += baseline_abundance_matrix[:, interferer_idx] * pure_score_matrix[interferer_idx, target_idx]
                predicted_baseline_matrix[:, target_idx] = predicted_baseline
                score_matrix_np[:, target_idx] = np.clip(raw_score_matrix_np[:, target_idx] - predicted_baseline, 0.0, None)

    truth_matrix_np = np.asarray(truth_matrix, dtype=bool)
    prauc_global = global_prauc(truth_matrix_np, score_matrix_np)
    threshold_free_metrics = {
        "mixture_prauc_supported_species": supported_species_prauc(truth_matrix_np, score_matrix_np),
        **dataset_prauc_metrics(truth_matrix_np, score_matrix_np, datasets_np, prefix="mixture"),
        **species_prauc_metrics(truth_matrix_np, score_matrix_np, species, prefix="mixture"),
    }
    if "num_species" in mixture_spots.columns:
        threshold_free_metrics.update(
            species_count_prauc_metrics(
                truth_matrix_np,
                score_matrix_np,
                mixture_spots["num_species"].astype(int).to_numpy(),
                prefix="mixture",
            )
        )
    thresholds = threshold_candidates_from_scores(score_matrix_np.ravel())
    if threshold_selection == "mean_dataset":
        selection_names = tuple(MEAN_DATASET_THRESHOLD_SELECTIONS.values())
    else:
        selection_names = (threshold_selection,)
    best_by_selection: dict[str, tuple[tuple[float, ...], dict[str, Any], np.ndarray]] = {}
    for threshold in thresholds:
        metrics = exact_match_metrics_at_threshold(truth_matrix_np, score_matrix_np, float(threshold))
        pred = score_matrix_np >= threshold
        exact = np.all(pred == truth_matrix_np, axis=1)
        dataset_metrics = {}
        dataset_exact_values = []
        dataset_precision_values = []
        dataset_f1_values = []
        for dataset in sorted(set(datasets)):
            dataset_mask = datasets_np == dataset
            if dataset_mask.any():
                subset_metrics = exact_match_metrics_at_threshold(
                    truth_matrix_np[dataset_mask],
                    score_matrix_np[dataset_mask],
                    float(threshold),
                )
                dataset_metrics[f"exact_match_{dataset}"] = float(np.mean(exact[dataset_mask]))
                for metric_name, metric_value in subset_metrics.items():
                    dataset_metrics[f"dataset_{safe_slug(dataset)}_{metric_name}"] = metric_value
                    if metric_name == "n_spots":
                        dataset_metrics[f"dataset_{safe_slug(dataset)}_n_samples"] = metric_value
                dataset_exact_values.append(float(subset_metrics["exact_match"]))
                dataset_precision_values.append(float(subset_metrics["precision"]))
                dataset_f1_values.append(float(subset_metrics["f1"]))
        species_count_metrics = {}
        if "num_species" in mixture_spots.columns:
            species_counts = mixture_spots["num_species"].astype(int).to_numpy()
            for count in sorted(set(species_counts.tolist())):
                count_mask = species_counts == count
                if count_mask.any():
                    subset_metrics = exact_match_metrics_at_threshold(
                        truth_matrix_np[count_mask],
                        score_matrix_np[count_mask],
                        float(threshold),
                    )
                    for metric_name, metric_value in subset_metrics.items():
                        species_count_metrics[f"species_count_{count}_{metric_name}"] = metric_value
                        if metric_name == "n_spots":
                            species_count_metrics[f"species_count_{count}_n_samples"] = metric_value
        species_index = {name: idx for idx, name in enumerate(species)}
        target_species = [name for name in TARGET_MIXTURE_SPECIES if name in species_index]
        species_binary_metrics = {}
        for species_name in target_species:
            species_idx = species_index[species_name]
            binary_metrics = binary_species_metrics(truth_matrix_np[:, species_idx], pred[:, species_idx])
            for metric_name, metric_value in binary_metrics.items():
                species_binary_metrics[f"species_{safe_slug(species_name)}_{metric_name}"] = metric_value
        abundance_metrics = {}
        for dataset in sorted(set(datasets)):
            if dataset == "local_mixture_set_b":
                continue
            dataset_mask = datasets_np == dataset
            for low, high in ABUNDANCE_BINS:
                occurrence_mask = np.zeros_like(truth_matrix_np, dtype=bool)
                for species_name in target_species:
                    species_idx = species_index[species_name]
                    abundance = abundance_percent_values(abundance_matrix_np[:, species_idx])
                    occurrence_mask[:, species_idx] = (
                        dataset_mask
                        & (abundance >= low)
                        & (abundance < high)
                        & truth_matrix_np[:, species_idx]
                    )
                n_occurrences = int(np.sum(occurrence_mask))
                tp_occurrences = int(np.sum(occurrence_mask & pred))
                fn_occurrences = int(np.sum(occurrence_mask & ~pred))
                recall_occurrences = tp_occurrences / n_occurrences if n_occurrences else float("nan")
                score_mean = float(np.mean(score_matrix_np[occurrence_mask])) if n_occurrences else float("nan")
                prefix = f"abundance_{safe_slug(dataset)}_{low}_{high}"
                abundance_metrics[f"{prefix}_n_occurrences"] = n_occurrences
                abundance_metrics[f"{prefix}_tp"] = tp_occurrences
                abundance_metrics[f"{prefix}_fn"] = fn_occurrences
                abundance_metrics[f"{prefix}_recall"] = recall_occurrences
                abundance_metrics[f"{prefix}_score_mean"] = score_mean
        summary = {
            "pipeline": pipeline_name,
            "pipeline_id": pipeline_name,
            "threshold": float(threshold),
            "selected_threshold": float(threshold),
            "prauc_global": prauc_global,
            **threshold_free_metrics,
            "mean_dataset_exact_match": mean_or_nan(dataset_exact_values),
            "mean_dataset_precision": mean_or_nan(dataset_precision_values),
            "mean_dataset_f1": mean_or_nan(dataset_f1_values),
            "n_estimated_abundance_spots": int(len(pred)),
            **metrics,
            **dataset_metrics,
            **species_count_metrics,
            **species_binary_metrics,
            **abundance_metrics,
        }
        precision_key = summary["precision"] if np.isfinite(summary["precision"]) else -np.inf
        recall_key = summary["recall"] if np.isfinite(summary["recall"]) else -np.inf
        f1_key = summary["f1"] if np.isfinite(summary["f1"]) else -np.inf
        mean_exact_key = summary["mean_dataset_exact_match"] if np.isfinite(summary["mean_dataset_exact_match"]) else -np.inf
        mean_precision_key = summary["mean_dataset_precision"] if np.isfinite(summary["mean_dataset_precision"]) else -np.inf
        mean_f1_key = summary["mean_dataset_f1"] if np.isfinite(summary["mean_dataset_f1"]) else -np.inf
        keys_by_selection = {
            "exact_match": (summary["exact_match"], precision_key, recall_key, summary["threshold"]),
            "f1": (f1_key, summary["exact_match"], precision_key, recall_key, -summary["threshold"]),
            "precision": (precision_key, recall_key, f1_key, summary["exact_match"], -summary["threshold"]),
            "recall": (recall_key, precision_key, f1_key, summary["exact_match"], -summary["threshold"]),
            "mean_dataset_exact_match": (mean_exact_key, mean_precision_key, mean_f1_key, -summary["threshold"]),
            "mean_dataset_precision": (mean_precision_key, mean_f1_key, mean_exact_key, -summary["threshold"]),
            "mean_dataset_f1": (mean_f1_key, mean_exact_key, mean_precision_key, -summary["threshold"]),
        }
        for selection_name in selection_names:
            if selection_name not in keys_by_selection:
                raise ValueError(f"Unknown threshold_selection: {threshold_selection}")
            key = keys_by_selection[selection_name]
            if selection_name not in best_by_selection or key > best_by_selection[selection_name][0]:
                selected_summary = dict(summary)
                selected_summary["threshold_selection_metric"] = selection_name
                best_by_selection[selection_name] = (key, selected_summary, pred)

    assert best_by_selection
    best_rows = [best_by_selection[name][1] for name in selection_names if name in best_by_selection]
    if not collect_diagnostics:
        return pd.DataFrame(best_rows), pd.DataFrame()
    abundance_diagnostics = pd.DataFrame(
        {
            "pipeline": pipeline_name,
            "mixture_dataset": datasets_np,
            **{f"known_abundance_{species_name}": abundance_matrix_np[:, idx] for idx, species_name in enumerate(species)},
            **{f"baseline_abundance_{species_name}": baseline_abundance_matrix[:, idx] for idx, species_name in enumerate(species)},
            **{f"predicted_baseline_{species_name}": predicted_baseline_matrix[:, idx] for idx, species_name in enumerate(species)},
        }
    )
    return pd.DataFrame(best_rows), abundance_diagnostics


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def first_existing_csv(path: Path, filenames: tuple[str, ...]) -> Path:
    if path.is_file():
        return path
    for filename in filenames:
        candidate = path / filename
        if candidate.exists():
            return candidate
    return path / filenames[0]


def clear_previous_result_csvs(output_root: Path) -> None:
    filenames = (
        *PIPELINE_EVALUATION_FILENAMES.values(),
        CANONICAL_RF_HOLDOUT_METRICS,
        CANONICAL_ERRORS,
    )
    for filename in filenames:
        path = output_root / filename
        if path.exists():
            path.unlink()


def clear_stale_temp_h5s(output_root: Path) -> None:
    temp_root = output_root / "_tmp_h5"
    if temp_root.exists():
        shutil.rmtree(temp_root)


def compact_final_metrics(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(summaries)
    alias_pairs = {
        "pipeline_id": "pipeline",
        "selected_threshold": "threshold",
        "global_exact_match": "exact_match",
        "global_precision": "precision",
        "global_recall": "recall",
        "global_f1": "f1",
        "global_mean_predicted_species": "mean_predicted_species",
        "global_mean_true_species": "mean_true_species",
        "global_n_samples": "n_spots",
        "global_species_accuracy": "species_accuracy",
        "baseline_method": "method_grid_baseline",
        "binning_method": "binning",
        "normalization_method": "normalization",
        "peak_picking_method": "peak_selection",
        "rf_balanced_accuracy_mean": "rf_holdout_balanced_accuracy_mean",
        "rf_f1_mean": "rf_holdout_f1_mean",
        "rf_average_precision_mean": "rf_holdout_average_precision_mean",
        "rf_roc_auc_mean": "rf_holdout_roc_auc_mean",
    }
    for target, source in alias_pairs.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    if "pipeline_parameters" not in frame.columns and not frame.empty:
        frame["pipeline_parameters"] = frame.apply(
            lambda row: json.dumps(
                {
                    key: row[key]
                    for key in (
                        "benchmark_stage",
                        "method_grid_family",
                        "method_grid_baseline",
                        "order",
                        "variance_stabilization",
                        "binning",
                        "normalization",
                        "peak_selection",
                    )
                    if key in row.index and pd.notna(row[key])
                },
                sort_keys=True,
                default=str,
            ),
            axis=1,
        )
    if "smoothing_method" not in frame.columns:
        frame["smoothing_method"] = "savgol"
    if "smoothing_window" not in frame.columns:
        frame["smoothing_window"] = 21
    rename: dict[str, str] = {}
    for column in frame.columns:
        if column == "prauc_global":
            rename[column] = "mixture_prauc_global"
        elif column.startswith("global_"):
            rename[column] = f"mixture_{column}"
        elif column in {"exact_match", "precision", "recall", "f1", "mean_predicted_species", "n_spots", "species_accuracy"}:
            metric = "n_samples" if column == "n_spots" else column
            rename[column] = f"mixture_global_{metric}"
        elif column.startswith("exact_match_") and safe_slug(column.removeprefix("exact_match_")) in MIXTURE_DATASET_SLUGS:
            rename[column] = f"mixture_dataset_{safe_slug(column.removeprefix('exact_match_'))}_exact_match"
        elif column.startswith("dataset_"):
            rename[column] = f"mixture_{column}"
        elif column.startswith("species_count_"):
            rename[column] = f"mixture_{column}"
        elif column.startswith("species_candida_"):
            rename[column] = f"mixture_{column}"
        elif column.startswith("abundance_"):
            rename[column] = f"mixture_{column}"
        elif column in {
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
        }:
            rename[column] = f"similarity_{column}"
    frame = frame.rename(columns=rename)
    duplicate_columns = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        merged_duplicate_columns = {
            column: frame.loc[:, frame.columns == column].bfill(axis=1).iloc[:, 0]
            for column in duplicate_columns
        }
        frame = frame.loc[:, ~frame.columns.duplicated(keep=False)].copy()
        frame = pd.concat([frame, pd.DataFrame(merged_duplicate_columns, index=frame.index)], axis=1)
    else:
        frame = frame.copy()
    redundant_columns = [
        column
        for column in [
            "threshold",
            "rf_balanced_accuracy_mean",
            "rf_f1_mean",
            "rf_average_precision_mean",
            "rf_roc_auc_mean",
            *REDUNDANT_OUTPUT_COLUMNS,
        ]
        if column in frame.columns
    ]
    redundant_columns.extend(
        column
        for column in frame.columns
        if (
            column.startswith("mixture_abundance_local_mixture_set_b_")
            or column.endswith("_species_accuracy")
            or column.endswith("_specificity")
            or column.endswith("_false_positive_rate")
        )
    )
    frame = frame.drop(columns=[column for column in redundant_columns if column in frame.columns])
    return frame.reindex(columns=ordered_metric_columns(frame.columns))


def ranked_final_metrics(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    frame = compact_final_metrics(summaries)
    sort_cols = ["mixture_global_exact_match", "mixture_global_precision", "mixture_global_recall"]
    fallback = {"mixture_global_exact_match": "exact_match", "mixture_global_precision": "precision", "mixture_global_recall": "recall"}
    for column, fallback_column in fallback.items():
        if column not in frame.columns and fallback_column in frame.columns:
            frame[column] = frame[fallback_column]
    for column in sort_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    pipeline_col = "pipeline_id" if "pipeline_id" in frame.columns else "pipeline"
    return frame.sort_values(
        ["benchmark_stage", *sort_cols, pipeline_col],
        ascending=[True, False, False, False, True],
        na_position="last",
    )


def write_pipeline_evaluation_outputs(summaries: list[dict[str, Any]], output_root: Path) -> None:
    compact = compact_final_metrics(summaries)
    ranked = ranked_final_metrics(summaries)
    sort_specs = {
        "best_exact_match": ["mixture_global_exact_match", "mixture_global_precision", "mixture_global_recall", "mixture_global_f1"],
        "best_precision": ["mixture_global_precision", "mixture_global_recall", "mixture_global_f1", "mixture_global_exact_match"],
        "best_f1": ["mixture_global_f1", "mixture_global_exact_match", "mixture_global_precision", "mixture_global_recall"],
    }
    for criterion, columns in sort_specs.items():
        frame = compact.copy()
        selection_metric = MEAN_DATASET_THRESHOLD_SELECTIONS.get(criterion)
        if selection_metric is not None and "threshold_selection_metric" in frame.columns:
            selected_frame = frame.loc[frame["threshold_selection_metric"].astype(str).eq(selection_metric)].copy()
            if not selected_frame.empty:
                frame = selected_frame
        sort_columns = [column for column in columns if column in frame.columns]
        for column in sort_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if sort_columns:
            pipeline_col = "pipeline_id" if "pipeline_id" in frame.columns else "pipeline"
            frame = frame.sort_values(
                [*sort_columns, pipeline_col],
                ascending=[False] * len(sort_columns) + [True],
                na_position="last",
            )
        write_csv_atomic(frame, output_root / PIPELINE_EVALUATION_FILENAMES[criterion])



def write_checkpoint(
    output_root: Path,
    summaries: list[dict[str, Any]],
    rf_summaries: list[pd.DataFrame],
    errors: list[dict[str, Any]],
) -> None:
    completed = {
        (str(row.get("benchmark_stage", "custom")), str(row.get("pipeline", "")))
        for row in summaries
    }
    errors = [
        row
        for row in errors
        if (str(row.get("benchmark_stage", "custom")), str(row.get("pipeline", ""))) not in completed
    ]
    if summaries:
        write_pipeline_evaluation_outputs(summaries, output_root)
    if rf_summaries:
        rf_frame = pd.concat(rf_summaries, ignore_index=True)
        write_csv_atomic(rf_frame, output_root / CANONICAL_RF_HOLDOUT_METRICS)
    error_path = output_root / CANONICAL_ERRORS
    if errors:
        error_frame = pd.DataFrame(errors)
        write_csv_atomic(error_frame, error_path)
    else:
        if error_path.exists():
            error_path.unlink()


def load_resume_state(output_root: Path) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    rf_summaries: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []

    final_path = first_existing_csv(output_root, (CANONICAL_FINAL_METRICS,))
    if final_path.exists():
        summaries = pd.read_csv(final_path).to_dict("records")

    rf_path = first_existing_csv(output_root, (CANONICAL_RF_HOLDOUT_METRICS,))
    if rf_path.exists():
        rf_summaries = [pd.read_csv(rf_path)]

    error_path = first_existing_csv(output_root, (CANONICAL_ERRORS,))
    if error_path.exists():
        errors = pd.read_csv(error_path).to_dict("records")

    return summaries, rf_summaries, errors


def completed_pipeline_keys(summaries: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row.get("benchmark_stage", "custom")), str(row.get("pipeline", row.get("pipeline_id", ""))))
        for row in summaries
    }


def completed_pipeline_keys_from_paths(paths: list[Path]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in paths:
        final_path = first_existing_csv(path, (CANONICAL_FINAL_METRICS,))
        if not final_path.exists():
            continue
        frame = pd.read_csv(final_path)
        pipeline_col = "pipeline" if "pipeline" in frame.columns else "pipeline_id" if "pipeline_id" in frame.columns else None
        if pipeline_col is None or "benchmark_stage" not in frame.columns:
            continue
        keys.update(
            (str(row["benchmark_stage"]), str(row[pipeline_col]))
            for _, row in frame[["benchmark_stage", pipeline_col]].dropna().drop_duplicates().iterrows()
        )
    return keys


def drop_resume_stages(
    summaries: list[dict[str, Any]],
    rf_summaries: list[pd.DataFrame],
    errors: list[dict[str, Any]],
    stages: set[str],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    if not stages:
        return summaries, rf_summaries, errors
    summaries = [
        row
        for row in summaries
        if str(row.get("benchmark_stage", "custom")) not in stages
    ]
    errors = [
        row
        for row in errors
        if str(row.get("benchmark_stage", "custom")) not in stages
    ]
    filtered_rf = []
    for frame in rf_summaries:
        if "benchmark_stage" in frame.columns:
            frame = frame.loc[~frame["benchmark_stage"].astype(str).isin(stages)].copy()
        if not frame.empty:
            filtered_rf.append(frame)
    return summaries, filtered_rf, errors


def resolve_pipeline_specs(args: argparse.Namespace, matrix: dict[str, Any]) -> list[dict[str, Any]]:
    stage_name = matrix.get("project", {}).get("benchmark_stage", "custom")
    specs = []
    if not args.method_grid_rf_only:
        for combo in iter_combinations(matrix):
            pipeline_name = combo["pipeline_slug"]
            specs.append(
                {
                    "pipeline": pipeline_name,
                    "benchmark_stage": stage_name,
                    **combo,
                }
            )
            if args.limit is not None and len(specs) >= args.limit:
                break
    if args.include_method_grid_rf or args.method_grid_rf_only:
        if args.method_grid_rf_all_orders:
            method_grid_orders = ALL_METHOD_GRID_ORDERS
        elif args.method_grid_rf_extra_orders_only:
            method_grid_orders = {
                name: order
                for name, order in ALL_METHOD_GRID_ORDERS.items()
                if name not in METHOD_GRID_ORDERS
            }
        else:
            method_grid_orders = METHOD_GRID_ORDERS
        method_grid_binnings = list(args.method_grid_rf_binnings)
        for baseline_method in args.method_grid_rf_baselines:
            for binning_name in method_grid_binnings:
                for order_name, step_order in method_grid_orders.items():
                    for transform_method in METHOD_GRID_TRANSFORMS:
                        pipeline_name = (
                            f"{PIPELINE_GRID_PREFIX}__{order_name}__{baseline_method}"
                            f"__{transform_method}__{binning_name}__tic"
                        )
                        specs.append(
                            {
                                "pipeline": pipeline_name,
                                "pipeline_id": pipeline_name,
                                "pipeline_slug": pipeline_name,
                                "benchmark_stage": stage_name,
                                "order": order_name,
                                "method_grid_order": order_name,
                                "method_grid_baseline": baseline_method,
                                "variance_stabilization": transform_method,
                                "peak_selection": "none",
                                "binning": binning_name,
                                "normalization": "tic",
                            }
                        )
    if args.pipelines:
        requested = set(args.pipelines)
        filtered = [spec for spec in specs if str(spec["pipeline"]) in requested]
        found = {str(spec["pipeline"]) for spec in filtered}
        missing = sorted(requested - found)
        if missing:
            raise KeyError(f"Requested pipelines are not generated by the selected options: {missing}")
        return filtered
    return specs


def select_top_pipeline_names_from_frame(frame: pd.DataFrame, stage_name: str, max_pipelines: int | None) -> list[str]:
    frame = frame.loc[frame["benchmark_stage"].astype(str).eq(stage_name)].copy()
    if frame.empty:
        return []
    pipeline_col = "pipeline" if "pipeline" in frame.columns else "pipeline_id"
    metric_cols = {
        "mixture_global_exact_match": "exact_match",
        "mixture_global_precision": "precision",
        "mixture_global_recall": "recall",
    }
    for column, fallback in metric_cols.items():
        if column not in frame.columns and fallback in frame.columns:
            frame[column] = frame[fallback]
    for column in metric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(
        [*metric_cols.keys(), pipeline_col],
        ascending=[False, False, False, True],
        na_position="last",
    )
    if max_pipelines is not None:
        frame = frame.head(max_pipelines)
    return frame[pipeline_col].astype(str).tolist()


def load_top_pipeline_names(path: Path, stage_name: str, n_top: int) -> list[str]:
    metric_path = first_existing_csv(path, (CANONICAL_FINAL_METRICS,))
    if not metric_path.exists():
        raise FileNotFoundError(f"Cannot select top pipelines; missing {metric_path}")
    frame = pd.read_csv(metric_path)
    required = {"benchmark_stage"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot select top pipelines from {metric_path}; missing columns: {sorted(missing)}")
    if "pipeline" not in frame.columns and "pipeline_id" not in frame.columns:
        raise ValueError(f"Cannot select top pipelines from {metric_path}; missing pipeline/pipeline_id")
    if "mixture_global_exact_match" not in frame.columns and "exact_match" not in frame.columns:
        raise ValueError(f"Cannot select top pipelines from {metric_path}; missing mixture_global_exact_match/exact_match")
    return select_top_pipeline_names_from_frame(frame, stage_name, n_top)


def run_one_pipeline(
    spec: dict[str, Any],
    matrix: dict[str, Any],
    write_diagnostics: bool = False,
    diagnostics_dir: Path | None = None,
    temp_root: Path | None = None,
    preprocess_workers: int = 1,
    threshold_selection: str = "exact_match",
    max_pairs: int = 20000,
    similarity_workers: int = 1,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    pipeline_name = spec["pipeline"]
    if temp_root is None:
        raise ValueError("temp_root is required when generating temporary H5s.")
    active_matrix = build_temporary_h5s_for_pipeline(spec, matrix, temp_root, preprocess_workers=preprocess_workers)

    spectra, labels, origins, mz_axis = load_training_data(active_matrix, pipeline_name)
    spectra, labels, origins = cap_training_data(
        spectra,
        labels,
        origins,
        None,
        int(active_matrix.get("training", {}).get("random_state", DEFAULT_RF_RANDOM_STATE)),
    )
    weights, rf_metrics, importances = train_forests(
        spectra,
        labels,
        origins,
        mz_axis,
        active_matrix,
        pipeline_name,
        collect_importances=write_diagnostics,
    )
    rf_holdout_summary = summarize_rf_holdout(rf_metrics, pipeline_name)
    means = species_means(spectra, labels, active_matrix["species"])
    pure_score_matrix = pure_score_matrix_from_means(means, weights, active_matrix["species"])
    mixture_spots = load_mixture_spots(active_matrix, pipeline_name, mz_axis)
    mixture_summary, abundance_diagnostics = evaluate_mixtures(
        pipeline_name,
        mixture_spots,
        means,
        weights,
        pure_score_matrix,
        active_matrix,
        collect_diagnostics=write_diagnostics,
        threshold_selection=threshold_selection,
    )

    pure_spots = load_pure_validation_spots(active_matrix, pipeline_name, mz_axis)
    pure_truth = pure_scores = None
    if not pure_spots.empty:
        pure_truth, pure_scores, _pure_abundance_matrix = score_evaluation_spots(
            active_matrix,
            pure_spots,
            means,
            weights,
            pure_score_matrix,
        )

    if write_diagnostics:
        if diagnostics_dir is None:
            raise ValueError("diagnostics_dir is required when write_diagnostics=True")
        run_dir = diagnostics_dir / pipeline_name
        run_dir.mkdir(parents=True, exist_ok=True)
        rf_metrics.to_csv(run_dir / "rf_holdout_metrics.csv", index=False)
        importances.to_csv(run_dir / "rf_bin_importances.csv", index=False)
        abundance_diagnostics.to_csv(run_dir / "baseline_abundance_diagnostics.csv", index=False)

    base_summary_row = {
        "pipeline": pipeline_name,
        "benchmark_stage": active_matrix.get("project", {}).get("benchmark_stage", "custom"),
        "n_training_spectra": int(len(spectra)),
        "n_training_origins": int(len(np.unique(origins))),
        "evaluation_unit": "dataset_composition_spot",
        "n_mixture_spots": int(len(mixture_spots)),
    }
    base_summary_row.update(
        similarity_summary(
            spectra,
            labels,
            active_matrix["species"],
            weights=None,
            max_pairs=max_pairs,
            random_state=int(active_matrix.get("training", {}).get("random_state", DEFAULT_RF_RANDOM_STATE)),
            workers=similarity_workers,
        )
    )
    base_summary_row.update(
        similarity_summary(
            spectra,
            labels,
            active_matrix["species"],
            weights=weights,
            max_pairs=max_pairs,
            random_state=int(active_matrix.get("training", {}).get("random_state", DEFAULT_RF_RANDOM_STATE)),
            prefix="weighted_",
            workers=similarity_workers,
        )
    )
    overall_rf = rf_holdout_summary.loc[rf_holdout_summary["species"].eq("ALL_SPECIES_MEAN")]
    if not overall_rf.empty:
        for col, value in overall_rf.iloc[0].items():
            if col not in {"pipeline", "species"}:
                base_summary_row[col] = value

    summary_rows = []
    for _, mixture_row in mixture_summary.iterrows():
        summary_row = dict(base_summary_row)
        summary_row.update(mixture_row.to_dict())
        selected_threshold = float(summary_row.get("selected_threshold", float("nan")))
        if pure_truth is not None and pure_scores is not None and np.isfinite(selected_threshold):
            pure_metrics = exact_match_metrics_at_threshold(pure_truth, pure_scores, selected_threshold)
            summary_row.update(
                {
                    "pure_global_exact_match": pure_metrics["exact_match"],
                    "pure_global_precision": pure_metrics["precision"],
                    "pure_global_recall": pure_metrics["recall"],
                    "pure_global_f1": pure_metrics["f1"],
                    "pure_global_mean_predicted_species": pure_metrics["mean_predicted_species"],
                    "pure_global_n_samples": pure_metrics["n_spots"],
                    "pure_global_species_accuracy": pure_metrics["species_accuracy"],
                    "pure_prauc_global": global_prauc(pure_truth, pure_scores),
                    "pure_prauc_supported_species": supported_species_prauc(pure_truth, pure_scores),
                }
            )
        summary_rows.append(summary_row)
    return summary_rows, rf_holdout_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override project.output_root so concurrent runs can write to separate result folders.",
    )
    parser.add_argument("--pipelines", nargs="*", help="Explicit preprocessing pipeline ids.")
    parser.add_argument("--include-optional-binning", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--benchmark-stage",
        default=None,
        help="Optional benchmark stage name from the matrix benchmark_stages block.",
    )
    parser.add_argument(
        "--top-n-from-ranking",
        type=int,
        default=None,
        help="Select this many top pipelines from an existing metrics CSV/folder and run only them.",
    )
    parser.add_argument(
        "--top-ranking-path",
        type=Path,
        default=None,
        help=(
            "Existing pipeline_evaluation_best_f1.csv/final_pipeline_metrics.csv or output folder "
            "used by --top-n-from-ranking. Defaults to --output-root."
        ),
    )
    parser.add_argument(
        "--top-selection-stage",
        default="local_mixtures",
        help="Stage used to rank pipelines for --top-n-from-ranking.",
    )
    parser.add_argument(
        "--include-method-grid-rf",
        action="store_true",
        help="Add RF specs for the canonical method grid.",
    )
    parser.add_argument(
        "--method-grid-rf-only",
        action="store_true",
        help="Run only RF specs from the canonical method grid.",
    )
    parser.add_argument(
        "--method-grid-rf-baselines",
        nargs="*",
        default=list(METHOD_GRID_BASELINES),
        choices=list(METHOD_GRID_BASELINES),
        help="Baseline correction methods for the RF method grid.",
    )
    parser.add_argument(
        "--method-grid-rf-binnings",
        nargs="*",
        default=["b3da_avg"],
        choices=list(METHOD_GRID_BINNINGS),
        help=(
            "Binning representations for the RF method grid. Defaults to b3da_avg so the main "
            "pipeline benchmark does not mix representation effects; run binning ablations for alternatives."
        ),
    )
    parser.add_argument(
        "--method-grid-rf-all-orders",
        action="store_true",
        help="Use all 24 permutations of VS/S/BC/N in the RF method grid.",
    )
    parser.add_argument(
        "--method-grid-rf-extra-orders-only",
        action="store_true",
        help="Use only the 18 VS/S/BC/N permutations not present in the default 6-order RF method grid.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing result CSVs and skip completed benchmark_stage/pipeline pairs.",
    )
    parser.add_argument(
        "--skip-completed-from",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Additional pipeline_evaluation_best_f1.csv/final_pipeline_metrics.csv files or output directories whose "
            "completed benchmark_stage/pipeline pairs should be skipped."
        ),
    )
    parser.add_argument(
        "--rerun-stages",
        nargs="*",
        default=[],
        help="With --resume, discard existing rows for these benchmark stages and run them again.",
    )
    parser.add_argument("--write-diagnostics", action="store_true", help="Write detailed per-pipeline CSVs for debugging only.")
    parser.add_argument(
        "--threshold-selection",
        choices=["exact_match", "f1", "precision", "recall", "mean_dataset"],
        default="f1",
        help=(
            "Metric used to select the mixture score threshold before ranking/output. "
            "Use mean_dataset to select separate thresholds by the mean exact_match, "
            "precision, and f1 over mixture datasets for the three ranking CSVs."
        ),
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Worker processes for temporary H5 preprocessing. Defaults to all CPUs; use 1 to disable preprocessing parallelism.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20000,
        help="Maximum spectrum pairs sampled per species when summarizing similarity metrics.",
    )
    parser.add_argument(
        "--similarity-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Worker threads for per-species similarity summaries. Use 1 to disable similarity parallelism.",
    )
    args = parser.parse_args()
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be >= 1")
    if args.similarity_workers < 1:
        raise ValueError("--similarity-workers must be >= 1")

    matrix_base = load_yaml(args.matrix)
    project_cfg = matrix_base.setdefault("project", {})
    default_output_root = project_cfg.get("output_root", "outputs/rf_preprocessing_benchmark")
    output_root = args.output_root if args.output_root is not None else PROJECT_ROOT / default_output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    project_cfg["output_root"] = str(output_root.relative_to(PROJECT_ROOT) if output_root.is_relative_to(PROJECT_ROOT) else output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    top_pipeline_names_from_ranking: list[str] | None = None
    if args.top_n_from_ranking is not None:
        if args.top_n_from_ranking < 1:
            raise ValueError("--top-n-from-ranking must be >= 1")
        ranking_path = args.top_ranking_path if args.top_ranking_path is not None else output_root
        if not ranking_path.is_absolute():
            ranking_path = PROJECT_ROOT / ranking_path
        top_pipeline_names_from_ranking = load_top_pipeline_names(
            ranking_path,
            args.top_selection_stage,
            args.top_n_from_ranking,
        )
        if not top_pipeline_names_from_ranking:
            raise RuntimeError(
                f"No top pipelines found in {ranking_path} for stage {args.top_selection_stage!r}."
            )
        print(
            f"Selected {len(top_pipeline_names_from_ranking)} top pipelines from "
            f"{ranking_path} stage {args.top_selection_stage}.",
            flush=True,
        )
    if args.resume:
        summaries, rf_summaries, errors = load_resume_state(output_root)
        summaries, rf_summaries, errors = drop_resume_stages(
            summaries,
            rf_summaries,
            errors,
            set(args.rerun_stages),
        )
    else:
        clear_previous_result_csvs(output_root)
        summaries = []
        rf_summaries = []
        errors = []
    if not args.resume:
        clear_stale_temp_h5s(output_root)

    external_completed = completed_pipeline_keys_from_paths(args.skip_completed_from)
    if external_completed:
        print(f"Loaded {len(external_completed)} externally completed stage/pipeline pairs to skip.")

    diagnostics_dir = output_root / "_diagnostics"
    all_pipeline_specs: list[dict[str, Any]] = []

    def write_manifest() -> None:
        pd.DataFrame(all_pipeline_specs).to_csv(output_root / "pipelines_to_test.csv", index=False)

    def execute_specs(specs: list[dict[str, Any]], matrices_by_stage: dict[str, dict[str, Any]]) -> None:
        for spec in specs:
            pipeline_name = spec["pipeline"]
            stage_name = spec.get("benchmark_stage", "custom")
            if (stage_name, pipeline_name) in external_completed:
                print(f"Skipping externally completed {stage_name}/{pipeline_name}")
                continue
            if args.resume and (stage_name, pipeline_name) in completed_pipeline_keys(summaries):
                print(f"Skipping completed {stage_name}/{pipeline_name}")
                continue
            if stage_name in matrices_by_stage:
                matrix = matrices_by_stage[stage_name]
            else:
                matrix = apply_benchmark_stage(matrix_base, stage_name if stage_name != "custom" else None)
            temp_root = output_root / "_tmp_h5" / stage_name / pipeline_name
            try:
                print(f"Running {stage_name}/{pipeline_name}")
                summary, rf_summary = run_one_pipeline(
                    spec,
                    matrix,
                    write_diagnostics=args.write_diagnostics,
                    diagnostics_dir=diagnostics_dir,
                    temp_root=temp_root,
                    preprocess_workers=args.preprocess_workers,
                    threshold_selection=args.threshold_selection,
                    max_pairs=args.max_pairs,
                    similarity_workers=args.similarity_workers,
                )
                for summary_row in summary:
                    summaries.append({**spec, **summary_row})
                rf_summaries.append(rf_summary)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                errors.append({"benchmark_stage": stage_name, "pipeline": pipeline_name, "error": repr(exc)})
                print(f"FAILED {stage_name}/{pipeline_name}: {exc}")
            finally:
                if temp_root.exists():
                    shutil.rmtree(temp_root)
                    try:
                        temp_root.parent.rmdir()
                    except OSError:
                        pass
                    try:
                        temp_root.parent.parent.rmdir()
                    except OSError:
                        pass
                write_checkpoint(output_root, summaries, rf_summaries, errors)

    active_stage = args.benchmark_stage or matrix_base.get("project", {}).get("benchmark_stage")
    active_matrix = apply_benchmark_stage(matrix_base, active_stage)
    stage_name = active_matrix.get("project", {}).get("benchmark_stage", "custom")
    pipeline_specs = resolve_pipeline_specs(args, active_matrix)
    if top_pipeline_names_from_ranking is not None:
        spec_by_pipeline = {str(spec["pipeline"]): spec for spec in pipeline_specs}
        pipeline_specs = [spec_by_pipeline[name] for name in top_pipeline_names_from_ranking if name in spec_by_pipeline]
        if not pipeline_specs:
            raise RuntimeError(f"No generated specs matched the selected top pipelines for stage {stage_name}.")
    all_pipeline_specs.extend(pipeline_specs)
    write_manifest()
    execute_specs(pipeline_specs, {stage_name: active_matrix})

    write_checkpoint(output_root, summaries, rf_summaries, errors)
    print(f"Finished. Final metrics: {output_root / CANONICAL_FINAL_METRICS}")


if __name__ == "__main__":
    main()
