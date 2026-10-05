#!/usr/bin/env python3
"""Benchmark pure-spectrum cross-domain classification for ranked pipelines.

The benchmark trains on MARISMA and tests on DRIAMS-A, then reverses the
direction. It evaluates svm, rf_multiclass, knn, and logistic_l1.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import os
import sys
import time
import warnings
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src/code").exists() and (candidate / "data/species_mapping.csv").exists():
            return candidate
    return start.parents[5]


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
RESULT_TABLE = None
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/benchmarks/pure_cross_domain"
PIPELINE_GRID_PREFIX = "pipeline_grid"
DEFAULT_PURE_METADATA = PROJECT_ROOT / "data/pures/driams_marisma/driams_marisma.csv"
CODE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = CODE_DIR / "models"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from helpers.SpectrumObject import SpectrumObject  # noqa: E402
from helpers import preprocess as preprocess_module  # noqa: E402
from benchmarking.core.config import coerce_path_args  # noqa: E402
from benchmarking.core.metrics import PURE_CROSS_DOMAIN_METRIC_NAMES, binary_prauc  # noqa: E402
from benchmarking.core.preprocessing import load_h5_records  # noqa: E402
from models.pure_algorithms import PURE_ALGORITHMS, make_pure_algorithms  # noqa: E402


METRICS = list(PURE_CROSS_DOMAIN_METRIC_NAMES)

ALGORITHMS = list(PURE_ALGORITHMS)

OBSOLETE_METHOD_COLUMNS = (
    "mixture_prauc_rank",
    "mixture_prauc_global",
    "original_binning",
    "benchmark_binning",
)

TARGET_SPECIES = [
    "candida_albicans",
    "candida_glabrata",
    "candida_parapsilosis",
    "candida_tropicalis",
]

DIRECTIONS = {
    "marisma_to_driams": ("marisma", "driams"),
    "driams_to_marisma": ("driams", "marisma"),
}


@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pure MARISMA/DRIAMS-A cross-domain benchmark for ranked pipelines."
    )
    parser.add_argument("--ranking", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pure-metadata", type=Path, default=DEFAULT_PURE_METADATA)
    parser.add_argument(
        "--max-per-species-domain",
        type=int,
        default=None,
        help="Optional cap for quick/debug runs. Default uses every pure spectrum.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pipeline-jobs",
        type=int,
        default=1,
        help="Pipelines evaluated concurrently. Default is 1 so all CPUs can preprocess spectra together.",
    )
    parser.add_argument(
        "--spectrum-jobs",
        type=int,
        default=None,
        help="Workers used to preprocess spectra inside each pipeline. Default uses available CPUs.",
    )
    parser.add_argument("--inner-jobs", type=int, default=None)
    parser.add_argument(
        "--algorithm",
        choices=["all", *ALGORITHMS],
        default="all",
        help="Run one method only, useful for shard-level parallel jobs. Default runs all methods.",
    )
    parser.add_argument(
        "--pipeline-file",
        type=Path,
        default=None,
        help="Text/CSV file containing pipeline_id values to run.",
    )
    parser.add_argument(
        "--write-pipeline-shards",
        type=int,
        default=None,
        help="Write N pipeline shard files under output-dir/pipeline_shards and exit.",
    )
    parser.add_argument(
        "--benchmark-binning",
        choices=["original", "b3da_avg", "b6da_avg", "b9da_avg", "b2000ppm_avg"],
        default="b9da_avg",
        help="Binning used for the pure benchmark.",
    )
    parser.add_argument(
        "--preprocessed-source-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Pure-spectrum H5 source directories to reuse when pipeline H5s already exist.",
    )
    parser.add_argument(
        "--disable-preprocessed-h5",
        action="store_true",
        help="Force preprocessing from raw file paths instead of reusing existing pipeline H5s.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--force", action="store_true", help="Recompute pipelines already present in CSVs.")
    args = parser.parse_args()
    coerce_path_args(
        args,
        ("ranking", "output_dir", "pipeline_file", "pure_metadata"),
        ("preprocessed_source_dirs",),
    )
    return args


def parse_pipeline_id(pipeline_id: str) -> dict[str, str]:
    parts = str(pipeline_id).split("__")
    if len(parts) < 6 or parts[0] != PIPELINE_GRID_PREFIX:
        raise ValueError(f"Unsupported pipeline id: {pipeline_id}")
    return {
        "order": parts[1],
        "baseline": parts[2],
        "transform": parts[3],
        "binning": parts[4],
        "normalization": parts[5],
    }


def binning_params_from_name(binning_name: str) -> dict[str, Any]:
    if binning_name == "b3da_avg":
        return {"kind": "fixed", "step": 3, "aggregation": "mean"}
    if binning_name == "b6da_avg":
        return {"kind": "fixed", "step": 6, "aggregation": "mean"}
    if binning_name == "b9da_avg":
        return {"kind": "fixed", "step": 9, "aggregation": "mean"}
    if binning_name == "b2000ppm_avg":
        return {"kind": "adaptive", "ppm_tolerance": 2000, "aggregation": "mean"}
    raise ValueError(f"Unsupported binning: {binning_name}")


def build_method_grid_preprocessor(pipeline_id: str, benchmark_binning: str = "original"):
    parsed = parse_pipeline_id(pipeline_id)
    order_steps = parsed["order"].split("-")
    if "N" in order_steps and parsed["normalization"] != "tic":
        raise ValueError(f"Unsupported normalization for {pipeline_id}: {parsed['normalization']}")

    baseline_params = None
    if "BC" in order_steps:
        baseline_params = {
            "snip": {"method": "SNIP", "snip_n_iter": 20},
            "tophat": {"method": "TopHat", "tophat_window": 150},
            "convexhull": {"method": "ConvexHull"},
        }.get(parsed["baseline"])
        if baseline_params is None:
            raise ValueError(f"Unsupported baseline for {pipeline_id}: {parsed['baseline']}")

    steps = [preprocess_module.Trimmer(min=2000, max=20000)]
    for step in order_steps:
        if step == "VS":
            steps.append(preprocess_module.VarStabilizer(method=parsed["transform"]))
        elif step == "S":
            steps.append(preprocess_module.Smoother(halfwindow=10, polyorder=3))
        elif step == "BC":
            steps.append(preprocess_module.BaselineCorrecter(**baseline_params))
        elif step == "N":
            steps.append(preprocess_module.Normalizer(sum=1))
        else:
            raise ValueError(f"Unknown preprocessing step in {pipeline_id}: {step}")

    binning_name = parsed["binning"] if benchmark_binning == "original" else benchmark_binning
    binning = binning_params_from_name(binning_name)
    if binning["kind"] == "fixed":
        steps.append(
            preprocess_module.Binner(
                start=2000,
                stop=20000,
                step=binning["step"],
                aggregation=binning["aggregation"],
            )
        )
    elif binning["kind"] == "adaptive":
        steps.append(
            preprocess_module.AdaptiveBinner(
                start=2000,
                stop=20000,
                ppm_tolerance=binning["ppm_tolerance"],
                aggregation=binning["aggregation"],
            )
        )
    return preprocess_module.SequentialPreprocessor(*steps)


def parse_species_name(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                if len(parsed):
                    return str(parsed[0]).strip().replace(" ", "_").lower()
            except Exception:
                pass
        return text.replace(" ", "_").lower()
    return str(value).strip().replace(" ", "_").lower()


def load_balanced_pure_metadata(path: Path, max_per_species_domain: int | None, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pure metadata CSV: {path}")
    meta = pd.read_csv(path)
    meta = meta.loc[meta["num_species"].astype(int).eq(1)].copy()
    meta["species_label"] = meta["species_names"].map(parse_species_name)
    origin = meta["origin"].astype(str).str.lower()
    meta["domain"] = np.select(
        [origin.str.contains("marisma"), origin.str.contains("driams")],
        ["marisma", "driams"],
        default="other",
    )
    meta = meta.loc[meta["domain"].isin(["marisma", "driams"])].reset_index(drop=True)

    mar_species = set(meta.loc[meta["domain"].eq("marisma"), "species_label"])
    dri_species = set(meta.loc[meta["domain"].eq("driams"), "species_label"])
    common_species = sorted(mar_species & dri_species)
    meta = meta.loc[meta["species_label"].isin(common_species)].copy()

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, group in meta.groupby(["domain", "species_label"], sort=False):
        idx = group.index.to_numpy()
        if max_per_species_domain is not None and len(idx) > max_per_species_domain:
            idx = rng.choice(idx, size=max_per_species_domain, replace=False)
        keep.extend(idx.tolist())
    return meta.loc[np.asarray(sorted(keep), dtype=int)].reset_index(drop=True)


def read_pipeline_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "pipeline_id" not in df.columns and "pipeline" in df.columns:
            df = df.rename(columns={"pipeline": "pipeline_id"})
        if "pipeline_id" not in df.columns:
            raise KeyError(f"{path} must contain pipeline_id")
        return df["pipeline_id"].dropna().astype(str).tolist()
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_pipeline_table(path: Path | None, limit: int | None, pipeline_file: Path | None = None) -> pd.DataFrame:
    if pipeline_file is not None:
        selected_ids = read_pipeline_file(pipeline_file)
        table = pd.DataFrame({"pipeline_id": selected_ids})
        table["mixture_prauc_global"] = np.nan
        table["mixture_prauc_rank"] = np.arange(1, len(table) + 1)
    else:
        if path is None:
            raise ValueError("Provide --pipeline-file or --ranking.")
        table = pd.read_csv(path)
        if "pipeline_id" not in table.columns and "pipeline" in table.columns:
            table = table.rename(columns={"pipeline": "pipeline_id"})
        if "mixture_prauc_global" not in table.columns:
            raise KeyError(f"{path} must contain mixture_prauc_global")
        table = table.dropna(subset=["pipeline_id", "mixture_prauc_global"]).copy()
        table["pipeline_id"] = table["pipeline_id"].astype(str)
        table["mixture_prauc_global"] = pd.to_numeric(table["mixture_prauc_global"], errors="coerce")
        table = table.sort_values(["mixture_prauc_global", "pipeline_id"], ascending=[False, True])
        table = table.drop_duplicates("pipeline_id", keep="first").reset_index(drop=True)
        table["mixture_prauc_rank"] = np.arange(1, len(table) + 1)
        if pipeline_file is not None:
            selected_ids = read_pipeline_file(pipeline_file)
            selected = set(selected_ids)
            table = table.loc[table["pipeline_id"].isin(selected)].copy()
            order = {pipeline_id: i for i, pipeline_id in enumerate(selected_ids)}
            table["_pipeline_file_order"] = table["pipeline_id"].map(order)
            table = table.sort_values("_pipeline_file_order").drop(columns=["_pipeline_file_order"]).reset_index(drop=True)
    if limit is not None:
        table = table.head(limit).copy()
    return table


def write_pipeline_shards(table: pd.DataFrame, output_dir: Path, n_shards: int) -> Path:
    if n_shards <= 0:
        raise ValueError("--write-pipeline-shards must be > 0")
    shard_dir = output_dir / "pipeline_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pipeline_ids = table["pipeline_id"].astype(str).tolist()
    shards = [pipeline_ids[i::n_shards] for i in range(n_shards)]
    width = len(str(n_shards))
    for index, shard in enumerate(shards, start=1):
        path = shard_dir / f"pipelines_shard{index:0{width}d}of{n_shards}.txt"
        path.write_text("\n".join(shard) + ("\n" if shard else ""))
    return shard_dir


def make_algorithms(seed: int, inner_jobs: int, algorithms_to_run: list[str]):
    return make_pure_algorithms(seed=seed, inner_jobs=inner_jobs, algorithms_to_run=algorithms_to_run)


def prediction_scores(model: Any, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(getattr(model, "classes_"))
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X_test), dtype=float)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_test), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
    else:
        raise TypeError(f"{type(model).__name__} does not expose scores")
    return scores, classes


def prauc_from_curve(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return binary_prauc(y_true, y_score)


def macro_prauc_from_curves(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    values = [
        prauc_from_curve(y_true_bin[:, class_index].astype(int), scores[:, class_index])
        for class_index in range(y_true_bin.shape[1])
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])
    return {
        "roc_auc_macro": float(roc_auc_score(y_true_bin, scores, average="macro", multi_class="ovr")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "prauc_macro": macro_prauc_from_curves(y_true_bin, scores),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def candida_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for species in TARGET_SPECIES:
        true_positive = np.logical_and(y_true == species, y_pred == species).sum()
        false_positive = np.logical_and(y_true != species, y_pred == species).sum()
        false_negative = np.logical_and(y_true == species, y_pred != species).sum()
        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        metrics[f"{species}_recall"] = (
            float(true_positive / recall_denominator)
            if recall_denominator
            else float("nan")
        )
        metrics[f"{species}_precision"] = (
            float(true_positive / precision_denominator)
            if precision_denominator
            else float("nan")
        )
    return metrics


def per_species_metrics(
    pipeline_id: str,
    algorithm: str,
    direction: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    species: list[str],
) -> list[dict[str, Any]]:
    class_to_score_col = {str(cls): idx for idx, cls in enumerate(classes)}
    rows: list[dict[str, Any]] = []
    for species_name in species:
        true_mask = y_true == species_name
        pred_mask = y_pred == species_name
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        fp = int(np.logical_and(~true_mask, pred_mask).sum())
        fn = int(np.logical_and(true_mask, ~pred_mask).sum())
        tn = int(np.logical_and(~true_mask, ~pred_mask).sum())
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        precision = tp / precision_denominator if precision_denominator else float("nan")
        recall = tp / recall_denominator if recall_denominator else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
            else float("nan")
        )
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        y_binary = true_mask.astype(int)
        score_col = class_to_score_col.get(species_name)
        y_score = scores[:, score_col] if score_col is not None else np.full(len(y_true), np.nan)
        rows.append(
            {
                "pipeline": pipeline_id,
                "model": algorithm,
                "direction": direction,
                "species": species_name,
                "n_test": int(len(y_true)),
                "support": int(true_mask.sum()),
                "n_predicted": int(pred_mask.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "specificity": float(specificity),
                "pr_auc": prauc_from_curve(y_binary, y_score),
                "roc_auc": safe_binary_roc_auc(y_binary, y_score),
            }
        )
    return rows


def safe_binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = np.isfinite(y_score)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask].astype(int), y_score[mask]))


def preprocess_spectrum_chunk(
    pipeline_id: str,
    chunk: list[tuple[int, str]],
    benchmark_binning: str,
) -> list[tuple[int, np.ndarray]]:
    preprocessor = build_method_grid_preprocessor(pipeline_id, benchmark_binning=benchmark_binning)
    out = []
    for index, file_path in chunk:
        spectrum = preprocessor(SpectrumObject.load(str(file_path)))
        out.append((index, np.asarray(spectrum.intensity, dtype=np.float32)))
    return out


def chunked(items: list[tuple[int, str]], n_chunks: int) -> list[list[tuple[int, str]]]:
    n_chunks = max(1, min(n_chunks, len(items)))
    return [items[i::n_chunks] for i in range(n_chunks)]


def preprocess_pipeline_data(
    pipeline_id: str,
    metadata: pd.DataFrame,
    spectrum_jobs: int,
    benchmark_binning: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    items = list(enumerate(metadata["file_path"].astype(str).tolist()))
    total = len(items)
    print(
        f"{pipeline_id} | preprocessing start | spectra={total} | spectrum_jobs={spectrum_jobs}",
        flush=True,
    )
    if spectrum_jobs <= 1:
        processed = preprocess_spectrum_chunk(pipeline_id, items, benchmark_binning)
        print(f"{pipeline_id} | spectra processed {len(processed)}/{total}", flush=True)
    else:
        chunks = [chunk for chunk in chunked(items, spectrum_jobs) if chunk]
        processed = []
        executor = ProcessPoolExecutor(max_workers=len(chunks))
        futures = []
        try:
            futures = [executor.submit(preprocess_spectrum_chunk, pipeline_id, chunk, benchmark_binning) for chunk in chunks]
            for future in as_completed(futures):
                chunk_result = future.result()
                processed.extend(chunk_result)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    processed.sort(key=lambda pair: pair[0])
    spectra = [intensity for _, intensity in processed]
    print(f"{pipeline_id} | preprocessing done | matrix={len(spectra)}x{len(spectra[0])}", flush=True)
    return (
        np.vstack(spectra),
        metadata["species_label"].to_numpy(dtype=str),
        metadata["domain"].to_numpy(dtype=str),
    )


def load_preprocessed_pipeline_data(
    pipeline_id: str,
    metadata: pd.DataFrame,
    source_dirs: list[Path],
    benchmark_binning: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not source_dirs:
        return None
    if benchmark_binning != "original":
        return None
    h5_paths = [source_dir / f"{source_dir.name}_{pipeline_id}.h5" for source_dir in source_dirs]
    if not all(path.exists() for path in h5_paths):
        return None

    selected = metadata[["file_path", "species_label", "domain"]].copy()
    selected["file_path"] = selected["file_path"].astype(str)
    selected_by_path = selected.drop_duplicates("file_path").set_index("file_path")

    spectra_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    domain_parts: list[np.ndarray] = []
    for h5_path in h5_paths:
        spectra, labels, _, h5_metadata, _ = load_h5_records(h5_path, pure_only=True)
        if "file_path" not in h5_metadata.columns:
            return None
        h5_file_paths = h5_metadata["file_path"].astype(str)
        keep_mask = h5_file_paths.isin(selected_by_path.index).to_numpy()
        if not keep_mask.any():
            continue
        kept_paths = h5_file_paths.loc[keep_mask].to_numpy(dtype=str)
        selected_rows = selected_by_path.loc[kept_paths]
        spectra_parts.append(np.asarray(spectra[keep_mask], dtype=np.float32))
        label_parts.append(selected_rows["species_label"].to_numpy(dtype=str))
        domain_parts.append(selected_rows["domain"].to_numpy(dtype=str))

    if not spectra_parts:
        return None
    return (
        np.vstack(spectra_parts),
        np.concatenate(label_parts),
        np.concatenate(domain_parts),
    )


def summarize_algorithm_records(records: list[dict[str, Any]], algorithms_to_run: list[str]) -> list[dict[str, Any]]:
    rows = []
    for algorithm in algorithms_to_run:
        algo_records = [record for record in records if record["algorithm"] == algorithm]
        if not algo_records:
            continue
        base = {
            key: algo_records[0][key]
            for key in [
                "pipeline_id",
                "order",
                "method_grid_baseline",
                "variance_stabilization",
                "normalization",
                "n_species",
                "n_features",
            ]
        }
        base["algorithm"] = algorithm
        for record in algo_records:
            direction = record["direction"]
            base[f"{direction}_n_train"] = record["n_train"]
            base[f"{direction}_n_test"] = record["n_test"]
            for metric in METRICS:
                base[f"{direction}_{metric}"] = record[metric]
        for metric in METRICS:
            vals = [record[metric] for record in algo_records if np.isfinite(record[metric])]
            base[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")
        rows.append(base)
    return rows


def evaluate_pipeline(
    pipeline_row: dict[str, Any],
    metadata: pd.DataFrame,
    seed: int,
    inner_jobs: int,
    spectrum_jobs: int,
    preprocessed_source_dirs: list[Path] | None,
    benchmark_binning: str,
    algorithms_to_run: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pipeline_id = str(pipeline_row["pipeline_id"])
    parsed = parse_pipeline_id(pipeline_id)
    loaded = load_preprocessed_pipeline_data(
        pipeline_id,
        metadata,
        preprocessed_source_dirs or [],
        benchmark_binning,
    )
    if loaded is None:
        X, y, domains = preprocess_pipeline_data(
            pipeline_id,
            metadata,
            spectrum_jobs=spectrum_jobs,
            benchmark_binning=benchmark_binning,
        )
    else:
        X, y, domains = loaded
        print(
            f"{pipeline_id} | loaded preprocessed H5s | spectra={len(y)} | features={X.shape[1]}",
            flush=True,
        )
    algorithms = make_algorithms(seed=seed, inner_jobs=inner_jobs, algorithms_to_run=algorithms_to_run)

    records: list[dict[str, Any]] = []
    direction_metric_rows: list[dict[str, Any]] = []
    species_metric_rows: list[dict[str, Any]] = []
    common_species = sorted(set(y[domains == "marisma"]) & set(y[domains == "driams"]))
    common_mask = np.isin(y, common_species)
    X, y, domains = X[common_mask], y[common_mask], domains[common_mask]

    for direction, (train_domain, test_domain) in DIRECTIONS.items():
        train_mask = domains == train_domain
        test_mask = domains == test_domain
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        for algorithm, estimator in algorithms.items():
            print(
                f"{pipeline_id} | {direction} | {algorithm} start | "
                f"n_train={len(y_train)} n_test={len(y_test)}",
                flush=True,
            )
            model = clone(estimator)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores, classes = prediction_scores(model, X_test)
            metrics = score_predictions(y_test, pred, scores, classes)
            species_metrics = candida_precision_recall(y_test, pred)
            species_metric_rows.extend(
                per_species_metrics(
                    pipeline_id,
                    algorithm,
                    direction,
                    y_test,
                    pred,
                    scores,
                    classes,
                    common_species,
                )
            )
            print(
                f"{pipeline_id} | {direction} | {algorithm} done | "
                f"roc_auc_macro={metrics['roc_auc_macro']:.4f} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"prauc_macro={metrics['prauc_macro']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )
            records.append(
                {
                    "pipeline_id": pipeline_id,
                    "order": pipeline_row.get("order", parsed["order"]),
                    "method_grid_baseline": pipeline_row.get("method_grid_baseline", parsed["baseline"]),
                    "variance_stabilization": pipeline_row.get("variance_stabilization", parsed["transform"]),
                    "normalization": pipeline_row.get("normalization", parsed["normalization"]),
                    "direction": direction,
                    "algorithm": algorithm,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "n_species": int(len(common_species)),
                    "n_features": int(X_train.shape[1]),
                    **metrics,
                }
            )
            direction_metric_rows.append(
                {
                    "pipeline": pipeline_id,
                    "model": algorithm,
                    "direction": direction,
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_f1": metrics["macro_f1"],
                    "pr_auc_macro": metrics["prauc_macro"],
                    "roc_auc_macro": metrics["roc_auc_macro"],
                    **species_metrics,
                }
            )
    return summarize_algorithm_records(records, algorithms_to_run), direction_metric_rows, species_metric_rows


def existing_pipeline_ids(output_dir: Path, algorithms_to_check: list[str]) -> set[str]:
    completed_sets: list[set[str]] = []
    for algorithm in algorithms_to_check:
        path = output_dir / f"{algorithm}.csv"
        if path.exists():
            try:
                with file_lock(output_dir / f".{algorithm}.csv.lock"):
                    completed_sets.append(set(pd.read_csv(path, usecols=["pipeline_id"])["pipeline_id"].astype(str)))
            except Exception:
                pass
    species_path = output_dir / "pure_species_metrics.csv"
    if not species_path.exists():
        return set()
    try:
        with file_lock(output_dir / ".pure_species_metrics.csv.lock"):
            species_frame = pd.read_csv(species_path, usecols=["pipeline", "model"])
        complete_species = set()
        required = set(algorithms_to_check)
        for pipeline, group in species_frame.groupby("pipeline"):
            if required.issubset(set(group["model"].astype(str))):
                complete_species.add(str(pipeline))
        completed_sets.append(complete_species)
    except Exception:
        return set()
    if not completed_sets:
        return set()
    return set.intersection(*completed_sets)


def append_method_rows(output_dir: Path, method_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in method_rows:
        path = output_dir / f"{row['algorithm']}.csv"
        lock_path = output_dir / f".{row['algorithm']}.csv.lock"
        with file_lock(lock_path):
            out = pd.DataFrame([row]).drop(columns=["algorithm"])
            if path.exists():
                existing = pd.read_csv(path)
                existing = existing.drop(columns=list(OBSOLETE_METHOD_COLUMNS), errors="ignore")
                out = pd.concat([existing, out], ignore_index=True, sort=False)
            out = out.drop(columns=list(OBSOLETE_METHOD_COLUMNS), errors="ignore")
            out.to_csv(path, index=False)


def append_direction_metric_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pure_direction_metrics.csv"
    lock_path = output_dir / ".pure_direction_metrics.csv.lock"
    columns = [
        "pipeline",
        "model",
        "direction",
        "accuracy",
        "macro_precision",
        "macro_f1",
        "pr_auc_macro",
        "roc_auc_macro",
        "candida_albicans_recall",
        "candida_albicans_precision",
        "candida_glabrata_recall",
        "candida_glabrata_precision",
        "candida_parapsilosis_recall",
        "candida_parapsilosis_precision",
        "candida_tropicalis_recall",
        "candida_tropicalis_precision",
    ]
    with file_lock(lock_path):
        out = pd.DataFrame(rows)
        out = out.reindex(columns=columns)
        out.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_species_metric_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pure_species_metrics.csv"
    lock_path = output_dir / ".pure_species_metrics.csv.lock"
    columns = [
        "pipeline",
        "model",
        "direction",
        "species",
        "n_test",
        "support",
        "n_predicted",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "specificity",
        "pr_auc",
        "roc_auc",
    ]
    with file_lock(lock_path):
        out = pd.DataFrame(rows)
        out = out.reindex(columns=columns)
        out.to_csv(path, mode="a", header=not path.exists(), index=False)


def finalize_method_csv(output_dir: Path, algorithm: str) -> None:
    path = output_dir / f"{algorithm}.csv"
    if not path.exists():
        return
    lock_path = output_dir / f".{algorithm}.csv.lock"
    with file_lock(lock_path):
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df.drop(columns=list(OBSOLETE_METHOD_COLUMNS), errors="ignore")
        df = df.drop_duplicates("pipeline_id", keep="first")
        df = df.sort_values("mean_roc_auc_macro", ascending=False).reset_index(drop=True)
        df["pure_rank_mean_roc_auc_macro"] = np.arange(1, len(df) + 1)
        df.to_csv(path, index=False)


def finalize_direction_metrics_csv(output_dir: Path) -> None:
    path = output_dir / "pure_direction_metrics.csv"
    if not path.exists():
        return
    lock_path = output_dir / ".pure_direction_metrics.csv.lock"
    with file_lock(lock_path):
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df.drop_duplicates(["pipeline", "model", "direction"], keep="first")
        df = df.sort_values(["model", "pipeline", "direction"]).reset_index(drop=True)
        df.to_csv(path, index=False)


def finalize_species_metrics_csv(output_dir: Path) -> None:
    path = output_dir / "pure_species_metrics.csv"
    if not path.exists():
        return
    lock_path = output_dir / ".pure_species_metrics.csv.lock"
    with file_lock(lock_path):
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df.drop_duplicates(["pipeline", "model", "direction", "species"], keep="first")
        df = df.sort_values(["model", "pipeline", "direction", "species"]).reset_index(drop=True)
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms_to_run = ALGORITHMS if args.algorithm == "all" else [args.algorithm]
    table = load_pipeline_table(args.ranking, args.limit, args.pipeline_file)
    if args.write_pipeline_shards is not None:
        shard_dir = write_pipeline_shards(table, output_dir, args.write_pipeline_shards)
        print(f"Wrote {args.write_pipeline_shards} pipeline shard files to {shard_dir}", flush=True)
        return

    metadata_path = args.pure_metadata
    metadata = load_balanced_pure_metadata(
        metadata_path,
        max_per_species_domain=args.max_per_species_domain,
        seed=args.seed,
    )
    if not args.force:
        done = existing_pipeline_ids(output_dir, algorithms_to_run)
        table = table.loc[~table["pipeline_id"].astype(str).isin(done)].reset_index(drop=True)

    total_cpus = max(1, os.cpu_count() or 1)
    pipeline_jobs = max(1, int(args.pipeline_jobs))
    spectrum_jobs = args.spectrum_jobs or max(1, total_cpus // pipeline_jobs)
    # Spectrum preprocessing and model fitting happen in separate phases, so the
    # model phase can reuse all CPUs assigned to the current pipeline.
    inner_jobs = args.inner_jobs or max(1, total_cpus // pipeline_jobs)

    counts = metadata.groupby(["domain", "species_label"]).size()
    print(f"Pure metadata: {metadata_path}", flush=True)
    print(
        "Selected rows: "
        f"{len(metadata)} | species={counts.index.get_level_values('species_label').nunique()} | "
        f"MARISMA={(metadata['domain'] == 'marisma').sum()} | "
        f"DRIAMS-A={(metadata['domain'] == 'driams').sum()}",
        flush=True,
    )
    print(f"Output directory: {output_dir}", flush=True)
    print(
        f"Pipelines to run: {len(table)} | pipeline_jobs={pipeline_jobs} | "
        f"spectrum_jobs={spectrum_jobs} | inner_jobs={inner_jobs}",
        flush=True,
    )
    print(f"Methods: {', '.join(algorithms_to_run)}", flush=True)

    if table.empty:
        print("Nothing to do. Existing CSVs already contain all selected pipelines.", flush=True)
        return

    start = time.time()
    completed = 0
    failures: list[dict[str, str]] = []
    if pipeline_jobs == 1:
        for row in table.itertuples(index=False):
            pipeline_id = str(row.pipeline_id)
            completed += 1
            try:
                print(f"[{completed}/{len(table)}] start {pipeline_id}", flush=True)
                rows, direction_rows, species_rows = evaluate_pipeline(
                    row._asdict(),
                    metadata,
                    args.seed,
                    inner_jobs,
                    spectrum_jobs,
                    [] if args.disable_preprocessed_h5 else args.preprocessed_source_dirs,
                    args.benchmark_binning,
                    algorithms_to_run,
                )
                append_method_rows(output_dir, rows)
                append_direction_metric_rows(output_dir, direction_rows)
                append_species_metric_rows(output_dir, species_rows)
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (len(table) - completed) / rate if rate > 0 else float("nan")
                print(
                    f"[{completed}/{len(table)}] done {pipeline_id} | "
                    f"elapsed={elapsed/60:.1f} min | eta={remaining/60:.1f} min",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"pipeline_id": pipeline_id, "error": repr(exc)})
                print(f"[{completed}/{len(table)}] FAILED {pipeline_id}: {exc!r}", flush=True)
    else:
        executor = ProcessPoolExecutor(max_workers=pipeline_jobs)
        futures = {}
        try:
            futures = {
                executor.submit(
                    evaluate_pipeline,
                    row._asdict(),
                    metadata,
                    args.seed,
                    inner_jobs,
                    spectrum_jobs,
                    [] if args.disable_preprocessed_h5 else args.preprocessed_source_dirs,
                    args.benchmark_binning,
                    algorithms_to_run,
                ): str(row.pipeline_id)
                for row in table.itertuples(index=False)
            }
            for future in as_completed(futures):
                pipeline_id = futures[future]
                completed += 1
                try:
                    rows, direction_rows, species_rows = future.result()
                    append_method_rows(output_dir, rows)
                    append_direction_metric_rows(output_dir, direction_rows)
                    append_species_metric_rows(output_dir, species_rows)
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = (len(table) - completed) / rate if rate > 0 else float("nan")
                    print(
                        f"[{completed}/{len(table)}] done {pipeline_id} | "
                        f"elapsed={elapsed/60:.1f} min | eta={remaining/60:.1f} min",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append({"pipeline_id": pipeline_id, "error": repr(exc)})
                    print(f"[{completed}/{len(table)}] FAILED {pipeline_id}: {exc!r}", flush=True)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            print("Interrupted by Ctrl-C. Stopping pending pipeline jobs.", flush=True)
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    for algorithm in algorithms_to_run:
        finalize_method_csv(output_dir, algorithm)
    finalize_direction_metrics_csv(output_dir)
    finalize_species_metrics_csv(output_dir)

    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
        print(f"Completed with {len(failures)} failures. See {output_dir / 'failures.csv'}", flush=True)
    else:
        print("Completed without failures.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by Ctrl-C.", flush=True)
        raise SystemExit(130)
