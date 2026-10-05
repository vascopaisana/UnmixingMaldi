"""Shared benchmark metrics and CSV column ordering."""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


MIXTURE_DATASET_SLUGS = (
    "local_mixture_set_a",
    "local_mixture_set_b",
    "local_mixture_set_c",
)
ABUNDANCE_DATASET_SLUGS = (
    "local_mixture_set_a",
    "local_mixture_set_c",
)
SPECIES_COUNT_VALUES = (2, 3, 4)
ABUNDANCE_BINS = [(low, low + 10) for low in range(10, 100, 10)]
TARGET_MIXTURE_SPECIES = (
    "candida_albicans",
    "candida_glabrata",
    "candida_parapsilosis",
    "candida_tropicalis",
)

SUBSET_METRIC_NAMES = (
    "prauc",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "mean_predicted_species",
    "mean_true_species",
    "n_samples",
)
SPECIES_METRIC_NAMES = (
    "prauc",
    "precision",
    "recall",
    "f1",
    "n_positive",
    "n_negative",
)
ABUNDANCE_METRIC_NAMES = (
    "n_occurrences",
    "tp",
    "fn",
    "recall",
    "score_mean",
)
MIXTURE_GLOBAL_METRIC_NAMES = (
    "prauc_global",
    "prauc_supported_species",
    "global_f1",
    "global_precision",
    "global_recall",
    "global_exact_match",
    "global_mean_predicted_species",
    "global_mean_true_species",
    "global_n_samples",
)
PURE_CROSS_DOMAIN_METRIC_NAMES = (
    "roc_auc_macro",
    "accuracy",
    "prauc_macro",
    "macro_precision",
    "macro_f1",
)
PURE_DIRECTION_METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_f1",
    "pr_auc_macro",
    "roc_auc_macro",
)
PURE_SPECIES_METRIC_NAMES = (
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
)
RF_HOLDOUT_METRIC_NAMES = (
    "balanced_accuracy",
    "f1",
    "average_precision",
    "roc_auc",
)
TRANSFORMER_SAMPLE_METRIC_NAMES = (
    "precision",
    "recall",
    "exact_match",
    "tp",
    "fp",
    "fn",
)
TRANSFORMER_SUMMARY_METRIC_NAMES = (
    "precision",
    "recall",
    "exact_match",
    "n_mixtures",
)


def safe_slug(value: Any) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def global_prauc(truth: np.ndarray, scores: np.ndarray) -> float:
    """Compute micro PR-AUC by flattening sample x species matrices."""
    y_true = np.asarray(truth, dtype=bool).ravel()
    y_score = np.asarray(scores, dtype=float).ravel()
    mask = np.isfinite(y_score)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true[mask].astype(int), y_score[mask])
    return float(auc(recall, precision))


def binary_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC for one binary target vector."""
    return global_prauc(np.asarray(y_true)[:, None], np.asarray(y_score)[:, None])


def samplewise_set_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-sample presence/absence metrics for multilabel predictions."""
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = np.logical_and(pred, truth).sum(axis=1)
    fp = np.logical_and(pred, ~truth).sum(axis=1)
    fn = np.logical_and(~pred, truth).sum(axis=1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    exact_match = np.all(pred == truth, axis=1).astype(float)
    values = {
        "precision": precision,
        "recall": recall,
        "exact_match": exact_match,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    return {metric: values[metric] for metric in TRANSFORMER_SAMPLE_METRIC_NAMES}


def aggregate_transformer_metrics(frame: Any) -> dict[str, float | int]:
    """Aggregate transformer CSV rows with the same micro metrics as mixture models."""
    if len(frame) == 0:
        values: dict[str, float | int] = {
            "precision": 0.0,
            "recall": 0.0,
            "exact_match": 0.0,
            "n_mixtures": 0,
        }
    else:
        tp = float(np.sum(frame["tp"])) if "tp" in frame else float("nan")
        fp = float(np.sum(frame["fp"])) if "fp" in frame else float("nan")
        fn = float(np.sum(frame["fn"])) if "fn" in frame else float("nan")
        precision = tp / (tp + fp) if np.isfinite(tp + fp) and (tp + fp) else float("nan")
        recall = tp / (tp + fn) if np.isfinite(tp + fn) and (tp + fn) else float("nan")
        values = {
            "precision": float(precision) if np.isfinite(precision) else float("nan"),
            "recall": float(recall) if np.isfinite(recall) else float("nan"),
            "exact_match": float(frame["exact_match"].mean()),
            "n_mixtures": int(len(frame)),
        }
    return {metric: values[metric] for metric in TRANSFORMER_SUMMARY_METRIC_NAMES}


def supported_species_mask(truth: np.ndarray, min_positive: int = 1) -> np.ndarray:
    truth = np.asarray(truth, dtype=bool)
    if truth.ndim != 2:
        raise ValueError("supported_species_mask expects a 2D truth matrix.")
    return truth.sum(axis=0) >= int(min_positive)


def supported_species_prauc(truth: np.ndarray, scores: np.ndarray, min_positive: int = 1) -> float:
    keep = supported_species_mask(truth, min_positive=min_positive)
    if not np.any(keep):
        return float("nan")
    return global_prauc(np.asarray(truth)[:, keep], np.asarray(scores)[:, keep])


def dataset_prauc_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    dataset_labels: Iterable[Any],
    prefix: str = "mixture",
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(list(dataset_labels), dtype=object)
    rows: dict[str, float] = {}
    for dataset in sorted(set(labels.tolist())):
        mask = labels == dataset
        if mask.any():
            rows[f"{prefix}_dataset_{safe_slug(dataset)}_prauc"] = global_prauc(truth[mask], scores[mask])
    return rows


def species_count_prauc_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    species_counts: Iterable[Any],
    prefix: str = "mixture",
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    counts = np.asarray(list(species_counts), dtype=int)
    rows: dict[str, float] = {}
    for count in sorted(set(counts.tolist())):
        mask = counts == count
        if mask.any():
            rows[f"{prefix}_species_count_{int(count)}_prauc"] = global_prauc(truth[mask], scores[mask])
    return rows


def species_prauc_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    species: Iterable[str],
    prefix: str = "mixture",
    min_positive: int = 1,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    rows: dict[str, float] = {}
    for idx, species_name in enumerate(species):
        y_true = truth[:, idx]
        if int(np.sum(y_true)) < int(min_positive):
            continue
        rows[f"{prefix}_species_{safe_slug(species_name)}_prauc"] = global_prauc(
            y_true[:, None],
            scores[:, idx][:, None],
        )
    return rows


def threshold_subset_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    if truth.size == 0:
        return {
            "n_samples": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "exact_match": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mean_predicted_species": float("nan"),
            "mean_true_species": float("nan"),
        }
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
        else float("nan")
    )
    return {
        "n_samples": int(truth.shape[0]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "exact_match": float(np.mean(np.all(pred == truth, axis=1))),
        "precision": float(precision) if np.isfinite(precision) else float("nan"),
        "recall": float(recall) if np.isfinite(recall) else float("nan"),
        "f1": float(f1) if np.isfinite(f1) else float("nan"),
        "mean_predicted_species": float(np.mean(pred.sum(axis=1))),
        "mean_true_species": float(np.mean(truth.sum(axis=1))),
    }


def binary_species_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute binary one-vs-rest metrics for one species."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
        else float("nan")
    )
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(np.sum(~y_true)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision) if np.isfinite(precision) else float("nan"),
        "recall": float(recall) if np.isfinite(recall) else float("nan"),
        "f1": float(f1) if np.isfinite(f1) else float("nan"),
        "specificity": float(specificity) if np.isfinite(specificity) else float("nan"),
        "false_positive_rate": float(fpr) if np.isfinite(fpr) else float("nan"),
    }


def _expand(prefixes: Iterable[str], metrics: Iterable[str]) -> list[str]:
    return [f"{prefix}_{metric}" for prefix in prefixes for metric in metrics]


def ordered_metric_columns(all_columns: Iterable[str]) -> list[str]:
    """Return a compact, analysis-oriented order for benchmark CSV columns."""
    all_cols = list(dict.fromkeys(all_columns))
    all_set = set(all_cols)

    identity = [
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
    ]
    mixture_global = [
        "mixture_prauc_global",
        "mixture_prauc_supported_species",
        "mixture_global_f1",
        "mixture_global_precision",
        "mixture_global_recall",
        "mixture_global_exact_match",
        "mixture_global_mean_predicted_species",
        "mixture_global_mean_true_species",
        "mixture_global_n_samples",
    ]
    dataset_prefixes = [f"mixture_dataset_{dataset}" for dataset in MIXTURE_DATASET_SLUGS]
    count_prefixes = [f"mixture_species_count_{count}" for count in SPECIES_COUNT_VALUES]
    candida_prefixes = [f"mixture_species_{species}" for species in TARGET_MIXTURE_SPECIES]
    abundance_prefixes = [
        f"mixture_abundance_{dataset}_{low}_{high}"
        for dataset in ABUNDANCE_DATASET_SLUGS
        for low, high in ABUNDANCE_BINS
    ]
    pure_global = [
        "pure_prauc_global",
        "pure_prauc_supported_species",
        "pure_global_f1",
        "pure_global_precision",
        "pure_global_recall",
        "pure_global_exact_match",
        "pure_global_mean_predicted_species",
        "pure_global_mean_true_species",
        "pure_global_n_samples",
    ]
    rf = [
        "rf_holdout_balanced_accuracy_mean",
        "rf_holdout_f1_mean",
        "rf_holdout_average_precision_mean",
        "rf_holdout_roc_auc_mean",
    ]
    similarity = [
        "similarity_intra_similarity_mean",
        "similarity_intra_similarity_std",
        "similarity_inter_similarity_mean",
        "similarity_inter_similarity_std",
        "similarity_closest_inter_similarity_mean",
        "similarity_closest_inter_similarity_std",
        "similarity_weighted_intra_similarity_mean",
        "similarity_weighted_intra_similarity_std",
        "similarity_weighted_inter_similarity_mean",
        "similarity_weighted_inter_similarity_std",
        "similarity_weighted_closest_inter_similarity_mean",
        "similarity_weighted_closest_inter_similarity_std",
    ]
    metadata = [
        "evaluation_unit",
        "n_mixture_spots",
        "n_estimated_abundance_spots",
        "n_training_spectra",
        "n_training_origins",
        "pipeline_parameters",
    ]

    preferred = [
        *identity,
        *mixture_global,
        *_expand(dataset_prefixes, SUBSET_METRIC_NAMES),
        *_expand(count_prefixes, SUBSET_METRIC_NAMES),
        *_expand(candida_prefixes, SPECIES_METRIC_NAMES),
        *_expand(abundance_prefixes, ABUNDANCE_METRIC_NAMES),
        *pure_global,
        *rf,
        *similarity,
        *metadata,
    ]
    ordered = [column for column in preferred if column in all_set]

    raw_count_suffixes = ("_tp", "_fp", "_fn", "_tn", "_n_spots", "_species_accuracy", "_specificity", "_false_positive_rate")
    raw_counts = sorted(
        column
        for column in all_cols
        if column not in ordered and (column in {"tp", "fp", "fn", "tn"} or column.endswith(raw_count_suffixes))
    )
    remaining = sorted(column for column in all_cols if column not in ordered and column not in raw_counts)
    return ordered + remaining + raw_counts
