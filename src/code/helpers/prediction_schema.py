"""Shared prediction table schema for tuning and evaluation outputs."""

import numpy as np
import pandas as pd


UNIFORM_COLUMNS = [
    "model",
    "mixture_id",
    "true_species_ids",
    "true_species_names",
    "true_abundances",
    "true_scores",
    "predicted_species_ids",
    "predicted_species_names",
    "predicted_abundances",
    "predicted_scores",
    "candidate_species_ids",
    "candidate_species_names",
    "candidate_scores",
    "candidate_abundances",
    "score_type",
    "abundance_type",
    "precision",
    "recall",
    "exact_match",
    "tp",
    "fp",
    "fn",
    "abundance_mae",
    "num_species",
    "avg_max_similarity",
    "threshold_used",
]


def cell_to_csv(value):
    """Convert list-like cells to stable comma-separated CSV strings."""
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = []
        for item in list(value):
            if item is None:
                parts.append("")
                continue
            try:
                if pd.isna(item):
                    parts.append("")
                    continue
            except Exception:
                pass
            parts.append(str(item))
        return ", ".join(parts)
    return value


def apply_uniform_schema(df: pd.DataFrame, threshold_used=None) -> pd.DataFrame:
    """Convert a prediction DataFrame to the shared output schema."""
    df = df.copy()

    rename_map = {
        "dataset_index": "mixture_id",
        "mixture_index": "mixture_id",
        "true_species": "true_species_names",
        "pred_species_ids": "predicted_species_ids",
        "pred_species_names": "predicted_species_names",
        "true_abundances_present_only": "true_abundances",
        "pred_abundances_present_only": "predicted_abundances",
        "pred_presence_probs_present_only": "predicted_scores",
        "candidate_species": "candidate_species_names",
        "k": "num_species",
    }
    df = df.rename(columns=rename_map)

    for col in UNIFORM_COLUMNS:
        if col not in df.columns:
            if col == "threshold_used":
                df[col] = np.nan if threshold_used is None else float(threshold_used)
            elif col in ("abundance_mae", "num_species", "avg_max_similarity", "threshold_used"):
                df[col] = np.nan
            else:
                df[col] = ""

    for col in [
        "true_species_ids", "true_species_names", "true_abundances", "true_scores",
        "predicted_species_ids", "predicted_species_names", "predicted_abundances", "predicted_scores",
        "candidate_species_ids", "candidate_species_names", "candidate_scores", "candidate_abundances",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(cell_to_csv)

    return df[UNIFORM_COLUMNS]
