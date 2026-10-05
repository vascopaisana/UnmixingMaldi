# src/utils/helpers_transformer.py
"""Helper functions for the MixtureTransformer model, including metrics computation, prediction formatting, and checkpoint loading."""
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from benchmarking.core.metrics import (
    aggregate_transformer_metrics,
    samplewise_set_metrics,
)

def compute_set_metrics_from_binary_tensors(true_bin: torch.Tensor, pred_bin: torch.Tensor):
    """Compute precision, recall, exact match, TP, FP, FN from binary tensors.

    Args:
        true_bin (torch.Tensor): Ground truth binary matrix (samples x species).
        pred_bin (torch.Tensor): Predicted binary matrix (samples x species).

    Returns:
        dict: Dictionary with keys:
            - "precision" (np.ndarray): Per-sample precision.
            - "recall" (np.ndarray): Per-sample recall.
            - "exact_match" (np.ndarray): Per-sample exact match (1 if identical).
            - "tp" (np.ndarray): True positives per sample.
            - "fp" (np.ndarray): False positives per sample.
            - "fn" (np.ndarray): False negatives per sample.
    """
    return samplewise_set_metrics(true_bin.cpu().numpy(), pred_bin.cpu().numpy())

def get_present_species_and_abundances(multihot_row: np.ndarray, abundance_row: np.ndarray):
    """Extract species IDs and their abundances from one-hot and abundance arrays.

    Args:
        multihot_row (np.ndarray): Binary array where 1 indicates species presence.
        abundance_row (np.ndarray): Array of abundance values (e.g., proportions).

    Returns:
        tuple: (ids, abunds)
            - ids (list[int]): List of species IDs where multihot_row > 0.
            - abunds (list[float]): Corresponding abundance values.
    """
    ids = [int(i) for i in np.where(multihot_row > 0)[0]]
    abunds = [float(abundance_row[i]) for i in ids]
    return ids, abunds

def floats_to_str(x, decimals=6):
    """Convert a list/array of floats to a comma-separated string with fixed decimals.

    Args:
        x (iterable): Collection of numbers.
        decimals (int, optional): Number of decimal places. Defaults to 6.

    Returns:
        str: Comma-separated string of formatted floats.
    """
    return ", ".join(f"{float(v):.{decimals}f}" for v in x)

def ints_to_str(x):
    """Convert a list/array of integers to a comma-separated string.

    Args:
        x (iterable): Collection of integer-convertible values.

    Returns:
        str: Comma-separated string of integers.
    """
    return ", ".join(str(int(v)) for v in x)

def predictions_to_dataframe(pred_batches, species_names=None):
    """Convert transformer prediction batches to a formatted DataFrame.

    Args:
        pred_batches (list): List of dictionaries, each containing tensors from
            a prediction batch. Expected keys: 'dataset_index', 'presence_probs',
            'presence_pred', 'proportions_pred', 'true_presence',
            'true_proportions', 'num_species', 'avg_max_similarity'.
        species_names (optional): Mapping from species ID to name. Can be a dict,
            list, or None. If None, IDs are converted to strings.

    Returns:
        pd.DataFrame: Sorted by dataset_index, containing columns for true/predicted
            species (IDs and names), abundances, probabilities, metrics, etc.
    """
    rows = []
    def sid_to_name(sid):
        """Resolve a numeric species ID to a display name."""
        if species_names is None:
            return str(int(sid))
        if isinstance(species_names, dict):
            return str(species_names.get(int(sid), str(int(sid))))
        if isinstance(species_names, (list, tuple)) and int(sid) < len(species_names):
            return str(species_names[int(sid)])
        return str(int(sid))

    for batch in pred_batches:
        idx = batch["dataset_index"].numpy()
        presence_probs = batch["presence_probs"].numpy()
        presence_pred = batch["presence_pred"].numpy()
        proportions_pred = batch["proportions_pred"].numpy()
        true_presence = batch["true_presence"].numpy()
        true_proportions = batch["true_proportions"].numpy()
        num_species = batch["num_species"].numpy()
        avg_max_sim = batch["avg_max_similarity"].numpy()
        validation_dataset = batch.get("validation_dataset")
        validation_dataset_index = batch.get("validation_dataset_index")

        metrics = compute_set_metrics_from_binary_tensors(
            torch.tensor(true_presence), torch.tensor(presence_pred)
        )
        for i in range(len(idx)):
            true_ids, true_abunds = get_present_species_and_abundances(true_presence[i], true_proportions[i])
            pred_ids, pred_abunds = get_present_species_and_abundances(presence_pred[i], proportions_pred[i])
            missed_true_ids = [sid for sid in true_ids if sid not in set(pred_ids)]
            candidate_ids = [
                int(sid)
                for sid in np.argsort(-presence_probs[i])
                if int(sid) not in set(pred_ids)
            ][:10]
            row = {
                "model": "transformer",
                "validation_dataset": validation_dataset,
                "validation_dataset_index": validation_dataset_index,
                "mixture_id": int(idx[i]),
                "true_species_ids": ints_to_str(true_ids),
                "true_species_names": ", ".join(sid_to_name(x) for x in true_ids),
                "true_abundances": floats_to_str(true_abunds),
                "true_scores": floats_to_str([presence_probs[i][sid] for sid in missed_true_ids]),
                "predicted_species_ids": ints_to_str(pred_ids),
                "predicted_species_names": ", ".join(sid_to_name(x) for x in pred_ids),
                "predicted_abundances": floats_to_str(pred_abunds),
                "predicted_scores": floats_to_str([presence_probs[i][sid] for sid in pred_ids]),
                "candidate_species_ids": ints_to_str(candidate_ids),
                "candidate_species_names": ", ".join(sid_to_name(x) for x in candidate_ids),
                "candidate_scores": floats_to_str([presence_probs[i][sid] for sid in candidate_ids]),
                "candidate_abundances": "",
                "score_type": "presence_probability",
                "abundance_type": "predicted_proportion",
                "num_species": int(num_species[i]),
                "avg_max_similarity": float(avg_max_sim[i]),
            }
            for metric, values in metrics.items():
                value = values[i]
                row[metric] = int(value) if metric in {"tp", "fp", "fn"} else float(value)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_columns = ["mixture_id"]
    if "validation_dataset_index" in df.columns and not df["validation_dataset_index"].isna().all():
        sort_columns = ["validation_dataset_index", "mixture_id"]
    return df.sort_values(sort_columns).reset_index(drop=True)

def aggregate_metrics_from_rows(df: pd.DataFrame):
    """Aggregate prediction rows with the shared transformer summary metrics.

    Args:
        df (pd.DataFrame): DataFrame containing per-sample TP/FP/FN and
            exact-match columns.

    Returns:
        dict: Dictionary with keys:
            - "precision" (float): Micro precision from summed TP/FP.
            - "recall" (float): Micro recall from summed TP/FN.
            - "exact_match" (float): Mean exact match.
            - "n_mixtures" (int): Number of samples.
    """
    return aggregate_transformer_metrics(df)

def save_test_outputs_csv(pred_batches, output_csv_path, species_names=None):
    """Save transformer predictions to a CSV file.

    Args:
        pred_batches (list): Output from a transformer's predict_step (same as
            for `predictions_to_dataframe`).
        output_csv_path (str or Path): Path where the CSV will be saved.
        species_names (optional): Mapping from species ID to name, passed to
            `predictions_to_dataframe`.

    Returns:
        pd.DataFrame: The generated DataFrame (also saved to disk).
    """
    df = predictions_to_dataframe(pred_batches, species_names)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"CSV saved to {output_csv_path}")
    return df

def save_test_summary_csv(test_outputs_df, output_csv_path, extra_info=None):
    """Compute summary metrics from a test outputs DataFrame and save as CSV.

    Args:
        test_outputs_df (pd.DataFrame): DataFrame returned by `save_test_outputs_csv`
            or `predictions_to_dataframe`.
        output_csv_path (str or Path): Path where the summary CSV will be saved.
        extra_info (dict, optional): Additional key-value pairs to include in the
            summary row (e.g., model name, timestamp). Defaults to None.

    Returns:
        dict: Summary dictionary containing aggregated metrics and any extra info.
    """
    summary = aggregate_metrics_from_rows(test_outputs_df)
    if extra_info:
        summary.update(extra_info)
    pd.DataFrame([summary]).to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Summary saved to {output_csv_path}")
    return summary

def load_pretrained_transformer_into_model(model, ckpt_path, strict=False):
    """Load pretrained transformer weights into a model's spectrum embedder.

    This function remaps keys from a pretrained checkpoint (which may have
    keys starting with 'transformer.') to the target model's spectrum embedder
    (keys starting with 'spectrum_embedder.'). Only compatible layers with
    matching shapes are loaded.

    Args:
        model (torch.nn.Module): Target model containing a 'spectrum_embedder'.
        ckpt_path (str): Path to the pretrained checkpoint (.ckpt or .pth).
        strict (bool, optional): If True, requires all keys to match exactly.
            Defaults to False (only compatible weights are loaded).

    Returns:
        torch.nn.Module: The model with loaded weights (modified in-place).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    remapped = {}
    for k, v in state.items():
        if k.startswith("transformer."):
            new_key = "spectrum_embedder." + k[len("transformer."):]
            remapped[new_key] = v
    compatible = {k: v for k, v in remapped.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} compatible weights from {ckpt_path}")
    return model

import torch

def infer_trf_dims_from_checkpoint(ckpt_path: str, default_output_dim: int = 256):
    """Infer transformer depth and embedding dimension from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    depth = 0

    for k in state.keys():
        if k.endswith("norm.weight") and ".layers." in k:
            parts = k.split(".")
            try:
                i = parts.index("layers")
                layer_idx = int(parts[i + 1])
                depth = max(depth, layer_idx + 1)
            except (ValueError, IndexError):
                pass

    if depth == 0:
        raise ValueError("Não foi possível inferir depth.")

    dim = None
    candidates = [
        "transformer.embed.embedder.0.weight",
        "transformer.embed.net.0.weight",
        "transformer.embed.layers.0.weight",
        "transformer.embed.cls.cls_embedding",
        "spectrum_embedder.embed.embedder.0.weight",
        "spectrum_embedder.embed.net.0.weight",
        "spectrum_embedder.embed.layers.0.weight",
        "spectrum_embedder.embed.cls.cls_embedding",
    ]

    for k in candidates:
        if k in state:
            if "cls_embedding" in k:
                dim = state[k].shape[-1]
            else:
                dim = state[k].shape[0]
            break

    if dim is None:
        # Safer fallback: look for a layer normalization weight vector.
        for k, v in state.items():
            if k.endswith("norm.weight") and ".layers." in k and v.ndim == 1:
                dim = v.shape[0]
                break

    if dim is None:
        raise ValueError("Não foi possível inferir dim.")

    print(f"Inferidos: depth={depth}, dim={dim}")

    return {
        "depth": depth,
        "dim": dim,
        "n_heads": 8,          # assumed, not inferred
        "dropout": 0.2,        # assumed, not inferred
        "cls": True,           # assumed, not inferred
        "reduce": "cls",       # assumed, not inferred
        "output_head_dim": default_output_dim,
    }
