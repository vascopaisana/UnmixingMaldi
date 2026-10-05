#!/usr/bin/env python3
"""Predict species per mixture and write one CSV row per mixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

np = None
pd = None
yaml = None


CLASSICAL_MODELS = {"rf_cosine", "mahe", "nmf", "yang"}
ALL_MODELS = sorted([*CLASSICAL_MODELS, "transformer"])


def require_common_dependencies() -> None:
    global np, pd, yaml
    if np is None:
        import numpy as _np

        np = _np
    if pd is None:
        import pandas as _pd

        pd = _pd
    if yaml is None:
        import yaml as _yaml

        yaml = _yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--mixtures-h5", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--pures-h5", type=Path, help="Required for classical models.")
    parser.add_argument("--checkpoint", type=Path, help="Required for transformer predictions.")
    parser.add_argument("--config", type=Path, help="Optional model/config YAML.")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k-candidates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def read_yaml(path: Path | None) -> dict[str, Any]:
    require_common_dependencies()
    from helpers.local_env import expand_config_values, load_local_env

    if path is None:
        return {}
    load_local_env()
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return expand_config_values(data)


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_vlen(dataset, index: int) -> np.ndarray:
    require_common_dependencies()
    return np.asarray(dataset[index])


def read_species_names(handle: h5py.File, index: int, species_ids: np.ndarray) -> list[str]:
    path = "samples/labels/species_name_json"
    if path not in handle:
        return [str(int(sid)) for sid in species_ids]
    raw = decode_text(handle[path][index])
    try:
        value = json.loads(raw)
    except Exception:
        return [raw]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_pures(path: Path) -> dict[str, Any]:
    require_common_dependencies()
    import h5py

    with h5py.File(path, "r") as handle:
        intensities = np.asarray(handle["samples/spectrum/intensities"][:], dtype=np.float64)
        ids = [decode_text(value) for value in handle["samples/id"][:]]
        origin = (
            [decode_text(value) for value in handle["samples/origin_dataset"][:]]
            if "samples/origin_dataset" in handle
            else None
        )
        labels_ds = handle["samples/labels/species_id"]
        labels = []
        species_names: dict[Any, str] = {}
        for index in range(len(intensities)):
            species_ids = read_vlen(labels_ds, index).astype(int)
            if len(species_ids) != 1:
                raise ValueError(f"Pure sample {ids[index]!r} has {len(species_ids)} species labels.")
            sid = int(species_ids[0])
            labels.append(sid)
            names = read_species_names(handle, index, species_ids)
            if names:
                species_names[sid] = names[0]
        mz = np.asarray(handle["central"][:], dtype=np.float64) if "central" in handle else None
    return {
        "spectra": intensities,
        "labels": np.asarray(labels, dtype=object),
        "ids": ids,
        "origin": origin,
        "species_names": species_names,
        "mz": mz,
    }


def load_mixtures(path: Path) -> list[dict[str, Any]]:
    require_common_dependencies()
    import h5py

    rows = []
    with h5py.File(path, "r") as handle:
        intensities = np.asarray(handle["samples/spectrum/intensities"][:], dtype=np.float64)
        sample_ids = [decode_text(value) for value in handle["samples/id"][:]]
        labels_ds = handle["samples/labels/species_id"]
        proportions_ds = handle["samples/labels/proportion"]
        for index, sample_id in enumerate(sample_ids):
            species_ids = read_vlen(labels_ds, index).astype(int)
            proportions = read_vlen(proportions_ds, index).astype(float)
            rows.append(
                {
                    "index": index,
                    "id": sample_id,
                    "spectrum": intensities[index],
                    "true_species_ids": species_ids.tolist(),
                    "true_species_names": read_species_names(handle, index, species_ids),
                    "true_abundances": proportions.tolist(),
                    "num_species": int(len(species_ids)),
                }
            )
    return rows


def model_params(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    block = config.get("default", config)
    params = dict(block.get("model_parameters", block.get("params", {})) or {})
    if model_name == "rf_cosine":
        if "sklearn_n_jobs" in params:
            params["n_jobs"] = params.pop("sklearn_n_jobs")
        if "subtract_cross_species_training_baseline" in params:
            params["adjust_baseline"] = params.pop("subtract_cross_species_training_baseline")
        params.pop("cross_species_baseline_method", None)
    return params


def fit_classical_model(model_name: str, pures: dict[str, Any], params: dict[str, Any]):
    from models.mahe import MaheModel
    from models.nmf import NMFModel
    from models.rf_cosine import RFCosineModel
    from models.yang import YangModel

    spectra = pures["spectra"]
    labels = pures["labels"]
    species_names = pures["species_names"]
    if model_name == "rf_cosine":
        model = RFCosineModel(**params)
        model.fit(spectra, labels, species_names=species_names, groups=pures["origin"], mz_axis=pures["mz"])
        return model
    if model_name == "mahe":
        model = MaheModel(**params)
        species_ids = sorted(set(labels.tolist()))
        prototypes = np.vstack([spectra[labels == sid].mean(axis=0) for sid in species_ids])
        model.fit(prototypes, species_ids, species_names=species_names)
        return model
    if model_name == "nmf":
        model = NMFModel(**params)
        profile_ids = list(range(len(spectra)))
        profile_to_species = {profile_id: labels[profile_id] for profile_id in profile_ids}
        model.fit(spectra, profile_ids, profile_to_species)
        return model
    if model_name == "yang":
        model = YangModel(**params)
        profile_ids = list(range(len(spectra)))
        id_to_species = {profile_id: labels[profile_id] for profile_id in profile_ids}
        model.fit(spectra, profile_ids, id_to_species=id_to_species, species_names=species_names)
        return model
    raise ValueError(f"Unsupported classical model: {model_name}")


def precision_recall_row(true_ids: list[int], predicted_ids: list[int]) -> dict[str, Any]:
    true_set = set(int(value) for value in true_ids)
    pred_set = set(int(value) for value in predicted_ids)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "exact_match": int(true_set == pred_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def sorted_candidates(species_ids: list[Any], scores: np.ndarray, top_k: int) -> tuple[list[int], list[float]]:
    require_common_dependencies()
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    if top_k is not None and top_k > 0:
        order = order[:top_k]
    return [int(species_ids[idx]) for idx in order], [float(scores[idx]) for idx in order]


def normalize_prediction_output(output: tuple[Any, Any, Any]) -> tuple[list[int], list[float], dict[str, Any]]:
    """Handle small return-shape differences across baseline model classes."""
    first, second, third = output
    if isinstance(first, dict):
        details = first
        predicted_ids = second or []
        predicted_scores = third or []
    else:
        predicted_ids = first or []
        predicted_scores = second or []
        details = third if isinstance(third, dict) else {}
    return [int(value) for value in predicted_ids], [float(value) for value in predicted_scores], details


def predict_classical(args: argparse.Namespace) -> pd.DataFrame:
    require_common_dependencies()
    from helpers.prediction_schema import apply_uniform_schema

    if args.pures_h5 is None:
        raise ValueError("--pures-h5 is required for classical models.")
    config = read_yaml(args.config)
    params = model_params(config, args.model)
    if args.threshold is not None:
        if args.model == "rf_cosine":
            params.setdefault("threshold", args.threshold)
        elif args.model in {"mahe", "nmf"}:
            params.setdefault("active_threshold", args.threshold)
    pures = load_pures(args.pures_h5)
    model = fit_classical_model(args.model, pures, params)
    mixtures = load_mixtures(args.mixtures_h5)

    rows = []
    species_names = getattr(model, "species_names_", None) or getattr(model, "species_id_to_name_", {})
    for item in mixtures:
        predicted_ids, predicted_scores, details = normalize_prediction_output(model.predict(item["spectrum"]))
        all_species_ids = list(details.get("species_ids", getattr(model, "species_ids_", [])))
        all_scores = np.asarray(
            details.get("scores", details.get("species_scores", details.get("coeffs", []))),
            dtype=float,
        )
        if all_scores.size == len(all_species_ids):
            candidate_ids, candidate_scores = sorted_candidates(all_species_ids, all_scores, args.top_k_candidates)
        else:
            candidate_ids, candidate_scores = predicted_ids, predicted_scores
        row = {
            "model": args.model,
            "mixture_id": item["id"],
            "true_species_ids": item["true_species_ids"],
            "true_species_names": item["true_species_names"],
            "true_abundances": item["true_abundances"],
            "true_scores": "",
            "predicted_species_ids": predicted_ids,
            "predicted_species_names": [str(species_names.get(sid, sid)) for sid in predicted_ids],
            "predicted_abundances": predicted_scores,
            "predicted_scores": predicted_scores,
            "candidate_species_ids": candidate_ids,
            "candidate_species_names": [str(species_names.get(sid, sid)) for sid in candidate_ids],
            "candidate_scores": candidate_scores,
            "candidate_abundances": "",
            "score_type": details.get("score_type", details.get("score_mode", "")),
            "abundance_type": "model_score",
            "num_species": item["num_species"],
            "avg_max_similarity": np.nan,
            "threshold_used": args.threshold,
        }
        row.update(precision_recall_row(item["true_species_ids"], predicted_ids))
        rows.append(row)
    return apply_uniform_schema(pd.DataFrame(rows), threshold_used=args.threshold)


def predict_transformer(args: argparse.Namespace) -> pd.DataFrame:
    require_common_dependencies()
    from lightning import Trainer

    from helpers.prediction_schema import apply_uniform_schema
    from data_generation.mixture_data import MixtureDataModule
    from models.mixture_transformer import MixtureTransformer
    from transformer.helpers_transformer import predictions_to_dataframe

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for transformer predictions.")
    config = read_yaml(args.config)
    preprocessing = config.get("preprocessing", {}) if config else {}
    dm = MixtureDataModule(
        test_h5_path=args.mixtures_h5,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        fixed_length=preprocessing.get("fixed_length"),
        top_k_peaks=preprocessing.get("top_k_peaks"),
        target_bins=preprocessing.get("target_bins"),
    )
    dm.setup("test")
    model = MixtureTransformer.load_from_checkpoint(args.checkpoint, map_location="cpu")
    trainer = Trainer(logger=False, enable_checkpointing=False, accelerator="auto", devices=1)
    batches = trainer.predict(model, datamodule=dm)
    species_names = getattr(dm, "species_names", None)
    df = predictions_to_dataframe(batches, species_names=species_names)
    return apply_uniform_schema(df)


def main() -> None:
    args = parse_args()
    if args.model in CLASSICAL_MODELS:
        df = predict_classical(args)
    else:
        df = predict_transformer(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} mixture predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
