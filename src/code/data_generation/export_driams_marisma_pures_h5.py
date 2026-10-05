#!/usr/bin/env python3
"""Export DRIAMS/MARISMA benchmark raw folders to a processed H5 file."""

import logging
import json
import argparse
import os
import random
import re
import ast
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd
import yaml
import h5py
import numpy as np

LOCAL_PROJECT_ROOT = CODE_DIR.parents[1]
CONFIG_DIR = CODE_DIR / "configs"
PROJECT_ROOT = LOCAL_PROJECT_ROOT

from helpers.SpectrumObject import SpectrumObject
from helpers.config_variants import load_config_with_variant
from helpers.local_env import load_local_env
from helpers import preprocess
from helpers.run_logging import redirect_std_streams, save_run_manifest, setup_run_logging
from data_generation.species_mapping import load_global_species_mapping, load_global_species_to_id

logger = logging.getLogger(__name__)


def normalize_species_name(name: str) -> str:
    """Match the species normalization used by integrate_datasets.py."""
    corrections = {
        "bacillus amyloliqefaciens": "bacillus amyloliquefaciens",
        "aspergillus westerdjikiae": "aspergillus westerdijkiae",
        "salmonella spp": "salmonella sp",
    }
    normalized = str(name).lower().strip()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\b(ssp|subsp)\.\s*", r"\1 ", normalized)
    normalized = re.sub(r"\bgroup\b", "group", normalized)
    normalized = re.sub(r"\bgroupe\b", "group", normalized)
    normalized = re.sub(r"\bgrupo\b", "group", normalized)
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = re.sub(r"\bspp?\b", "sp", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = corrections.get(normalized, normalized)
    return normalized.replace(" ", "_")


def resolve_path(path_value: str) -> Path:
    load_local_env()
    path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def source_species(root_dir: Path) -> Set[str]:
    return {
        normalize_species_name(path.name)
        for path in root_dir.iterdir()
        if path.is_dir()
    }


def _first_file(sample_dir: Path, suffixes: tuple[str, ...]) -> Path | None:
    for path in sorted(sample_dir.iterdir()):
        if path.is_file() and path.name.lower().endswith(suffixes):
            return path
    for child in sorted(sample_dir.iterdir()):
        if not child.is_dir():
            continue
        for path in sorted(child.iterdir()):
            if path.is_file() and path.name.lower().endswith(suffixes):
                return path
    for path in sorted(sample_dir.rglob("*")):
        if path.is_file() and path.name.lower().endswith(suffixes):
            return path
    return None


def _first_bruker_dir(sample_dir: Path) -> Path | None:
    direct = sample_dir / "1SLin"
    if direct.is_dir():
        names = {p.name.lower() for p in direct.iterdir() if p.is_file()}
        if "fid" in names and ("acqu" in names or "acqus" in names):
            return direct
    names = {p.name.lower() for p in sample_dir.iterdir() if p.is_file()}
    if "fid" in names and ("acqu" in names or "acqus" in names):
        return sample_dir
    for fid_path in sorted(sample_dir.rglob("*")):
        if not fid_path.is_file() or fid_path.name.lower() != "fid":
            continue
        folder = fid_path.parent
        names = {p.name.lower() for p in folder.iterdir() if p.is_file()}
        if "acqu" in names or "acqus" in names:
            return folder
    return None


def iter_spectrum_paths(species_dir: Path) -> Iterable[Path]:
    """Yield one SpectrumObject.load target per sample folder."""
    for sample_dir in sorted(path for path in species_dir.iterdir() if path.is_dir()):
        text_file = _first_file(sample_dir, (".txt", ".tsv", ".mzml", ".mzxml"))
        if text_file is not None:
            yield text_file
            continue
        bruker_dir = _first_bruker_dir(sample_dir)
        if bruker_dir is not None:
            yield bruker_dir


def build_metadata(config: Dict) -> pd.DataFrame:
    metadata_csv = config.get("metadata_csv")
    if metadata_csv:
        metadata_path = resolve_path(metadata_csv)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")
        df = pd.read_csv(metadata_path)
        if "species" not in df.columns and "species_names" in df.columns:
            df["species"] = df["species_names"].apply(lambda value: normalize_species_name(json.loads(value)[0]))
        if "dataset" not in df.columns and "origin" in df.columns:
            df["dataset"] = df["origin"].astype(str).str.replace("raw_spectra_", "", regex=False)
            df["dataset"] = df["dataset"].str.replace("_unzipped", "", regex=False).str.upper()
        if "sample_id" not in df.columns and "mixture_name" in df.columns:
            df["sample_id"] = df["mixture_name"]
        required = {"file_path", "dataset", "species", "sample_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Metadata CSV missing required columns: {sorted(missing)}")
        logger.info(
            "Loaded metadata CSV %s with %d spectra, %d species, %d datasets.",
            metadata_path,
            len(df),
            df["species"].nunique(),
            df["dataset"].nunique(),
        )
        return df

    sources = config["sources"]
    source_sets = []
    resolved_sources = []
    for source in sources:
        root_dir = resolve_path(source["root_dir"])
        if not root_dir.exists():
            raise FileNotFoundError(f"Source root does not exist: {root_dir}")
        resolved_sources.append({**source, "root_dir": root_dir})
        species = source_species(root_dir)
        source_sets.append(species)
        logger.info("%s: %d species found in %s", source["dataset"], len(species), root_dir)

    selection_cfg = config.get("species_selection", {})
    mode = str(selection_cfg.get("mode", "intersection")).lower()
    if mode == "intersection":
        selected_species = set.intersection(*source_sets)
    elif mode == "all":
        selected_species = set.union(*source_sets)
    else:
        raise ValueError("species_selection.mode must be 'intersection' or 'all'")

    max_per_species = selection_cfg.get("max_per_species_per_dataset")
    if max_per_species is not None:
        max_per_species = int(max_per_species)
    sampling_seed = selection_cfg.get("random_seed")
    if sampling_seed is not None:
        sampling_seed = int(sampling_seed)
    dataset_rngs = {
        str(source["dataset"]): random.Random(sampling_seed + idx)
        for idx, source in enumerate(resolved_sources)
    } if sampling_seed is not None else {}

    rows: List[Dict] = []
    for source in resolved_sources:
        dataset = str(source["dataset"])
        root_dir = source["root_dir"]
        for species_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
            species = normalize_species_name(species_dir.name)
            if species not in selected_species:
                continue
            paths = list(iter_spectrum_paths(species_dir))
            if max_per_species is not None:
                if sampling_seed is None or len(paths) <= max_per_species:
                    paths = paths[:max_per_species]
                else:
                    paths = dataset_rngs[dataset].sample(paths, max_per_species)
            for sample_idx, spectrum_path in enumerate(paths):
                rows.append(
                    {
                        "dataset": dataset,
                        "species": species,
                        "species_original": species_dir.name,
                        "sample_id": spectrum_path.parent.name,
                        "sample_index_in_species": sample_idx,
                        "file_path": str(spectrum_path.resolve()),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No spectra found for export.")
    logger.info(
        "Prepared %d spectra across %d species and %d datasets.",
        len(df),
        df["species"].nunique(),
        df["dataset"].nunique(),
    )
    return df


def build_preprocessor(pipeline_path: Path):
    with open(pipeline_path, "r") as handle:
        preprocess_cfg = yaml.safe_load(handle)
    steps = []
    for step_name in preprocess_cfg["pipeline"]:
        params = preprocess_cfg.get("params", {}).get(step_name, {})
        cls = getattr(preprocess, step_name)
        steps.append(cls(**params))
    return preprocess.SequentialPreprocessor(*steps), preprocess_cfg


def relative_dataset_path(path_value: str) -> str:
    path = Path(str(path_value))
    for base in (PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data", PROJECT_ROOT):
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def preprocessing_schema(preprocess_cfg: Dict) -> Dict:
    return {
        "steps": [
            {"name": name, "params": preprocess_cfg.get("params", {}).get(name, {})}
            for name in preprocess_cfg.get("pipeline", [])
        ],
    }


def export_unified_pures_h5(
    output_path: Path,
    dataframe: pd.DataFrame,
    preprocessor,
    preprocess_cfg: Dict,
    chunk_size: int,
) -> pd.DataFrame:
    species_to_id = load_global_species_to_id()
    id_to_species = load_global_species_mapping()
    records = []
    valid_indices = []
    for row_idx, row in dataframe.iterrows():
        try:
            spec = SpectrumObject.load(str(row["file_path"]))
            if preprocessor is not None:
                spec = preprocessor(spec)
            species_name = str(row["species"])
            species_id = species_to_id[species_name]
            species_label = str(ast.literal_eval(str(row["species_names"]))[0]) if "species_names" in dataframe.columns else species_name
            records.append(
                {
                    "id": relative_dataset_path(row["file_path"]),
                    "origin_dataset": str(row["dataset"]),
                    "mz": np.asarray(spec.mz, dtype=np.float32),
                    "intensities": np.asarray(spec.intensity, dtype=np.float32),
                    "species_id": np.asarray([species_id], dtype=np.int32),
                    "species_name": [species_label],
                    "proportion": np.asarray([1.0], dtype=np.float32),
                }
            )
            valid_indices.append(row_idx)
        except Exception:
            logger.exception("Failed to export pure spectrum %s", row.get("file_path"))

    if not records:
        raise ValueError("No spectra could be exported.")

    config = {
        "id": output_path.stem,
        "origin_datasets": sorted(str(value) for value in dataframe.loc[valid_indices, "dataset"].unique()),
        "species_vocab": sorted(id_to_species),
        "species_names": {str(idx): name for idx, name in id_to_species.items()},
        "n_species": len(id_to_species),
        "preprocessing": preprocessing_schema(preprocess_cfg),
    }
    dt_str = h5py.string_dtype(encoding="utf-8")
    dt_f32 = h5py.vlen_dtype(np.dtype("float32"))
    dt_i32 = h5py.vlen_dtype(np.dtype("int32"))
    def vlen_array(values):
        output = np.empty(len(values), dtype=object)
        output[:] = values
        return output

    def set_h5torch_attrs(dataset, mode: str, shape, dtype) -> None:
        dtype_name = np.dtype(dtype).name if not hasattr(dtype, "metadata") or dtype.metadata is None else str(dtype)
        dataset.attrs["mode"] = mode
        dataset.attrs["shape"] = np.asarray(shape)
        dataset.attrs["filled_to"] = int(shape[0]) if shape else 1
        dataset.attrs["dtypes"] = np.asarray([dtype_name, dtype_name], dtype=object)

    with h5py.File(output_path, "w") as handle:
        central = handle.create_dataset("central", data=np.arange(len(records), dtype=np.int64))
        set_h5torch_attrs(central, "N-D", (len(records),), "int64")
        unstructured = handle.create_group("unstructured")
        config_json = unstructured.create_dataset("config_json", data=np.asarray([json.dumps(config)]).astype(bytes))
        set_h5torch_attrs(config_json, "N-D", (1,), config_json.dtype)
        samples = handle.create_group("samples")
        samples.create_dataset("id", data=np.asarray([record["id"] for record in records], dtype=object), dtype=dt_str)
        samples.create_dataset("origin_dataset", data=np.asarray([record["origin_dataset"] for record in records], dtype=object), dtype=dt_str)
        samples.create_dataset("type", data=np.asarray(["pure"] * len(records), dtype=object), dtype=dt_str)
        samples.create_dataset("num_species", data=np.ones(len(records), dtype=np.int32))
        spectrum = samples.create_group("spectrum")
        spectrum.create_dataset("mz", data=vlen_array([record["mz"] for record in records]), dtype=dt_f32)
        spectrum.create_dataset("intensities", data=vlen_array([record["intensities"] for record in records]), dtype=dt_f32)
        labels = samples.create_group("labels")
        labels.create_dataset("species_id", data=vlen_array([record["species_id"] for record in records]), dtype=dt_i32)
        labels.create_dataset("species_name_json", data=np.asarray([json.dumps(record["species_name"]) for record in records], dtype=object), dtype=dt_str)
        labels.create_dataset("proportion", data=vlen_array([record["proportion"] for record in records]), dtype=dt_f32)
        compat = handle.create_group("0")
        compat["mz"] = spectrum["mz"]
        compat["intensity"] = spectrum["intensities"]
        compat["species_ids"] = labels["species_id"]
        compat["abundances"] = labels["proportion"]
        compat["num_species"] = samples["num_species"]
        for dataset in (compat["mz"], compat["intensity"], compat["species_ids"], compat["abundances"]):
            set_h5torch_attrs(dataset, "vlen", dataset.shape, dataset.dtype)
        set_h5torch_attrs(compat["num_species"], "N-D", (len(records),), "int32")
        avg = compat.create_dataset("avg_max_similarity", data=np.zeros(len(records), dtype=np.float32))
        set_h5torch_attrs(avg, "N-D", (len(records),), "float32")
        bins = compat.create_dataset("avg_max_similarity_bin", data=np.full(len(records), -1, dtype=np.int32))
        set_h5torch_attrs(bins, "N-D", (len(records),), "int32")
        generation = compat.create_dataset("generation_mode", data=np.zeros(len(records), dtype=np.int32))
        set_h5torch_attrs(generation, "N-D", (len(records),), "int32")
        sample_indices = np.empty(len(records), dtype=object)
        sample_indices[:] = [np.asarray([-1], dtype=np.int32)] * len(records)
        sample_ds = compat.create_dataset("sample_indices", data=sample_indices, dtype=dt_i32)
        set_h5torch_attrs(sample_ds, "vlen", (len(records),), sample_ds.dtype)
    return dataframe.loc[valid_indices].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "data_generation" / "export_driams_marisma_pures.example.yaml"),
        help="Path to export YAML config.",
    )
    parser.add_argument("--variant", default=None, help="Variant name for grouped YAML configs.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = CONFIG_DIR / config_path
    config = load_config_with_variant(config_path, variant=args.variant)

    run_log = getattr(main, "run_log", None)
    if run_log is not None:
        save_run_manifest(run_log, config_path=config_path, config=config)

    df = build_metadata(config)
    export_cfg = config["export"]
    output_path = resolve_path(export_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_name(output_path.stem + "_metadata.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Metadata CSV saved to %s", csv_path)

    pipeline_path = resolve_path(export_cfg["preprocessing_pipeline"])
    preprocessor, preprocess_cfg = build_preprocessor(pipeline_path)
    logger.info("Using preprocessing pipeline %s", pipeline_path)

    exported_df = export_unified_pures_h5(
        output_path=output_path,
        dataframe=df,
        preprocessor=preprocessor,
        preprocess_cfg=preprocess_cfg,
        chunk_size=int(export_cfg.get("chunk_size", 1000)),
    )
    exported_df.to_csv(csv_path, index=False)
    logger.info("Metadata CSV updated with %d exported spectra.", len(exported_df))
    logger.info("Processed H5 saved to %s", output_path)


if __name__ == "__main__":
    run_log = setup_run_logging("export_driams_marisma_pures_h5")
    main.run_log = run_log
    try:
        with redirect_std_streams("export_driams_marisma_pures_h5"):
            main()
    except Exception:
        logger.exception("Run failed")
        raise
