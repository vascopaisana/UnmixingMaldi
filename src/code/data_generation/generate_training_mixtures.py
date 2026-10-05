#!/usr/bin/env python3
"""Generate efficient offline mixtures from pre-binning pure spectra."""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import h5py
import numpy as np
import yaml
from helpers.config_variants import load_config_with_variant
from helpers.local_env import load_local_env
from data_generation.species_mapping import load_global_species_mapping

PROJECT_ROOT = CODE_DIR.parents[1]
OUTPUT_DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DATA_DIR = OUTPUT_DATA_DIR
PROCESSED_DIR = SOURCE_DATA_DIR / "processed_h5"
MIXTURES_DIR = OUTPUT_DATA_DIR

try:
    from helpers.run_logging import save_run_manifest, setup_run_logging
except ImportError:
    def save_run_manifest(*args, **kwargs):
        return None

    def setup_run_logging(*args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        return None

logger = logging.getLogger(__name__)


def resolve_path(path_value: str, base: Path) -> Path:
    load_local_env()
    path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    if path.is_absolute():
        return path
    return base / path


def slug_value(value) -> str:
    text = str(value).strip().replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def pipeline_tag(cfg: Dict[str, Any]) -> str:
    explicit = cfg.get("pipeline_tag")
    if explicit:
        return slug_value(explicit)
    pre_steps = []
    for step in cfg.get("post_mix_preprocessing", []):
        name = str(step.get("name", step.get("step", ""))).lower()
        params = step.get("params", {}) or {}
        if name == "normalizer":
            continue
        if name == "varstabilizer":
            pre_steps.append(slug_value(params.get("method", "sqrt")))
        elif name:
            pre_steps.append(slug_value(name))
    binning = cfg["post_mix_binning"]
    aggregation = str(binning.get("aggregation", "mean")).lower()
    if aggregation == "mean":
        aggregation = "avg"
    bin_tag = f"b{slug_value(binning.get('step', 9))}da_{slug_value(aggregation)}"
    method_tag = "_".join([*pre_steps, bin_tag])
    return f"s_bc_n_vs__{method_tag}" if method_tag else "s_bc_n_vs"


def generation_tag(generation: Dict[str, Any]) -> str:
    alpha = generation.get("dirichlet_alpha", 1.0)
    if isinstance(alpha, (list, tuple)):
        alpha_tag = "-".join(slug_value(v) for v in alpha)
    else:
        alpha_tag = slug_value(alpha)
    return (
        f"random_n{int(generation['n_mixtures'])}"
        f"_k{int(generation.get('min_species', 2))}to{int(generation.get('max_species', 6))}"
        f"_alpha{alpha_tag}"
        f"_minab{slug_value(generation.get('min_abundance', 0.05))}"
        f"_seed{int(generation.get('random_seed', 42))}"
    )


def resolve_output_paths(cfg: Dict[str, Any]) -> tuple[Path, Path]:
    output_root = resolve_path(cfg.get("output_root", str(OUTPUT_DATA_DIR)), PROJECT_ROOT)
    pipe_tag = pipeline_tag(cfg)
    gen_tag = generation_tag(cfg["generation"])
    pure_dataset = slug_value(cfg.get("pure_dataset", "driams_marisma"))
    mixture_dataset = slug_value(cfg.get("mixture_dataset", "driams_marisma"))
    output_h5 = cfg.get("output_h5")
    pure_output_h5 = cfg.get("pure_output_h5")
    if output_h5:
        output_path = resolve_path(output_h5, output_root)
    else:
        output_path = output_root / "mixtures" / "IN_SILICO" / mixture_dataset / pipe_tag / f"mixtures_{gen_tag}.h5"
    if pure_output_h5:
        pure_output_path = resolve_path(pure_output_h5, output_root)
    else:
        pure_output_path = output_root / "pures" / pure_dataset / pipe_tag / f"pures_{pipe_tag}.h5"
    return pure_output_path, output_path


def normalize_species_names(names):
    if isinstance(names, dict):
        return {int(i): str(name) for i, name in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def load_reference_species_names(path: Optional[Path]) -> Optional[Dict[int, str]]:
    return load_global_species_mapping()


def build_indices(
    h5_path: Path,
    *,
    reference_species_names: Optional[Dict[int, str]],
):
    with h5py.File(h5_path, "r") as f:
        if "samples" not in f:
            raise ValueError(f"{h5_path} does not use the unified samples schema.")
        sample_species_ids = [np.asarray(value, dtype=np.int32) for value in f["samples/labels/species_id"][:]]
        sample_origin_datasets = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in f["samples/origin_dataset"][:]
        ]

    if reference_species_names is None:
        id_to_species = load_global_species_mapping()
    else:
        id_to_species = dict(reference_species_names)

    grouped = defaultdict(lambda: defaultdict(list))
    for idx, species_ids in enumerate(sample_species_ids):
        if len(species_ids) != 1:
            continue
        species_id = int(species_ids[0])
        if species_id not in id_to_species:
            continue
        grouped[str(sample_origin_datasets[idx])][species_id].append(idx)

    group_to_species_indices = {}
    for group_name, species_map in grouped.items():
        capped = {}
        for sid, indices in species_map.items():
            selected = sorted(indices)
            if selected:
                capped[int(sid)] = np.asarray(selected, dtype=np.int64)
        group_to_species_indices[str(group_name)] = capped

    counts = Counter()
    for group_name, species_map in group_to_species_indices.items():
        for sid, indices in species_map.items():
            counts[(group_name, sid)] = int(len(indices))
    vals = list(counts.values())
    logger.info(
        "Prepared pure index: groups=%s, species/group pairs=%s, min=%s, max=%s",
        sorted(group_to_species_indices),
        len(counts),
        min(vals) if vals else None,
        max(vals) if vals else None,
    )
    return group_to_species_indices, id_to_species


def load_spectra_cache(h5_path: Path):
    logger.info("Loading prebin spectra into RAM from %s", h5_path)
    with h5py.File(h5_path, "r") as f:
        mz_cache = [np.asarray(f["samples/spectrum/mz"][i], dtype=np.float32) for i in range(len(f["central"]))]
        intensity_cache = [np.asarray(f["samples/spectrum/intensities"][i], dtype=np.float32) for i in range(len(f["central"]))]
    total_points = sum(arr.size for arr in mz_cache)
    logger.info("Cached %s spectra with %.2f million m/z-intensity points", len(mz_cache), total_points / 1e6)
    return mz_cache, intensity_cache


def relative_source_path(path_value: str) -> str:
    """Return a stable path relative to the project data directory when possible."""
    path = Path(str(path_value))
    for base in (PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data", PROJECT_ROOT):
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def load_source_metadata(h5_path: Path) -> List[Dict[str, Any]]:
    with h5py.File(h5_path, "r") as f:
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in f["samples/id"][:]]
    return [{"id": sample_id} for sample_id in ids]


def sample_source_paths(sample_indices: np.ndarray, source_metadata: List[Dict[str, Any]]) -> List[str]:
    paths = []
    for sample_idx in sample_indices:
        row = source_metadata[int(sample_idx)] if int(sample_idx) < len(source_metadata) else {}
        path_value = row.get("id") or str(int(sample_idx))
        paths.append(relative_source_path(path_value))
    return paths


def sample_abundances(rng: np.random.RandomState, k: int, alpha_cfg, min_abundance: float) -> np.ndarray:
    if isinstance(alpha_cfg, (list, tuple)):
        alpha = float(alpha_cfg[int(rng.randint(len(alpha_cfg)))])
    else:
        alpha = float(alpha_cfg)
    abundances = rng.dirichlet(np.ones(k, dtype=np.float32) * alpha).astype(np.float32)
    if np.any(abundances < min_abundance):
        abundances = np.clip(abundances, min_abundance, None)
        abundances = abundances / abundances.sum()
    return abundances.astype(np.float32)


def post_mix_preprocess(intensity: np.ndarray, steps: List[Dict[str, Any]]) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=np.float32)
    for step in steps:
        name = str(step.get("name", step.get("step", "")))
        params = dict(step.get("params", {}))
        if name == "Normalizer":
            target_sum = float(params.get("sum", 1.0))
            total = float(np.sum(intensity))
            if not np.isfinite(total) or total <= 0:
                intensity = np.zeros_like(intensity, dtype=np.float32)
            else:
                intensity = (intensity / total * target_sum).astype(np.float32, copy=False)
        elif name == "VarStabilizer":
            method = str(params.get("method", "sqrt"))
            nonnegative = np.maximum(intensity, 0.0)
            if method == "sqrt":
                intensity = np.sqrt(nonnegative)
            elif method == "log":
                intensity = np.log1p(nonnegative)
            elif method == "log2":
                intensity = np.log2(nonnegative + 1.0)
            elif method == "log10":
                intensity = np.log10(nonnegative + 1.0)
            else:
                raise ValueError("VarStabilizer method must be 'sqrt', 'log', 'log2', or 'log10'.")
            intensity = intensity.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported post_mix_preprocessing step: {name!r}")
    return intensity


def pre_mix_preprocess(intensity: np.ndarray, steps: List[Dict[str, Any]]) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=np.float32)
    for step in steps:
        name = str(step.get("name", step.get("step", "")))
        params = dict(step.get("params", {}))
        if name != "Normalizer":
            raise ValueError(f"Unsupported pre_mix_preprocessing step: {name!r}")
        target_sum = float(params.get("sum", 1.0))
        total = float(np.sum(intensity))
        if not np.isfinite(total) or total <= 0:
            intensity = np.zeros_like(intensity, dtype=np.float32)
        else:
            intensity = (intensity / total * target_sum).astype(np.float32, copy=False)
    return intensity


def configure_bins(cfg: Dict[str, Any]):
    start = float(cfg.get("start", 2000.0))
    stop = float(cfg.get("stop", 20000.0))
    step = float(cfg.get("step", 9.0))
    aggregation = str(cfg.get("aggregation", "mean")).lower()
    if aggregation == "avg":
        aggregation = "mean"
    if aggregation not in {"sum", "mean", "max"}:
        raise ValueError("post_mix_binning aggregation must be 'sum', 'mean', or 'max'.")
    edges = np.arange(start, stop + 1e-8, step, dtype=np.float32)
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    return edges, centers, aggregation


def bin_mixture(mz: np.ndarray, intensity: np.ndarray, edges: np.ndarray, centers: np.ndarray, aggregation: str):
    if aggregation == "sum":
        binned, _ = np.histogram(mz, bins=edges, weights=intensity)
    elif aggregation == "mean":
        summed, _ = np.histogram(mz, bins=edges, weights=intensity)
        counts, _ = np.histogram(mz, bins=edges)
        binned = np.divide(summed, counts, out=np.zeros_like(summed, dtype=np.float64), where=counts > 0)
    else:
        bin_ids = np.searchsorted(edges, mz, side="right") - 1
        valid = (bin_ids >= 0) & (bin_ids < len(centers))
        binned = np.zeros(len(centers), dtype=np.float32)
        if np.any(valid):
            np.maximum.at(binned, bin_ids[valid], intensity[valid])
    return np.asarray(binned, dtype=np.float32)


def build_mixture_arrays(
    sample_indices: np.ndarray,
    abundances: np.ndarray,
    *,
    mz_cache,
    intensity_cache,
    mz_data,
    intensity_data,
    pre_mix_preprocessing_steps: List[Dict[str, Any]],
    preprocessing_steps: List[Dict[str, Any]],
    edges: np.ndarray,
    mz_axis: np.ndarray,
    aggregation: str,
):
    mz_parts = []
    intensity_parts = []
    for sample_idx, abundance in zip(sample_indices, abundances):
        if mz_cache is None:
            mz_part = np.asarray(mz_data[int(sample_idx)], dtype=np.float32)
            intensity_part = np.asarray(intensity_data[int(sample_idx)], dtype=np.float32)
        else:
            mz_part = mz_cache[int(sample_idx)]
            intensity_part = intensity_cache[int(sample_idx)]
        if pre_mix_preprocessing_steps:
            intensity_part = pre_mix_preprocess(intensity_part, pre_mix_preprocessing_steps)
        mz_parts.append(mz_part)
        intensity_parts.append(intensity_part * float(abundance))
    mix_mz = np.concatenate(mz_parts).astype(np.float32, copy=False)
    mix_intensity = np.concatenate(intensity_parts).astype(np.float32, copy=False)
    mix_intensity = post_mix_preprocess(mix_intensity, preprocessing_steps)
    return bin_mixture(mix_mz, mix_intensity, edges, mz_axis, aggregation)


def set_h5torch_attrs(dataset, mode: str, shape, dtype):
    dataset.attrs["mode"] = mode
    dataset.attrs["shape"] = np.asarray(shape)
    dataset.attrs["filled_to"] = int(shape[0]) if shape else 1
    dtype_name = np.dtype(dtype).name
    dataset.attrs["dtypes"] = np.asarray([dtype_name, dtype_name], dtype=object)


def create_output_h5(
    path: Path,
    n: int,
    n_bins: int,
    config_json: Dict[str, Any],
    mz_axis: np.ndarray,
    include_source_spectra: bool,
):
    dt_f32 = h5py.vlen_dtype(np.dtype("float32"))
    dt_i32 = h5py.vlen_dtype(np.dtype("int32"))
    dt_str = h5py.string_dtype(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    central = f.create_dataset("central", data=np.arange(n, dtype=np.int64))
    set_h5torch_attrs(central, "N-D", (n,), "int64")
    unstructured = f.create_group("unstructured")
    mz = unstructured.create_dataset("mz_axis", data=mz_axis.astype(np.float32))
    set_h5torch_attrs(mz, "N-D", (n_bins,), "float32")
    cfg_bytes = np.asarray([json.dumps(config_json)]).astype(bytes)
    cfg = unstructured.create_dataset("config_json", data=cfg_bytes)
    set_h5torch_attrs(cfg, "N-D", (1,), cfg.dtype)

    samples = f.create_group("samples")
    samples.create_dataset("id", shape=(n,), dtype=dt_str)
    samples.create_dataset("origin_dataset", shape=(n,), dtype=dt_str)
    samples.create_dataset("type", shape=(n,), dtype=dt_str)
    samples.create_dataset("num_species", shape=(n,), dtype=np.int32)
    spectrum = samples.create_group("spectrum")
    spectrum.create_dataset("mz", shape=(n,), dtype=dt_f32)
    spectrum.create_dataset("intensities", shape=(n,), dtype=dt_f32)
    labels = samples.create_group("labels")
    labels.create_dataset("species_id", shape=(n,), dtype=dt_i32)
    labels.create_dataset("species_name_json", shape=(n,), dtype=dt_str)
    labels.create_dataset("proportion", shape=(n,), dtype=dt_f32)
    for dataset in (
        samples["id"],
        samples["origin_dataset"],
        samples["type"],
        samples["num_species"],
        spectrum["mz"],
        spectrum["intensities"],
        labels["species_id"],
        labels["species_name_json"],
        labels["proportion"],
    ):
        mode = "vlen" if dataset.dtype.kind == "O" else "N-D"
        set_h5torch_attrs(dataset, mode, dataset.shape, dataset.dtype)

    compat = f.create_group("0")
    compat["mz"] = spectrum["mz"]
    compat["intensity"] = spectrum["intensities"]
    compat["species_ids"] = labels["species_id"]
    compat["abundances"] = labels["proportion"]
    compat["num_species"] = samples["num_species"]
    compat_avg = compat.create_dataset("avg_max_similarity", shape=(n,), dtype=np.float32)
    compat_avg[:] = 0.0
    set_h5torch_attrs(compat_avg, "N-D", (n,), "float32")
    compat_bin = compat.create_dataset("avg_max_similarity_bin", shape=(n,), dtype=np.int32)
    compat_bin[:] = -1
    set_h5torch_attrs(compat_bin, "N-D", (n,), "int32")
    compat_mode = compat.create_dataset("generation_mode", shape=(n,), dtype=np.int32)
    compat_mode[:] = 0
    set_h5torch_attrs(compat_mode, "N-D", (n,), "int32")
    compat_sample_indices = compat.create_dataset("sample_indices", shape=(n,), dtype=dt_i32)
    set_h5torch_attrs(compat_sample_indices, "vlen", (n,), dt_i32)
    dsets = {
        "id": samples["id"],
        "origin_dataset": samples["origin_dataset"],
        "type": samples["type"],
        "num_species": samples["num_species"],
        "mz": spectrum["mz"],
        "intensities": spectrum["intensities"],
        "species_id": labels["species_id"],
        "species_name_json": labels["species_name_json"],
        "proportion": labels["proportion"],
        "compat_sample_indices": compat_sample_indices,
    }
    if include_source_spectra:
        samples.create_dataset("source_spectra_json", shape=(n,), dtype=dt_str)
        set_h5torch_attrs(samples["source_spectra_json"], "N-D", (n,), samples["source_spectra_json"].dtype)
        dsets["source_spectra_json"] = samples["source_spectra_json"]
    return f, dsets


def write_pure_output_h5(
    *,
    source_h5: Path,
    output_h5: Path,
    id_to_species: Dict[int, str],
    config_json: Dict[str, Any],
    mz_axis: np.ndarray,
    edges: np.ndarray,
    aggregation: str,
    mz_cache,
    intensity_cache,
    pre_mix_preprocessing_steps: List[Dict[str, Any]],
    preprocessing_steps: List[Dict[str, Any]],
    batch_size: int,
) -> None:
    with h5py.File(source_h5, "r") as source:
        source_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["samples/id"][:]
        ]
        origin_datasets = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["samples/origin_dataset"][:]
        ]
        species_ids_rows = [np.asarray(value, dtype=np.int32) for value in source["samples/labels/species_id"][:]]
        pure_indices = [
            idx
            for idx, species_ids in enumerate(species_ids_rows)
            if len(species_ids) == 1 and int(species_ids[0]) in id_to_species
        ]
        pure_config = dict(config_json)
        pure_config.pop("generation", None)
        pure_config["id"] = output_h5.stem

        logger.info("Writing %s pure spectra to %s", len(pure_indices), output_h5)
        out, dsets = create_output_h5(output_h5, len(pure_indices), len(mz_axis), pure_config, mz_axis, include_source_spectra=False)
        try:
            mz_data = source["samples/spectrum/mz"]
            intensity_data = source["samples/spectrum/intensities"]
            for batch_start in range(0, len(pure_indices), batch_size):
                batch = pure_indices[batch_start:batch_start + batch_size]
                for offset, source_idx in enumerate(batch, start=batch_start):
                    species_id = int(species_ids_rows[source_idx][0])
                    binned = build_mixture_arrays(
                        np.asarray([source_idx], dtype=np.int32),
                        np.asarray([1.0], dtype=np.float32),
                        mz_cache=mz_cache,
                        intensity_cache=intensity_cache,
                        mz_data=mz_data,
                        intensity_data=intensity_data,
                        pre_mix_preprocessing_steps=pre_mix_preprocessing_steps,
                        preprocessing_steps=preprocessing_steps,
                        edges=edges,
                        mz_axis=mz_axis,
                        aggregation=aggregation,
                    )
                    dsets["id"][offset] = relative_source_path(source_ids[source_idx])
                    dsets["origin_dataset"][offset] = origin_datasets[source_idx]
                    dsets["type"][offset] = "pure"
                    dsets["num_species"][offset] = np.int32(1)
                    dsets["mz"][offset] = mz_axis.astype(np.float32)
                    dsets["intensities"][offset] = binned
                    dsets["species_id"][offset] = np.asarray([species_id], dtype=np.int32)
                    dsets["species_name_json"][offset] = json.dumps([str(id_to_species[species_id])])
                    dsets["proportion"][offset] = np.asarray([1.0], dtype=np.float32)
        finally:
            out.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(CODE_DIR / "configs" / "data_generation" / "generate_training_mixtures.example.yaml"),
    )
    parser.add_argument("--variant", default=None, help="Variant name for grouped YAML configs.")
    args = parser.parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    cfg = load_config_with_variant(config_path, variant=args.variant)

    run_log = getattr(main, "run_log", None)
    if run_log is not None:
        save_run_manifest(run_log, config_path=config_path, config=cfg)

    source_h5 = resolve_path(cfg["source_h5"], PROCESSED_DIR)
    pure_output_h5, output_h5 = resolve_output_paths(cfg)
    reference_h5 = cfg.get("reference_mixture_h5")
    reference_names = None
    if reference_h5:
        reference_names = load_reference_species_names(resolve_path(reference_h5, MIXTURES_DIR))

    generation = cfg["generation"]
    group_to_species_indices, id_to_species = build_indices(
        source_h5,
        reference_species_names=reference_names,
    )
    min_species = int(generation.get("min_species", 2))
    max_species = int(generation.get("max_species", 6))
    mode = str(generation.get("mode", "random")).lower()
    if mode != "random":
        raise ValueError("Only generation.mode='random' is supported for in-silico mixtures.")
    n_mixtures = int(generation["n_mixtures"])
    rng = np.random.RandomState(int(generation.get("random_seed", 42)))
    edges, mz_axis, aggregation = configure_bins(cfg["post_mix_binning"])
    n_bins = len(mz_axis)

    config_json = {
        "id": output_h5.stem,
        "origin_datasets": sorted(group_to_species_indices),
        "species_vocab": sorted(int(i) for i in id_to_species),
        "species_names": {str(int(i)): str(name) for i, name in id_to_species.items()},
        "n_species": len(id_to_species),
        "preprocessing": {
            "steps": cfg.get("post_mix_preprocessing", []),
            "binning": {**cfg["post_mix_binning"], "aggregation": aggregation},
        },
    }
    config_json["generation"] = {
        "mode": mode,
        "random_seed": int(generation.get("random_seed", 42)),
        "n_samples": n_mixtures,
        "min_species": min_species,
        "max_species": max_species,
        "dirichlet_alpha": generation.get("dirichlet_alpha", 1.0),
        "min_abundance": float(generation.get("min_abundance", 0.05)),
    }

    logger.info("Writing %s mixtures to %s", n_mixtures, output_h5)
    mz_cache = intensity_cache = None
    if bool(cfg.get("cache_spectra", True)):
        mz_cache, intensity_cache = load_spectra_cache(source_h5)
    source_metadata = load_source_metadata(source_h5)
    num_workers_cfg = generation.get("num_workers", cfg.get("num_workers", 1))
    if str(num_workers_cfg).lower() == "auto":
        num_workers = max(1, os.cpu_count() or 1)
    else:
        num_workers_value = int(num_workers_cfg)
        num_workers = max(1, os.cpu_count() or 1) if num_workers_value == -1 else max(1, num_workers_value)
    batch_size = max(1, int(generation.get("batch_size", cfg.get("batch_size", 512))))
    if num_workers > 1 and mz_cache is None:
        logger.warning("Parallel generation reads from RAM cache; forcing num_workers=1 because cache_spectra=false.")
        num_workers = 1
    logger.info("Generation workers=%s, batch_size=%s", num_workers, batch_size)

    write_pure_output_h5(
        source_h5=source_h5,
        output_h5=pure_output_h5,
        id_to_species=id_to_species,
        config_json=config_json,
        mz_axis=mz_axis,
        edges=edges,
        aggregation=aggregation,
        mz_cache=mz_cache,
        intensity_cache=intensity_cache,
        pre_mix_preprocessing_steps=cfg.get("pre_mix_preprocessing", []),
        preprocessing_steps=cfg.get("post_mix_preprocessing", []),
        batch_size=batch_size,
    )

    out, dsets = create_output_h5(output_h5, n_mixtures, n_bins, config_json, mz_axis, include_source_spectra=True)
    try:
        with h5py.File(source_h5, "r") as source:
            mz_data = source["samples/spectrum/mz"]
            intensity_data = source["samples/spectrum/intensities"]
            group_names = sorted(group_to_species_indices)

            def make_plan(i: int):
                k = int(rng.randint(min_species, max_species + 1))
                eligible_groups = [g for g in group_names if len(group_to_species_indices[g]) >= k]
                group = eligible_groups[int(rng.randint(len(eligible_groups)))]
                species_map = group_to_species_indices[group]
                species_pool = np.asarray(sorted(species_map), dtype=np.int32)
                species_ids = np.sort(rng.choice(species_pool, size=k, replace=False)).astype(np.int32)
                abundances = sample_abundances(
                    rng,
                    k,
                    generation.get("dirichlet_alpha", 1.0),
                    float(generation.get("min_abundance", 0.05)),
                )
                sample_indices = np.asarray([rng.choice(species_map[int(sid)]) for sid in species_ids], dtype=np.int32)
                return i, "mixture_in_silico", group, k, species_ids, abundances, sample_indices

            def build_from_plan(plan):
                i, sample_type, origin_dataset, k, species_ids, abundances, sample_indices = plan
                binned = build_mixture_arrays(
                    sample_indices,
                    abundances,
                    mz_cache=mz_cache,
                    intensity_cache=intensity_cache,
                    mz_data=mz_data,
                    intensity_data=intensity_data,
                    pre_mix_preprocessing_steps=cfg.get("pre_mix_preprocessing", []),
                    preprocessing_steps=cfg.get("post_mix_preprocessing", []),
                    edges=edges,
                    mz_axis=mz_axis,
                    aggregation=aggregation,
                )
                return i, sample_type, origin_dataset, k, species_ids, abundances, sample_indices, binned

            log_every = int(cfg.get("log_every", 1000))
            next_log = log_every
            executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
            try:
                for batch_start in range(0, n_mixtures, batch_size):
                    batch_end = min(batch_start + batch_size, n_mixtures)
                    plans = [make_plan(i) for i in range(batch_start, batch_end)]
                    if executor is None:
                        results = [build_from_plan(plan) for plan in plans]
                    else:
                        results = list(executor.map(build_from_plan, plans))
                    for i, sample_type, origin_dataset, k, species_ids, abundances, sample_indices, binned in results:
                        source_paths = sample_source_paths(sample_indices, source_metadata)
                        species_names = [str(id_to_species.get(int(sid), int(sid))) for sid in species_ids]
                        sample_id = f"synthetic/{output_h5.stem}/{i:06d}"
                        dsets["id"][i] = sample_id
                        dsets["origin_dataset"][i] = str(origin_dataset or "unknown")
                        dsets["type"][i] = sample_type
                        dsets["num_species"][i] = np.int32(k)
                        dsets["mz"][i] = mz_axis.astype(np.float32)
                        dsets["intensities"][i] = binned
                        dsets["species_id"][i] = species_ids
                        dsets["species_name_json"][i] = json.dumps(species_names)
                        dsets["proportion"][i] = abundances
                        dsets["compat_sample_indices"][i] = sample_indices
                        dsets["source_spectra_json"][i] = json.dumps(source_paths)
                    if batch_end >= next_log or batch_end == n_mixtures:
                        logger.info("Generated %s/%s mixtures", batch_end, n_mixtures)
                        while next_log <= batch_end:
                            next_log += log_every
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
    finally:
        out.close()
    logger.info("Saved offline post-mix mixtures to %s", output_h5)


if __name__ == "__main__":
    run_log = setup_run_logging("generate_training_mixtures")
    main.run_log = run_log
    main()
