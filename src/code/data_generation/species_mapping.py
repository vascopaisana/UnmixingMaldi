"""Canonical species id mapping shared by data generation, benchmarks, and transformer utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_SPECIES_MAPPING_CSV = REPO_ROOT / "data" / "species_mapping.csv"


def normalize_species_name(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.replace(" ", "_")


def load_global_species_mapping(path: Path | None = None) -> dict[int, str]:
    mapping_path = path or GLOBAL_SPECIES_MAPPING_CSV
    frame = pd.read_csv(mapping_path)
    required = {"species_id", "species_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{mapping_path} is missing required columns: {sorted(missing)}")
    mapping = {
        int(row.species_id): normalize_species_name(row.species_name)
        for row in frame.itertuples(index=False)
    }
    expected_ids = list(range(len(mapping)))
    actual_ids = sorted(mapping)
    if actual_ids != expected_ids:
        raise ValueError(f"{mapping_path} must contain contiguous species_id values from 0 to {len(mapping) - 1}.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"{mapping_path} contains duplicated species_name values.")
    return mapping


def load_global_species_names(path: Path | None = None) -> list[str]:
    mapping = load_global_species_mapping(path)
    return [mapping[idx] for idx in sorted(mapping)]


def load_global_species_to_id(path: Path | None = None) -> dict[str, int]:
    return {
        species_name: species_id
        for species_id, species_name in load_global_species_mapping(path).items()
    }


def map_species_names_to_global_ids(species_names: Iterable[object], path: Path | None = None) -> list[int]:
    species_to_id = load_global_species_to_id(path)
    normalized = [normalize_species_name(name) for name in species_names]
    missing = [name for name in normalized if name not in species_to_id]
    if missing:
        raise ValueError(f"Species are not present in global species mapping: {missing}")
    return [int(species_to_id[name]) for name in normalized]
