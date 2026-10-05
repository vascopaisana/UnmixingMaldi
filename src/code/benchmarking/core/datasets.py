"""Dataset resolution and leakage checks for benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, resolve_path


DRIAMS_MARISMA_PURES = "DRIAMS_MARISMA_PURES"


def canonical_reference_mode(value: str | None) -> str:
    if value is None:
        return DRIAMS_MARISMA_PURES
    normalized = str(value).strip().upper()
    aliases = {
        "PURES_DRIAMS_MARISMA": DRIAMS_MARISMA_PURES,
        DRIAMS_MARISMA_PURES: DRIAMS_MARISMA_PURES,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported reference mode {value!r}.")
    return aliases[normalized]


def reference_stage(mode: str) -> str:
    if mode == DRIAMS_MARISMA_PURES:
        return "PURES_DRIAMS_MARISMA"
    raise ValueError(f"Unsupported reference mode {mode!r}.")


def _count_csv(path: Path) -> tuple[int, int | None]:
    rows = 0
    species: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            for key in ("species", "species_name", "label", "organism"):
                value = row.get(key)
                if value:
                    species.add(value.strip().lower())
                    break
    return rows, len(species) or None


def _species_dirs(root: Path) -> tuple[int | None, int]:
    species_dirs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    return None, len(species_dirs)


def resolve_datasets(config: dict[str, Any]) -> dict[str, Any]:
    datasets = config.get("datasets") or {}
    marisma = resolve_path(datasets.get("marisma", {}).get("root", "data/raw/MARISMA"))
    driams_a = resolve_path(datasets.get("driams_a", {}).get("root", "data/raw/DRIAMS-A"))
    manifests = [
        resolve_path(path)
        for path in (datasets.get("experimental_mixtures", {}).get("manifests") or [])
    ]
    return {
        "marisma": {"root": marisma},
        "driams_a": {"root": driams_a},
        "experimental_mixtures": {"manifests": manifests},
    }


def dataset_summary(resolved: dict[str, Any]) -> dict[str, Any]:
    marisma_csv = PROJECT_ROOT / "data/pures/driams_marisma/marisma.csv"
    driams_csv = PROJECT_ROOT / "data/pures/driams_marisma/driams_a.csv"
    marisma_count, marisma_species = _count_csv(marisma_csv) if marisma_csv.exists() else _species_dirs(resolved["marisma"]["root"])
    driams_count, driams_species = _count_csv(driams_csv) if driams_csv.exists() else _species_dirs(resolved["driams_a"]["root"])
    manifest_rows = {}
    for manifest in resolved["experimental_mixtures"]["manifests"]:
        if manifest.exists():
            rows, species = _count_csv(manifest)
            manifest_rows[str(manifest)] = {"rows": rows, "species": species}
        else:
            manifest_rows[str(manifest)] = {"rows": None, "species": None, "missing": True}
    species_count = min(count for count in (marisma_species, driams_species) if count) if marisma_species or driams_species else None
    return {
        "training": {
            "marisma": marisma_count,
            "driams_a": driams_count,
            "total": (marisma_count or 0) + (driams_count or 0) if marisma_count and driams_count else None,
            "species": species_count,
        },
        "evaluation": {"experimental_mixture_manifests": manifest_rows},
    }


def validate_reference_mode(model: str, mode: str, datasets: dict[str, Any]) -> None:
    if mode == DRIAMS_MARISMA_PURES:
        return
