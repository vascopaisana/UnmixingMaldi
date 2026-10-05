"""Mixture generation, H5Torch persistence, and PyTorch data loading utilities."""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import h5py
import h5torch
import numpy as np
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

# ============================================================================
# Mixture Configuration
# ============================================================================

@dataclass
class MixtureConfig:
    """
    Configuration for mixture generation.

    Attributes:
        min_species: Minimum number of species in a mixture.
        max_species: Maximum number of species in a mixture.
        mixtures_per_k: Number of mixtures per species count (k).
        total_mixtures: Total number of mixtures (distributed evenly across k).
        dirichlet_alpha: Concentration parameter for Dirichlet distribution.
            Accepts a scalar, a list of scalar choices sampled per mixture, or a callable.
        random_seed: Seed for random number generator.
        min_abundance: Minimum allowed abundance for any species (after normalization).
        reuse_samples: Whether to allow reusing the same pure spectrum in multiple mixtures.
        abundance_templates: Predefined abundance vectors per k.
        species_vocab: Full output label vocabulary to preserve even when samples are filtered.
        species_names: Optional id -> name mapping for the full output label vocabulary.
        n_species: Full output label count.
        reset_usage_per_k: Whether to reset sample usage tracking when changing k.
    """
    min_species: int = 1
    max_species: int = 6
    mixtures_per_k: Optional[Dict[int, int]] = None
    total_mixtures: Optional[int] = None
    dirichlet_alpha: Union[float, List[float], Callable[[int], np.ndarray]] = 1.0
    random_seed: int = 42
    min_abundance: float = 0.05
    reuse_samples: bool = False
    reset_usage_per_k: bool = True
    abundance_templates: Optional[Dict[int, List[np.ndarray]]] = None
    species_vocab: Optional[List[int]] = None
    species_names: Optional[Dict[int, str]] = None
    n_species: Optional[int] = None


@dataclass(frozen=True)
class RandomGenerationSpec:
    max_attempts: int = 100
    fixed_abundance_ratios: Optional[Dict[int, List[float]]] = None
    reset_usage_per_k: bool = False


# ============================================================================
# Mixture Generator
# ============================================================================

class MixtureGenerator:
    """
    Generates synthetic mixtures from pure spectra.

    Random species combinations with Dirichlet-sampled abundances.
    """

    def __init__(self, samples: List[Dict[str, Any]], config: MixtureConfig):
        """
        Initialise the mixture generator.

        Args:
            samples: List of pure spectra, each must contain 'species_id', 'mz', 'intensity'.
            config: MixtureConfig instance.

        Raises:
            ValueError: If samples list is empty.
        """
        if not samples:
            raise ValueError("No samples were provided to MixtureGenerator.")

        self.samples = samples
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

        self.sample_intensities = np.asarray(
            [sample["intensity"] for sample in samples],
            dtype=np.float32,
        )

        # Map species id -> list of sample indices
        self.species_to_indices = defaultdict(list)
        for idx, sample in enumerate(samples):
            species_id = int(sample["species_id"])
            self.species_to_indices[species_id].append(idx)

        self.available_species = sorted(self.species_to_indices.keys())
        self.used = np.zeros(len(samples), dtype=bool)

        # Reference m/z vector (same for all samples)
        self.reference_mz = np.asarray(samples[0]["mz"], dtype=np.float32).copy()
        self.abundance_templates = config.abundance_templates or {}

    def _reset_usage(self) -> None:
        """Reset reusable sample tracking for a fresh generation block."""
        self.used = np.zeros(len(self.samples), dtype=bool)

    def _attempt_budget(self, target_count: int, max_attempts: int) -> int:
        """Translate per-target attempt settings into a total loop budget."""
        return max_attempts * max(1, int(target_count))

    def _resolve_fixed_abundances(
        self,
        abundance_map: Optional[Dict[int, List[float]]],
        k: int,
        *,
        field_name: str,
    ) -> Optional[np.ndarray]:
        """Validate and normalize fixed abundance definitions for a given k."""
        if not abundance_map or k not in abundance_map:
            return None
        abundances = np.asarray(abundance_map[k], dtype=np.float32)
        if len(abundances) != k:
            raise ValueError(f"{field_name}[{k}] has length {len(abundances)}, expected {k}.")
        return self._normalize_abundances(abundances)

    def _finalize_generation(self, mixtures: List[Dict[str, Any]], output_h5_path: str, mode_label: str) -> List[Dict[str, Any]]:
        """Persist generated mixtures and print a consistent summary."""
        self._save_h5torch(mixtures, output_h5_path)
        print(f"Generated {len(mixtures)} {mode_label} mixtures. Saved to {output_h5_path}")
        return mixtures

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    def _get_available_indices(self, species_id: int) -> List[int]:
        """Return list of sample indices for a species, respecting reuse_samples."""
        indices = self.species_to_indices.get(species_id, [])
        if self.config.reuse_samples:
            return indices
        return [idx for idx in indices if not self.used[idx]]

    def _resolve_target_counts(self) -> Dict[int, int]:
        """Determine how many mixtures to generate per k (number of species)."""
        ks = list(range(self.config.min_species, self.config.max_species + 1))

        if self.config.mixtures_per_k is not None:
            return {k: int(self.config.mixtures_per_k.get(k, 0)) for k in ks}

        if self.config.total_mixtures is not None:
            total = int(self.config.total_mixtures)
            if total <= 0:
                return {k: 0 for k in ks}
            base = total // len(ks)
            remainder = total % len(ks)
            return {k: base + (1 if i < remainder else 0) for i, k in enumerate(ks)}

        raise ValueError("Define mixtures_per_k or total_mixtures.")

    def _sample_species_ids(self, k: int) -> List[int]:
        """Sample k distinct species uniformly from available species."""
        if len(self.available_species) < k:
            raise ValueError(f"Need {k} species, but only {len(self.available_species)} available.")
        chosen = self.rng.choice(self.available_species, size=k, replace=False)
        return sorted(int(x) for x in chosen)

    def _normalize_abundances(self, abundances: np.ndarray) -> np.ndarray:
        """Normalise abundances to sum to 1 and enforce min_abundance."""
        abundances = np.asarray(abundances, dtype=np.float32)
        if abundances.ndim != 1:
            raise ValueError("abundances must be a 1D array.")
        if abundances.sum() <= 0:
            raise ValueError("abundances must have a positive sum.")

        abundances = abundances / abundances.sum()
        if np.any(abundances < self.config.min_abundance):
            abundances = np.clip(abundances, self.config.min_abundance, None)
            abundances = abundances / abundances.sum()
        return abundances.astype(np.float32)

    def _sample_abundances(self, k: int) -> np.ndarray:
        """Generate random abundances using Dirichlet or templates."""
        if k in self.abundance_templates:
            templates = self.abundance_templates[k]
            idx = self.rng.randint(len(templates))
            abundances = np.asarray(templates[idx], dtype=np.float32).copy()
            return self._normalize_abundances(abundances)

        alpha = self.config.dirichlet_alpha
        if callable(alpha):
            alpha = np.asarray(alpha(k), dtype=np.float32)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            alpha_values = np.asarray(alpha, dtype=np.float32)
            if alpha_values.ndim != 1 or alpha_values.size == 0:
                raise ValueError("dirichlet_alpha list must contain one or more numeric scalar choices.")
            if np.any(alpha_values <= 0):
                raise ValueError("dirichlet_alpha values must be positive.")
            chosen_alpha = float(alpha_values[self.rng.randint(alpha_values.size)])
            alpha = np.ones(k, dtype=np.float32) * chosen_alpha
        else:
            alpha = np.ones(k, dtype=np.float32) * float(alpha)
        if np.any(np.asarray(alpha, dtype=np.float32) <= 0):
            raise ValueError("dirichlet_alpha values must be positive.")

        last_abundances = None
        for _ in range(100):
            abundances = self.rng.dirichlet(alpha).astype(np.float32)
            last_abundances = abundances
            if np.all(abundances >= self.config.min_abundance):
                return abundances

        # Fallback: clip and renormalize
        abundances = np.clip(last_abundances, self.config.min_abundance, None)
        abundances = abundances / abundances.sum()
        return abundances.astype(np.float32)

    def _draw_sample_indices(self, species_ids: List[int]) -> Optional[List[int]]:
        """Select one spectrum per species, respecting reuse_samples."""
        chosen = []
        for sid in species_ids:
            available = self._get_available_indices(sid)
            if not available:
                return None
            idx = int(self.rng.choice(available))
            chosen.append(idx)
        return chosen

    def _build_mixture(
        self,
        species_ids: List[int],
        abundances: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single mixture dictionary.
        Returns None if sample indices are not available.
        """
        local_sample_indices = self._draw_sample_indices(species_ids)
        if local_sample_indices is None:
            return None

        # Determine abundances
        if abundances is None:
            abundances = self._sample_abundances(len(species_ids))
        else:
            abundances = self._normalize_abundances(abundances)

        # Build mixture spectrum: linear combination of pure spectra
        intensity_matrix = self.sample_intensities[local_sample_indices]
        mixture_intensity = abundances @ intensity_matrix

        if not self.config.reuse_samples:
            for idx in local_sample_indices:
                self.used[idx] = True

        return {
            "mz": self.reference_mz,
            "intensity": mixture_intensity.astype(np.float32, copy=False),
            "species_ids": np.asarray(species_ids, dtype=np.int32),
            "abundances": abundances.astype(np.float32, copy=False),
            "num_species": np.int32(len(species_ids)),
        }

    # ------------------------------------------------------------------------
    # Generation modes
    # ------------------------------------------------------------------------

    def generate_random_mixtures(
        self,
        output_h5_path: str,
        max_attempts: int = 100,
        fixed_abundance_ratios: Optional[Dict[int, List[float]]] = None,
        reset_usage_per_k: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Random mode: sample random species combinations and random abundances.

        Args:
            output_h5_path: Path to save the HDF5 file.
            max_attempts: Maximum attempts per mixture.
            fixed_abundance_ratios: Optional dict k -> fixed abundance vector.

        Returns:
            List of generated mixture dictionaries (also saved to disk).
        """
        target_counts = self._resolve_target_counts()
        mixtures = []

        for k, target_n in target_counts.items():
            if target_n <= 0:
                continue

            if reset_usage_per_k:
                self._reset_usage()

            created = 0
            attempts = 0
            max_total_attempts = self._attempt_budget(target_n, max_attempts)
            fixed_abundances = self._resolve_fixed_abundances(
                fixed_abundance_ratios,
                k,
                field_name="fixed_abundance_ratios",
            )

            while created < target_n and attempts < max_total_attempts:
                attempts += 1
                species_ids = self._sample_species_ids(k)

                mixture = self._build_mixture(
                    species_ids=species_ids,
                    abundances=fixed_abundances,
                )

                if mixture is not None:
                    mixtures.append(mixture)
                    created += 1

            if created < target_n:
                print(f"Warning: [k={k}] Generated {created}/{target_n} random mixtures.")

        return self._finalize_generation(mixtures, output_h5_path, "random")

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def generate_mixtures(
        self,
        output_h5_path: str,
        mode: str = "random",
        max_attempts: int = 100,
        fixed_abundance_ratios: Optional[Dict[int, List[float]]] = None,
        reset_usage_per_k: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate mixtures using the current random-original strategy.

        Args:
            output_h5_path: Path to save the HDF5 file.
            mode: Must be 'random'.
            max_attempts: Maximum attempts per mixture.
            fixed_abundance_ratios: Optional fixed abundances per k.

        Returns:
            List of mixture dictionaries (also saved to disk).
        """
        mode = str(mode).strip().lower()
        if mode != "random":
            raise ValueError("Only mode='random' is supported.")
        return self.generate_random_mixtures(
            output_h5_path, max_attempts, fixed_abundance_ratios, reset_usage_per_k
        )

    # ------------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------------

    def _save_h5torch(self, mixtures: List[Dict[str, Any]], path: str) -> None:
        """Save mixtures to HDF5 file using h5torch format."""
        n = len(mixtures)

        with h5torch.File(path, "w") as f:
            f.register(np.arange(n), "central")

            # Shared m/z axis plus variable-length mixture data
            f.register(self.reference_mz.astype(np.float32), "unstructured", name="mz_axis")
            f.register([m["intensity"] for m in mixtures], 0, name="intensity", mode="vlen")
            f.register([m["species_ids"] for m in mixtures], 0, name="species_ids", mode="vlen")
            f.register([m["abundances"] for m in mixtures], 0, name="abundances", mode="vlen")

            # Fixed-length arrays
            f.register(
                np.asarray([m["num_species"] for m in mixtures], dtype=np.int32),
                0, name="num_species"
            )

            # Save configuration as JSON
            mixture_species = sorted({int(sid) for m in mixtures for sid in m["species_ids"]})
            species_vocab = self.config.species_vocab or mixture_species
            species_vocab = sorted(int(sid) for sid in species_vocab)
            n_species = self.config.n_species
            if n_species is None:
                n_species = int(max(species_vocab) + 1) if species_vocab else 0
            species_names = self.config.species_names
            if species_names is not None:
                species_names = {str(int(sid)): str(name) for sid, name in species_names.items()}
            config_dict = {
                "min_species": self.config.min_species,
                "max_species": self.config.max_species,
                "mixtures_per_k": self.config.mixtures_per_k,
                "total_mixtures": self.config.total_mixtures,
                "dirichlet_alpha": np.asarray(self.config.dirichlet_alpha).tolist()
                if isinstance(self.config.dirichlet_alpha, (list, tuple, np.ndarray))
                else self.config.dirichlet_alpha if not callable(self.config.dirichlet_alpha) else "callable",
                "random_seed": self.config.random_seed,
                "min_abundance": self.config.min_abundance,
                "reuse_samples": self.config.reuse_samples,
                "species_vocab": species_vocab,
                "species_names": species_names,
                "n_species": int(n_species),
            }
            f.register(
                np.asarray([json.dumps(config_dict)]).astype(bytes),
                "unstructured",
                name="config_json",
            )


# ============================================================================
# Mixture Dataset and DataModule
# ============================================================================

class MixtureDataset(Dataset):
    """PyTorch Dataset for mixtures stored in h5torch format."""

    def __init__(self, h5_path: str):
        """
        Initialise the dataset.

        Args:
            h5_path: Path to the h5torch file.
        """
        self.h5_path = h5_path
        self.f = None
        self._shared_mz = None
        with h5py.File(h5_path, "r") as f:
            if "samples" not in f:
                raise ValueError(f"{h5_path} does not use the unified samples schema.")
            self.config = json.loads(f["unstructured/config_json"][0].decode())
            self._length = len(f["central"])
            if "unstructured/mz_axis" in f:
                self._shared_mz = np.asarray(f["unstructured/mz_axis"][:], dtype=np.float32)

    def __getstate__(self):
        """Return picklable state without the open H5Torch file handle."""
        state = self.__dict__.copy()
        if 'f' in state:
            del state['f']
        return state

    def __setstate__(self, state):
        """Restore dataset state after worker-process pickling."""
        self.__dict__.update(state)
        self.f = None

    def _ensure_open(self):
        """Open the h5torch file if not already open."""
        if self.f is None:
            self.f = h5py.File(self.h5_path, "r")

    def __len__(self):
        """Return the number of mixtures stored in the file."""
        return self._length

    def indices_by_num_species(
        self,
        min_species: Optional[int] = None,
        max_species: Optional[int] = None,
    ) -> List[int]:
        """Return row indices whose species count falls inside the requested range."""
        with h5py.File(self.h5_path, "r") as f:
            counts = np.asarray(f["samples/num_species"][:], dtype=np.int64)

        mask = np.ones_like(counts, dtype=bool)
        if min_species is not None:
            mask &= counts >= int(min_species)
        if max_species is not None:
            mask &= counts <= int(max_species)
        return np.nonzero(mask)[0].astype(int).tolist()

    def __getitem__(self, idx):
        """
        Get a single mixture.

        Args:
            idx: Index of the mixture.

        Returns:
            dict: Contains model-facing tensors derived from the unified samples schema.
        """
        self._ensure_open()
        species_ids = torch.as_tensor(self.f["samples/labels/species_id"][idx], dtype=torch.long)
        if self._shared_mz is not None:
            mz = torch.as_tensor(self._shared_mz, dtype=torch.float32)
        else:
            mz = torch.as_tensor(self.f["samples/spectrum/mz"][idx], dtype=torch.float32)
        num_species = torch.as_tensor(self.f["samples/num_species"][idx], dtype=torch.long)
        return {
            "dataset_index": torch.tensor(idx, dtype=torch.long),
            "mz": mz,
            "intensity": torch.as_tensor(self.f["samples/spectrum/intensities"][idx], dtype=torch.float32),
            "species_ids": species_ids,
            "abundances": torch.as_tensor(self.f["samples/labels/proportion"][idx], dtype=torch.float32),
            "num_species": num_species,
        }

    def close(self):
        """Close the underlying h5torch file."""
        if self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass
            self.f = None

    def __del__(self):
        """Best-effort cleanup for an open H5Torch file handle."""
        self.close()


class MixtureDataModule(LightningDataModule):
    """
    LightningDataModule for mixtures.

    Handles loading of train/val/test splits, optional fixed-length padding,
    and top‑k peak selection.
    """

    def __init__(
        self,
        train_h5_path: Optional[Union[str, Path]] = None,
        train_h5_paths: Optional[List[Union[str, Path]]] = None,
        train_h5_filter_bounds: Optional[List[Tuple[Optional[int], Optional[int]]]] = None,
        train_h5_fractions: Optional[List[Optional[float]]] = None,
        val_h5_path: Optional[Union[str, Path]] = None,
        val_h5_paths: Optional[List[Union[str, Path]]] = None,
        val_global_h5_count: Optional[int] = None,
        val_h5_filter_bounds: Optional[List[Tuple[Optional[int], Optional[int]]]] = None,
        val_h5_names: Optional[List[str]] = None,
        test_h5_path: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        fixed_length: Optional[int] = None,
        top_k_peaks: Optional[int] = None,
        pure_top_k_peaks: Optional[int] = None,
        target_bins: Optional[int] = None,
        min_species: Optional[int] = None,
        max_species: Optional[int] = None,
        val_min_species: Optional[int] = None,
        val_max_species: Optional[int] = None,
        train_fraction: Optional[float] = None,
        train_max_samples: Optional[int] = None,
        train_subsample_seed: int = 42,
    ):
        """
        Initialise the DataModule.

        Args:
            train_h5_path: Path to training HDF5 file (single file).
            train_h5_paths: List of paths for training (alternative to single file).
            train_h5_filter_bounds: Per-training-H5 species-count filters.
            train_h5_fractions: Per-training-H5 deterministic subsampling fractions.
            val_h5_path: Path to validation HDF5 file.
            val_h5_paths: List of paths for validation (alternative to single file).
            val_global_h5_count: Number of validation H5 files included in global validation metrics.
            val_h5_filter_bounds: Per-validation-H5 species-count filters.
            val_h5_names: Optional metric labels for validation H5 files.
            test_h5_path: Path to test HDF5 file.
            batch_size: Batch size for DataLoaders.
            shuffle: Whether to shuffle the training DataLoader.
            num_workers: Number of worker processes.
            fixed_length: If provided, pad/truncate spectra to this length.
            top_k_peaks: If provided, keep only the top‑k highest peaks (by intensity)
                after sorting by m/z.
            pure_top_k_peaks: If provided, zero all but the top-k intensities for
                pure spectra (num_species == 1) while preserving the m/z grid.
            target_bins: If provided, coarsen collated spectra to this number of
                ordered bins by summing intensity windows and averaging m/z.
            min_species: Optional lower bound for filtering samples by species count.
            max_species: Optional upper bound for filtering samples by species count.
            val_min_species: Optional validation-only lower bound for species count.
            val_max_species: Optional validation-only upper bound for species count.
        """
        super().__init__()
        self.train_h5_path = train_h5_path
        self.train_h5_paths = train_h5_paths
        self.train_h5_filter_bounds = train_h5_filter_bounds
        self.train_h5_fractions = train_h5_fractions
        self.val_h5_path = val_h5_path
        self.val_h5_paths = val_h5_paths
        self.val_global_h5_count = val_global_h5_count
        self.val_h5_filter_bounds = val_h5_filter_bounds
        self.val_h5_names = val_h5_names
        self.test_h5_path = test_h5_path
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.persistent_workers = self.num_workers > 0 if persistent_workers is None else bool(persistent_workers)
        self.prefetch_factor = prefetch_factor
        self.fixed_length = fixed_length
        self.top_k_peaks = top_k_peaks
        self.pure_top_k_peaks = pure_top_k_peaks
        self.target_bins = target_bins
        self.min_species = min_species
        self.max_species = max_species
        self.val_min_species = val_min_species
        self.val_max_species = val_max_species
        self.train_fraction = train_fraction
        self.train_max_samples = train_max_samples
        self.train_subsample_seed = int(train_subsample_seed)

        self.train_dataset = None
        self.val_dataset = None
        self.val_datasets = None
        self.val_dataset_names = None
        self.test_dataset = None

        # Load configuration from the first training file.
        if self.train_h5_paths is not None:
            first_path = self.train_h5_paths[0]
        else:
            first_path = self.train_h5_path
        if first_path is None:
            raise ValueError("No training data provided.")
        train_ds = MixtureDataset(str(first_path))
        self.config = train_ds.config
        self.n_species = self._resolve_n_species(self.config)
        self.species_names = self.config.get("species_names", None)
        train_ds.close()
        self._validate_configs()

    def setup(self, stage=None):
        """Set up the datasets. Called automatically by Lightning."""
        if self.train_dataset is None:
            if self.train_h5_paths is not None:
                datasets = []
                skipped_paths = []
                for idx, path in enumerate(self.train_h5_paths):
                    if self.train_h5_filter_bounds is not None and idx < len(self.train_h5_filter_bounds):
                        min_species, max_species = self.train_h5_filter_bounds[idx]
                        use_default_filter = False
                    else:
                        min_species, max_species = None, None
                        use_default_filter = True
                    filtered = self._maybe_filter_dataset(
                        MixtureDataset(str(path)),
                        min_species=min_species,
                        max_species=max_species,
                        allow_empty=True,
                        use_default_filter=use_default_filter,
                    )
                    if filtered is None:
                        skipped_paths.append(str(path))
                        continue
                    if self.train_h5_fractions is not None and idx < len(self.train_h5_fractions):
                        filtered = self._subsample_dataset(filtered, fraction=self.train_h5_fractions[idx])
                    datasets.append(filtered)
                if not datasets:
                    raise ValueError(
                        "Species-count filtering produced an empty training dataset across all train_h5_paths "
                        f"with min_species={self.min_species}, max_species={self.max_species}. "
                        f"Skipped paths: {skipped_paths}"
                    )
                self.train_dataset = torch.utils.data.ConcatDataset(datasets)
            elif self.train_h5_path is not None:
                self.train_dataset = self._maybe_filter_dataset(MixtureDataset(str(self.train_h5_path)))
            else:
                raise ValueError("No training data provided.")
            if self.train_h5_fractions is None:
                self.train_dataset = self._maybe_subsample_train_dataset(self.train_dataset)

        if self.val_dataset is None:
            if self.val_h5_paths is not None:
                self.val_datasets = []
                self.val_dataset_names = []
                for idx, path in enumerate(self.val_h5_paths):
                    if self.val_h5_filter_bounds is not None and idx < len(self.val_h5_filter_bounds):
                        min_species, max_species = self.val_h5_filter_bounds[idx]
                    else:
                        min_species, max_species = self.val_min_species, self.val_max_species
                    dataset = self._maybe_filter_dataset(
                        MixtureDataset(str(path)),
                        min_species=min_species,
                        max_species=max_species,
                        use_default_filter=False,
                    )
                    self.val_datasets.append(dataset)
                    if self.val_h5_names is not None and idx < len(self.val_h5_names):
                        self.val_dataset_names.append(str(self.val_h5_names[idx]))
                    else:
                        self.val_dataset_names.append(str(path))
                self.val_dataset = torch.utils.data.ConcatDataset(self.val_datasets)
            elif self.val_h5_path is not None:
                self.val_dataset = self._maybe_filter_dataset(
                    MixtureDataset(str(self.val_h5_path)),
                    min_species=self.val_min_species,
                    max_species=self.val_max_species,
                    use_default_filter=False,
                )
        if self.test_dataset is None and self.test_h5_path is not None:
            self.test_dataset = self._maybe_filter_dataset(
                MixtureDataset(str(self.test_h5_path)),
                use_default_filter=False,
            )

    def _maybe_filter_dataset(
        self,
        dataset: MixtureDataset,
        min_species: Optional[int] = None,
        max_species: Optional[int] = None,
        allow_empty: bool = False,
        use_default_filter: bool = True,
    ):
        """Apply species-count filtering while preserving the existing default behavior."""
        if use_default_filter and min_species is None:
            min_species = self.min_species
        if use_default_filter and max_species is None:
            max_species = self.max_species
        if min_species is None and max_species is None:
            return dataset
        indices = dataset.indices_by_num_species(min_species, max_species)
        if not indices:
            if allow_empty:
                dataset.close()
                return None
            raise ValueError(
                "Species-count filtering produced an empty dataset for "
                f"{dataset.h5_path} with min_species={min_species}, max_species={max_species}."
            )
        return Subset(dataset, indices)

    def _maybe_subsample_train_dataset(self, dataset):
        """Deterministically keep only a fraction or maximum count of training rows."""
        return self._subsample_dataset(dataset, fraction=self.train_fraction, max_samples=self.train_max_samples)

    def _subsample_dataset(self, dataset, fraction=None, max_samples=None):
        """Deterministically keep only a fraction or maximum count of dataset rows."""
        n_samples = len(dataset)
        target = n_samples
        if fraction is not None:
            fraction = float(fraction)
            if not 0 < fraction <= 1:
                raise ValueError(f"train_fraction must be in (0, 1], got {fraction}.")
            target = min(target, max(1, int(round(n_samples * fraction))))
        if max_samples is not None:
            target = min(target, int(max_samples))
        if target >= n_samples:
            return dataset
        rng = np.random.RandomState(self.train_subsample_seed)
        indices = np.sort(rng.choice(n_samples, size=target, replace=False)).astype(int).tolist()
        return Subset(dataset, indices)

    def _extract_species_info(self, cfg: Dict):
        """Extract n_species and species_names from a config dict."""
        n = cfg.get("n_species", None)
        names = cfg.get("species_names", None)
        if n is None and names is not None:
            n = len(names)
        return n, names

    def _normalize_species_names(self, names):
        if names is None:
            return None
        if isinstance(names, dict):
            return {int(i): str(name) for i, name in names.items()}
        return {i: str(name) for i, name in enumerate(names)}

    def _resolve_n_species(self, cfg: Dict) -> int:
        n, _ = self._extract_species_info(cfg)
        if n is None:
            raise ValueError("Could not find n_species in dataset config.")
        return int(n)

    def _validate_configs(self):
        """Ensure that all splits have consistent species vocabularies."""
        ref_n, ref_names = self._extract_species_info(self.config)
        split_paths = []
        if self.val_h5_paths is not None:
            split_paths.extend(("val", path) for path in self.val_h5_paths)
        else:
            split_paths.append(("val", self.val_h5_path))
        split_paths.append(("test", self.test_h5_path))
        for split_name, path in split_paths:
            if path is None:
                continue
            ds = MixtureDataset(path)
            n, names = self._extract_species_info(ds.config)
            ds.close()
            if int(n) != int(ref_n):
                raise ValueError(f"Inconsistent n_species in {split_name}: {n} vs {ref_n}")
            if (
                ref_names is not None
                and names is not None
                and self._normalize_species_names(names) != self._normalize_species_names(ref_names)
            ):
                raise ValueError(f"Inconsistent species_names in {split_name}.")

    def _pad_or_truncate_1d(self, x: torch.Tensor, target_len: int, pad_value: float = 0.0):
        """Pad or truncate a 1D tensor to a fixed length."""
        if x.numel() == target_len:
            return x
        if x.numel() > target_len:
            return x[:target_len]
        out = torch.full((target_len,), pad_value, dtype=x.dtype)
        out[:x.numel()] = x
        return out

    def _keep_topk_peaks(self, mz: torch.Tensor, intensity: torch.Tensor, k: Optional[int] = None):
        """
        Keep only the top‑k peaks by intensity, then sort by m/z.

        Args:
            mz: m/z values.
            intensity: corresponding intensities.

        Returns:
            (mz_sorted, intensity_sorted) after filtering and sorting.
        """
        if k is None:
            k = self.top_k_peaks
        if k is None:
            order = torch.argsort(mz)
            return mz[order], intensity[order]
        k = int(k)
        if k <= 0 or intensity.numel() <= k:
            order = torch.argsort(mz)
            return mz[order], intensity[order]
        k = min(k, intensity.numel())
        top_idx = torch.topk(intensity, k=k, largest=True).indices
        mz_top = mz[top_idx]
        int_top = intensity[top_idx]
        order = torch.argsort(mz_top)
        return mz_top[order], int_top[order]

    def _mask_to_topk_intensity(self, intensity: torch.Tensor, k: Optional[int]):
        """Preserve the grid while zeroing all but the k largest intensities."""
        if k is None:
            return intensity
        k = int(k)
        if k <= 0 or intensity.numel() <= k:
            return intensity
        top_idx = torch.topk(intensity, k=k, largest=True).indices
        masked = torch.zeros_like(intensity)
        masked[top_idx] = intensity[top_idx]
        return masked

    def _rebin_1d_to_target_bins(self, mz: torch.Tensor, intensity: torch.Tensor):
        """Coarsen one ordered spectrum before batch padding."""
        if self.target_bins is None:
            return mz, intensity
        target_bins = int(self.target_bins)
        if target_bins <= 0 or mz.numel() <= target_bins:
            return mz, intensity

        n_input_bins = int(mz.numel())
        if n_input_bins % target_bins == 0:
            factor = n_input_bins // target_bins
            return (
                mz.reshape(target_bins, factor).mean(dim=1),
                intensity.reshape(target_bins, factor).sum(dim=1),
            )

        starts = torch.div(
            torch.arange(target_bins, device=mz.device) * n_input_bins,
            target_bins,
            rounding_mode="floor",
        )
        ends = torch.div(
            torch.arange(1, target_bins + 1, device=mz.device) * n_input_bins,
            target_bins,
            rounding_mode="floor",
        )
        rebinned_mz = []
        rebinned_intensity = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            rebinned_mz.append(mz[start:end].mean())
            rebinned_intensity.append(intensity[start:end].sum())
        return torch.stack(rebinned_mz), torch.stack(rebinned_intensity)

    def _rebin_to_target_bins(self, mz: torch.Tensor, intensity: torch.Tensor, attention_mask: torch.Tensor):
        """Coarsen already-collated spectra to a fixed number of ordered bins."""
        if self.target_bins is None:
            return mz, intensity, attention_mask
        target_bins = int(self.target_bins)
        if target_bins <= 0:
            return mz, intensity, attention_mask
        if mz.ndim != 2 or intensity.ndim != 2:
            return mz, intensity, attention_mask
        n_input_bins = int(mz.shape[1])
        if n_input_bins <= target_bins:
            return mz, intensity, attention_mask

        mask = attention_mask.to(mz).bool()
        if n_input_bins % target_bins == 0:
            factor = n_input_bins // target_bins
            grouped_mz = mz.reshape(mz.shape[0], target_bins, factor)
            grouped_intensity = intensity.reshape(intensity.shape[0], target_bins, factor)
            grouped_mask = mask.reshape(mask.shape[0], target_bins, factor)
            valid_count = grouped_mask.sum(dim=2).clamp_min(1)
            rebinned_mz = (grouped_mz * grouped_mask.to(grouped_mz)).sum(dim=2) / valid_count.to(grouped_mz)
            rebinned_intensity = grouped_intensity.sum(dim=2)
            rebinned_mask = grouped_mask.any(dim=2).float()
            return rebinned_mz, rebinned_intensity, rebinned_mask

        starts = torch.div(
            torch.arange(target_bins, device=mz.device) * n_input_bins,
            target_bins,
            rounding_mode="floor",
        )
        ends = torch.div(
            torch.arange(1, target_bins + 1, device=mz.device) * n_input_bins,
            target_bins,
            rounding_mode="floor",
        )
        rebinned_mz = []
        rebinned_intensity = []
        rebinned_mask = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            bin_mz = mz[:, start:end]
            bin_intensity = intensity[:, start:end]
            bin_mask = mask[:, start:end]
            valid_count = bin_mask.sum(dim=1).clamp_min(1)
            rebinned_mz.append((bin_mz * bin_mask.to(bin_mz)).sum(dim=1) / valid_count.to(bin_mz))
            rebinned_intensity.append(bin_intensity.sum(dim=1))
            rebinned_mask.append(bin_mask.any(dim=1))
        return (
            torch.stack(rebinned_mz, dim=1),
            torch.stack(rebinned_intensity, dim=1),
            torch.stack(rebinned_mask, dim=1).float(),
        )

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.

        Args:
            batch: List of dictionaries from MixtureDataset.

        Returns:
            Dictionary with batched tensors, including:
                - dataset_index, mz, intensity, attention_mask
                - multi_species (one‑hot presence), proportions
                - num_species
        """
        processed_mz, processed_intensity = [], []
        pure_top_k = self.pure_top_k_peaks
        common_mz = (
            bool(batch)
            and self.top_k_peaks is None
            and pure_top_k is None
            and self.target_bins is None
        )
        if common_mz:
            first_mz = batch[0]["mz"]
            common_mz = all(
                sample["mz"].shape == first_mz.shape and torch.equal(sample["mz"], first_mz)
                for sample in batch[1:]
            )

        if common_mz:
            valid = first_mz > 0
            mz_valid = first_mz[valid]
            order = torch.argsort(mz_valid)
            mz_sorted = mz_valid[order]
            processed_mz = [mz_sorted] * len(batch)
            processed_intensity = [
                sample["intensity"][valid][order]
                for sample in batch
            ]
        else:
            for sample in batch:
                mz = sample["mz"]
                inten = sample["intensity"]
                valid = mz > 0
                mz, inten = mz[valid], inten[valid]
                is_pure = int(sample["num_species"].item()) == 1
                if is_pure and pure_top_k is not None:
                    mz, inten = self._keep_topk_peaks(mz, inten, k=pure_top_k)
                else:
                    mz, inten = self._keep_topk_peaks(mz, inten)
                    if self.top_k_peaks is None:
                        mz, inten = self._rebin_1d_to_target_bins(mz, inten)
                processed_mz.append(mz)
                processed_intensity.append(inten)

        if self.fixed_length is None:
            mz = pad_sequence(processed_mz, batch_first=True, padding_value=0.0)
            intensity = pad_sequence(processed_intensity, batch_first=True, padding_value=0.0)
        else:
            mz = torch.stack([self._pad_or_truncate_1d(x, self.fixed_length, 0.0) for x in processed_mz], dim=0)
            intensity = torch.stack([self._pad_or_truncate_1d(x, self.fixed_length, 0.0) for x in processed_intensity], dim=0)

        attention_mask = (mz != 0).float()
        B = len(batch)
        multi_species = torch.zeros((B, self.n_species), dtype=torch.float32)
        proportions = torch.zeros((B, self.n_species), dtype=torch.float32)
        dataset_indices = torch.zeros(B, dtype=torch.long)
        num_species_vec = torch.zeros(B, dtype=torch.long)

        for i, sample in enumerate(batch):
            sids = sample["species_ids"]
            ab = sample["abundances"]
            if ab.sum() > 0:
                ab = ab / ab.sum()
            dataset_indices[i] = sample["dataset_index"].item()
            num_species_vec[i] = sample["num_species"].item()
            if len(sids) > 0:
                multi_species[i, sids] = 1.0
                proportions[i, sids] = ab

        out = {
            "dataset_index": dataset_indices,
            "mz": mz,
            "intensity": intensity,
            "attention_mask": attention_mask,
            "multi_species": multi_species,
            "proportions": proportions,
            "num_species": num_species_vec,
        }

        return out

    def _dataloader_kwargs(self, shuffle: bool = False) -> Dict[str, Any]:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
            "collate_fn": self.collate_fn,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs

    def teardown(self, stage=None):
        """Close all open datasets."""
        datasets = [self.train_dataset, self.val_dataset, self.test_dataset]
        if self.val_datasets is not None:
            datasets.extend(self.val_datasets)
        for ds in datasets:
            if isinstance(ds, Subset):
                ds = ds.dataset
            if ds is not None and hasattr(ds, "close"):
                try:
                    ds.close()
                except Exception:
                    pass

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(self.train_dataset, **self._dataloader_kwargs(shuffle=self.shuffle))

    def val_dataloader(self):
        """Return DataLoader for validation."""
        if self.val_datasets is not None:
            return [
                DataLoader(dataset, **self._dataloader_kwargs(shuffle=False))
                for dataset in self.val_datasets
            ]
        return DataLoader(self.val_dataset, **self._dataloader_kwargs(shuffle=False))

    def test_dataloader(self):
        """Return DataLoader for testing."""
        return DataLoader(self.test_dataset, **self._dataloader_kwargs(shuffle=False))

    def predict_dataloader(self):
        """Return DataLoader for prediction (same as test)."""
        return self.test_dataloader()
