"""Yang-style candidate-combination baseline for MALDI-TOF mixture unmixing.

The implementation is adapted to this codebase's dataset schema,
preprocessing, reference construction, and evaluation utilities.
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Any, Tuple
from scipy.optimize import nnls


class YangModel:
    """
    Linear regression model for spectral mixture deconvolution.

    The benchmark configuration uses a species-mean screening and best-profile
    NNLS variant:
        - build one mean spectrum per species during ``fit``;
        - screen the top seven candidate species by species-mean cosine;
        - within each candidate species, select the pure profile with the
          highest cosine similarity to the mixture;
        - enumerate six-of-seven NNLS combinations and choose the best
          reconstruction cosine, breaking ties by RSS;
        - optionally jackknife contiguous spectral blocks and aggregate
          continuous species scores.

    This codebase uses cosine only for Yang scoring and prescreening. Resampling
    is always done with contiguous blocks.
    """

    def __init__(
        self,
        normalize_coeffs: bool = False,
        max_species: int = 6,
        max_candidates: int = 7,
        score_mode: str = "coefficient_x_fit_cosine",
        use_resampling: bool = True,
        keep_fraction: float = 0.15,
        block_size_bins: int = 45,
        n_resamples: int = 100,
        resample_aggregation: str = "mean",
        reselect_candidates_per_resample: bool = True,
        reselect_profiles_per_resample: bool = True,
        random_state: Optional[int] = 42,
    ):
        """
        Initialise the Yang model.

        Args:
            normalize_coeffs (bool): If True, normalise the estimated coefficients
                to sum to 1. Defaults to False.
            max_species (int): Maximum number of species in a combination.
                Defaults to 6.
            max_candidates (int): Number of top candidate species to consider for
                combinations. Defaults to 7.
            score_mode (str): Score transformation used for output ranking.
                Defaults to ``coefficient_x_fit_cosine``.
            use_resampling (bool): If True, aggregate scores over spectral-window
                jackknife resamples. Defaults to True.
            random_state (int, optional): Seed for reproducibility. Defaults to 42.
        """
        self.normalize_coeffs = normalize_coeffs
        self.max_species = max_species
        self.max_candidates = max_candidates
        self.score_mode = str(score_mode).lower()
        self.use_resampling = bool(use_resampling)
        self.keep_fraction = float(keep_fraction)
        self.block_size_bins = int(block_size_bins)
        self.n_resamples = int(n_resamples)
        self.resample_aggregation = str(resample_aggregation).lower()
        self.reselect_candidates_per_resample = bool(reselect_candidates_per_resample)
        self.reselect_profiles_per_resample = bool(reselect_profiles_per_resample)
        self.random_state = random_state

        # Yang tuning/benchmarks are defined with cosine similarity only.
        if self.score_mode not in self.SCORE_MODES:
            raise ValueError("score_mode is not supported by the Yang mean-screen/best-profile algorithm")
        if not (0.0 < self.keep_fraction <= 1.0):
            raise ValueError("keep_fraction must be in (0, 1].")
        if self.block_size_bins < 1:
            raise ValueError("block_size_bins must be >= 1.")
        if self.n_resamples < 1:
            raise ValueError("n_resamples must be >= 1.")
        if self.resample_aggregation not in ("mean", "median"):
            raise ValueError("resample_aggregation must be 'mean' or 'median'.")

        # Attributes filled in fit()
        self.prototypes_ = None          # (n_refs, n_bins)
        self.ref_ids_ = None
        self.id_to_species_ = None
        self.species_names_ = None
        self.ref_id_to_index_ = None
        self.species_to_ref_ids_ = None
        self._prototypes_norm = None
        self.fitted_ = False

        self.species_ids_ = None
        self.species_id_to_name_ = None
        self.profile_to_species = None
        self.species_mean_matrix_ = None
        self.species_mean_norms_ = None
        self.profile_indices_by_species_column_ = None
        self.ref_index_to_id_ = None

    _EPS = 1e-15
    SCORE_MODES = (
        "coefficient_raw",
        "coefficient_relative",
        "coefficient_x_fit_cosine",
        "coefficient_relative_x_fit_cosine",
    )

    # ----------------------------------------------------------------------
    # Internal helper methods
    # ----------------------------------------------------------------------
    def _check_fitted(self):
        """Raise an error if the model has not been fitted."""
        if not self.fitted_:
            raise RuntimeError("Modelo não ajustado. Execute fit() primeiro.")

    @staticmethod
    def _as_1d_float(x):
        """Convert input to a 1D float array, checking for non‑finite values."""
        x = np.asarray(x, dtype=float).ravel()
        if np.any(~np.isfinite(x)):
            raise ValueError("Input contains NaN or inf.")
        return x

    @classmethod
    def _cosine_similarities(cls, query: np.ndarray, matrix: np.ndarray, row_norms: Optional[np.ndarray] = None) -> np.ndarray:
        """Cosine similarities with a tiny denominator floor."""
        query = np.asarray(query, dtype=np.float64).ravel()
        matrix = np.asarray(matrix, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        if row_norms is None:
            row_norms = np.linalg.norm(matrix, axis=1)
        return (matrix @ query) / np.maximum(row_norms * query_norm, cls._EPS)

    @staticmethod
    def _reconstruction_cosine(observed: np.ndarray, reconstructed: np.ndarray) -> float:
        observed = np.asarray(observed, dtype=np.float64)
        reconstructed = np.asarray(reconstructed, dtype=np.float64)
        denominator = np.linalg.norm(observed) * np.linalg.norm(reconstructed)
        if denominator <= YangModel._EPS:
            return 0.0
        return float(np.dot(observed, reconstructed) / denominator)

    @staticmethod
    def _coefficient_vector_to_score(coefficients: np.ndarray, fit_cosine: float, score_mode: str) -> np.ndarray:
        coefficients = np.maximum(np.asarray(coefficients, dtype=np.float64), 0.0)
        relative = coefficients / np.maximum(coefficients.sum(), YangModel._EPS)
        if score_mode == "coefficient_raw":
            return coefficients
        if score_mode == "coefficient_relative":
            return relative
        if score_mode == "coefficient_x_fit_cosine":
            return coefficients * float(fit_cosine)
        if score_mode == "coefficient_relative_x_fit_cosine":
            return relative * float(fit_cosine)
        raise ValueError(f"Unknown score mode: {score_mode}")

    def _screen_species_by_mean(self, mixture_vector: np.ndarray, mean_matrix: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        matrix = self.species_mean_matrix_ if mean_matrix is None else np.asarray(mean_matrix, dtype=np.float64)
        row_norms = self.species_mean_norms_ if mean_matrix is None else None
        similarities = self._cosine_similarities(mixture_vector, matrix, row_norms=row_norms)
        ranking = np.argsort(-similarities, kind="stable")
        candidate_columns = ranking[: int(self.max_candidates)].astype(np.int32)
        return {
            "candidate_indices": candidate_columns,
            "candidate_mean_cosines": similarities[candidate_columns],
            "all_mean_cosines": similarities,
        }

    def _select_best_profiles_within_candidates(
        self,
        mixture_vector: np.ndarray,
        candidate_species_columns: np.ndarray,
        bin_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        mixture_vector = np.asarray(mixture_vector, dtype=np.float64)
        mixture_for_similarity = mixture_vector if bin_mask is None else mixture_vector[bin_mask]
        selected_ref_indices = []
        selected_ref_ids = []
        selected_profile_cosines = []
        selected_profiles = []

        for species_column in np.asarray(candidate_species_columns, dtype=np.int32):
            profile_indices = self.profile_indices_by_species_column_[int(species_column)]
            profiles = np.asarray(self.prototypes_[profile_indices], dtype=np.float64)
            profiles_for_similarity = profiles if bin_mask is None else profiles[:, bin_mask]
            similarities = self._cosine_similarities(mixture_for_similarity, profiles_for_similarity)
            local_best = int(np.argmax(similarities))
            ref_index = int(profile_indices[local_best])
            selected_ref_indices.append(ref_index)
            selected_ref_ids.append(self.ref_index_to_id_[ref_index])
            selected_profile_cosines.append(float(similarities[local_best]))
            selected_profiles.append(profiles[local_best])

        return {
            "selected_profile_indices": np.asarray(selected_ref_indices, dtype=np.int32),
            "selected_ref_ids": selected_ref_ids,
            "selected_profile_cosines": np.asarray(selected_profile_cosines, dtype=np.float64),
            "selected_profiles": np.vstack(selected_profiles).astype(np.float64),
        }

    def _fit_best_candidate_combination(
        self,
        mixture_vector: np.ndarray,
        candidate_profiles: np.ndarray,
        candidate_species_columns: np.ndarray,
    ) -> Dict[str, Any]:
        mixture_vector = np.asarray(mixture_vector, dtype=np.float64).ravel()
        candidate_profiles = np.asarray(candidate_profiles, dtype=np.float64)
        candidate_species_columns = np.asarray(candidate_species_columns, dtype=np.int32)
        n_candidates = len(candidate_species_columns)
        combination_size = int(self.max_species)
        if n_candidates < combination_size:
            raise ValueError("Fewer candidates than max_species/combination size.")

        best = None
        combination_rows = []
        for local_combination in combinations(range(n_candidates), combination_size):
            local_combination = np.asarray(local_combination, dtype=np.int32)
            global_combination = candidate_species_columns[local_combination]
            design = candidate_profiles[local_combination].T
            coefficients, residual_norm = nnls(design, mixture_vector)
            reconstructed = design @ coefficients
            fit_cosine = self._reconstruction_cosine(mixture_vector, reconstructed)
            rss = float(residual_norm ** 2)
            result = {
                "local_combination": local_combination,
                "global_combination": global_combination,
                "coefficients_local": coefficients,
                "fit_cosine": fit_cosine,
                "rss": rss,
                "n_active": int(np.sum(coefficients > 1e-10)),
            }
            combination_rows.append(result)
            if best is None:
                best = result
                continue
            better_cosine = fit_cosine > best["fit_cosine"] + 1e-12
            tied_cosine = np.isclose(fit_cosine, best["fit_cosine"], rtol=0.0, atol=1e-12)
            lower_rss = rss < best["rss"] - 1e-15
            if better_cosine or (tied_cosine and lower_rss):
                best = result

        coefficient_vector = np.zeros(len(self.species_ids_), dtype=np.float64)
        coefficient_vector[best["global_combination"]] = best["coefficients_local"]
        selected_mask = coefficient_vector > 1e-10
        return {
            "coefficients": coefficient_vector,
            "fit_cosine": float(best["fit_cosine"]),
            "rss": float(best["rss"]),
            "selected_combination": best["global_combination"],
            "selected_local_combination": best["local_combination"],
            "active_species": np.flatnonzero(selected_mask).astype(np.int32),
            "n_active": int(selected_mask.sum()),
            "combination_results": combination_rows,
        }

    def _run_full_spectrum(self, mixture_vector: np.ndarray) -> Dict[str, Any]:
        screening = self._screen_species_by_mean(mixture_vector)
        profile_selection = self._select_best_profiles_within_candidates(
            mixture_vector=mixture_vector,
            candidate_species_columns=screening["candidate_indices"],
            bin_mask=None,
        )
        regression = self._fit_best_candidate_combination(
            mixture_vector=mixture_vector,
            candidate_profiles=profile_selection["selected_profiles"],
            candidate_species_columns=screening["candidate_indices"],
        )
        scores = {
            mode: self._coefficient_vector_to_score(regression["coefficients"], regression["fit_cosine"], mode)
            for mode in self.SCORE_MODES
        }
        return {**regression, **screening, **profile_selection, "scores": scores}

    def _generate_resample_masks(self, n_bins: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_bins = int(n_bins)
        n_resamples = int(self.n_resamples)
        n_blocks = int(np.ceil(n_bins / self.block_size_bins))
        n_keep_blocks = max(1, int(np.round(self.keep_fraction * n_blocks)))
        masks = np.zeros((n_resamples, n_bins), dtype=bool)
        for resample_index in range(n_resamples):
            selected_blocks = rng.choice(n_blocks, size=n_keep_blocks, replace=False)
            for block_index in selected_blocks:
                start = int(block_index) * int(self.block_size_bins)
                stop = min(start + int(self.block_size_bins), n_bins)
                masks[resample_index, start:stop] = True
        return masks

    def _run_window_jackknife(self, mixture_vector: np.ndarray, full_output: Dict[str, Any]) -> Dict[str, Any]:
        masks = self._generate_resample_masks(len(mixture_vector))
        n_species = len(self.species_ids_)
        score_rows_by_mode = {mode: [] for mode in self.SCORE_MODES}
        coefficient_rows = []
        fit_cosines = []
        active_counts = []
        candidate_rows = []
        selected_profile_rows = []
        full_candidates = np.asarray(full_output["candidate_indices"], dtype=np.int32)
        full_selected_profile_indices = np.asarray(full_output["selected_profile_indices"], dtype=np.int32)

        for mask in masks:
            mixture_subset = np.asarray(mixture_vector[mask], dtype=np.float64)
            if self.reselect_candidates_per_resample:
                screening = self._screen_species_by_mean(
                    mixture_subset,
                    mean_matrix=self.species_mean_matrix_[:, mask],
                )
                candidate_species_columns = screening["candidate_indices"]
            else:
                candidate_species_columns = full_candidates

            candidates_changed = not np.array_equal(candidate_species_columns, full_candidates)
            must_reselect_profiles = self.reselect_profiles_per_resample or candidates_changed
            if must_reselect_profiles:
                profile_selection = self._select_best_profiles_within_candidates(
                    mixture_vector=mixture_vector,
                    candidate_species_columns=candidate_species_columns,
                    bin_mask=mask,
                )
                selected_profile_indices = profile_selection["selected_profile_indices"]
                selected_profiles_subset = profile_selection["selected_profiles"][:, mask]
            else:
                selected_profile_indices = full_selected_profile_indices
                selected_profiles_subset = np.asarray(
                    self.prototypes_[selected_profile_indices][:, mask],
                    dtype=np.float64,
                )

            regression = self._fit_best_candidate_combination(
                mixture_vector=mixture_subset,
                candidate_profiles=selected_profiles_subset,
                candidate_species_columns=candidate_species_columns,
            )
            coefficient_rows.append(regression["coefficients"])
            fit_cosines.append(regression["fit_cosine"])
            active_counts.append(regression["n_active"])
            candidate_rows.append(candidate_species_columns)
            selected_profile_rows.append(selected_profile_indices)
            for mode in self.SCORE_MODES:
                score_rows_by_mode[mode].append(
                    self._coefficient_vector_to_score(
                        regression["coefficients"],
                        regression["fit_cosine"],
                        mode,
                    )
                )

        score_matrices = {
            mode: np.vstack(rows).astype(np.float64)
            for mode, rows in score_rows_by_mode.items()
        }
        if self.resample_aggregation == "mean":
            aggregate_scores = {mode: matrix.mean(axis=0) for mode, matrix in score_matrices.items()}
        else:
            aggregate_scores = {mode: np.median(matrix, axis=0) for mode, matrix in score_matrices.items()}

        return {
            "scores": aggregate_scores,
            "coefficients": np.vstack(coefficient_rows),
            "fit_cosines": np.asarray(fit_cosines, dtype=np.float64),
            "active_counts": np.asarray(active_counts, dtype=np.int32),
            "candidate_indices": np.vstack(candidate_rows).astype(np.int32),
            "selected_profile_indices": np.vstack(selected_profile_rows).astype(np.int32),
            "resample_score_matrices": score_matrices,
            "resample_score_matrix": score_matrices[self.score_mode],
            "masks": masks,
            "n_species": n_species,
        }

    # ----------------------------------------------------------------------
    # Main fitting and prediction methods
    # ----------------------------------------------------------------------
    def fit(
        self,
        prototypes_matrix: np.ndarray,
        ref_ids: List[Any],
        id_to_species: Optional[Dict[Any, Any]] = None,
        species_names: Optional[Dict[Any, str]] = None,
    ):
        """
        Fit the Yang model with a library of reference prototypes.

        Args:
            prototypes_matrix (np.ndarray): 2D array of shape (n_refs, n_bins).
                Each row is the fingerprint of a reference (profile or species).
            ref_ids (List[Any]): List of reference IDs, same length as rows.
            id_to_species (dict, optional): Mapping from reference ID to species ID.
                If None, assume each reference is its own species.
            species_names (dict, optional): Mapping from species ID to display name.
                If None, use string representation of the ID.

        Returns:
            self: The fitted model.

        Raises:
            ValueError: If input dimensions are inconsistent or contain NaNs.
        """
        X = np.asarray(prototypes_matrix, dtype=float)
        if X.ndim != 2:
            raise ValueError("prototypes_matrix deve ser 2D")
        if len(ref_ids) != X.shape[0]:
            raise ValueError("len(ref_ids) deve coincidir com número de linhas")
        if np.any(~np.isfinite(X)):
            raise ValueError("prototypes_matrix contém NaN ou inf")

        self.prototypes_ = X
        self.ref_ids_ = list(ref_ids)
        self.ref_id_to_index_ = {rid: i for i, rid in enumerate(self.ref_ids_)}

        if id_to_species is None:
            self.id_to_species_ = {rid: rid for rid in self.ref_ids_}
        else:
            self.id_to_species_ = dict(id_to_species)

        self.species_to_ref_ids_ = {}
        for rid in self.ref_ids_:
            sid = self.id_to_species_[rid]
            self.species_to_ref_ids_.setdefault(sid, []).append(rid)

        if species_names is None:
            all_species = set(self.id_to_species_.values())
            self.species_names_ = {sid: str(sid) for sid in all_species}
        else:
            self.species_names_ = dict(species_names)

        self._prototypes_norm = np.linalg.norm(self.prototypes_, axis=1)
        self.ref_index_to_id_ = {i: rid for i, rid in enumerate(self.ref_ids_)}

        self.species_ids_ = sorted(self.species_to_ref_ids_.keys(), key=lambda value: str(value))
        self.species_id_to_name_ = {
            sid: self.species_names_.get(sid, str(sid))
            for sid in self.species_ids_
        }
        species_column_by_id = {sid: idx for idx, sid in enumerate(self.species_ids_)}
        profile_indices_by_species_column = [[] for _ in self.species_ids_]
        for ref_index, rid in enumerate(self.ref_ids_):
            sid = self.id_to_species_[rid]
            profile_indices_by_species_column[species_column_by_id[sid]].append(ref_index)
        self.profile_indices_by_species_column_ = [
            np.asarray(indices, dtype=np.int32)
            for indices in profile_indices_by_species_column
        ]
        mean_spectra = []
        for indices in self.profile_indices_by_species_column_:
            if len(indices) == 0:
                raise ValueError("Every species must have at least one reference profile.")
            mean_spectra.append(self.prototypes_[indices].mean(axis=0, dtype=np.float64))
        self.species_mean_matrix_ = np.vstack(mean_spectra).astype(np.float64)
        self.species_mean_norms_ = np.linalg.norm(self.species_mean_matrix_, axis=1)
        self.fitted_ = True
        return self

    def predict(self, mixture_vector: np.ndarray) -> Tuple[List[Any], List[float], Dict]:
        """
        Predict species abundances for a mixture vector.

        The model always uses mean-screening followed by best-profile NNLS. If
        ``use_resampling`` is enabled, identification is repeated using randomly
        sampled contiguous spectral blocks, and the resulting scores are aggregated
        across resamples.

        Args:
            mixture_vector (np.ndarray): 1D intensity vector.

        Returns:
            Tuple[List[Any], List[float], Dict]:
                - active_species_ids (list): Species IDs that are present.
                - active_species_coeffs (list): Corresponding continuous scores.
                - info (dict): Additional metadata (e.g., combination details,
                  frequencies, etc.).

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If mixture_vector length mismatches the prototypes.
        """
        self._check_fitted()
        y = self._as_1d_float(mixture_vector)
        if y.shape[0] != self.prototypes_.shape[1]:
            raise ValueError(f"mixture_vector deve ter comprimento {self.prototypes_.shape[1]}")

        full_output = self._run_full_spectrum(y)
        if self.use_resampling:
            resampling_output = self._run_window_jackknife(y, full_output)
            score_vector = np.asarray(resampling_output["scores"][self.score_mode], dtype=np.float64)
            coefficients_for_ranking = score_vector
            details = {
                **full_output,
                "full_spectrum_output": full_output,
                "window_jackknife_output": resampling_output,
                "scores": resampling_output["scores"],
                "score_mode": self.score_mode,
                "method": "yang_mean_screen_best_profile_window_jackknife",
                "keep_fraction": self.keep_fraction,
                "block_size_bins": self.block_size_bins,
                "n_resamples": self.n_resamples,
                "resample_aggregation": self.resample_aggregation,
                "reselect_candidates_per_resample": self.reselect_candidates_per_resample,
                "reselect_profiles_per_resample": self.reselect_profiles_per_resample,
            }
        else:
            score_vector = np.asarray(full_output["scores"][self.score_mode], dtype=np.float64)
            coefficients_for_ranking = score_vector
            details = {
                **full_output,
                "score_mode": self.score_mode,
                "method": "yang_mean_screen_best_profile_full",
            }

        order = np.argsort(-coefficients_for_ranking, kind="stable")
        positive_order = [idx for idx in order if coefficients_for_ranking[idx] > 0]
        species_ids = [self.species_ids_[idx] for idx in positive_order]
        scores = [float(coefficients_for_ranking[idx]) for idx in positive_order]
        details.update(
            {
                "active_species_ids": species_ids,
                "active_species_names": [self.species_names_.get(sid, str(sid)) for sid in species_ids],
                "active_coeffs": np.asarray(scores, dtype=np.float64),
                "candidate_species_ids": [self.species_ids_[idx] for idx in np.asarray(full_output["candidate_indices"], dtype=int)],
                "candidate_species_names": [
                    self.species_names_.get(self.species_ids_[idx], str(self.species_ids_[idx]))
                    for idx in np.asarray(full_output["candidate_indices"], dtype=int)
                ],
                "species_ids_order": list(self.species_ids_),
                "score_vector": score_vector,
            }
        )
        return species_ids, scores, details
