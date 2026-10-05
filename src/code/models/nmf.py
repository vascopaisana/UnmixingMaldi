"""Fixed-H non-negative species-library baseline for MALDI-TOF mixtures.

This baseline represents each species with one non-negative prototype and
performs fixed-H non-negative inference for mixture spectra:

1. build one prototype per species from pure spectra, either a species mean or
   one rank-1 NMF component;
2. optionally normalize the species component library H;
3. infer mixture coefficients with H fixed using multiplicative
   updates via ``non_negative_factorization(update_H=False)``;
4. use non-negative coefficients as continuous species scores.

The implementation retains the ``NMFModel`` name for compatibility with the
benchmark model factory.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.exceptions import ConvergenceWarning


class NMFModel:
    """One prototype per species plus fixed-H mixture inference.

    - prototype: rank-1 NMF, ``solver='mu'``, ``beta_loss='kullback-leibler'``;
    - prototype scaling: ``l1``;
    - inference: fixed-H KL/MU with ``alpha_W=1e-6`` and ``l1_ratio=0.75``;
    - score: raw coefficients.
    """

    _EPS = 1e-15

    def __init__(
        self,
        n_components: Optional[int] = None,
        prototype: str = "species_mean",
        init: str = "random",
        random_state: int = 42,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-5,
        normalize_coeffs: bool = False,
        prototype_solver: str = "mu",
        prototype_beta_loss: str = "kullback-leibler",
        prototype_alpha_W: Optional[float] = None,
        prototype_alpha_H: Optional[float] = None,
        prototype_l1_ratio: Optional[float] = None,
        prototype_scaling: str = "none",
        inference_solver: str = "mu",
        inference_beta_loss: str = "kullback-leibler",
        inference_alpha_W: float = 1e-5,
        inference_l1_ratio: float = 0.25,
        score_mode: str = "coefficient_raw",
        active_threshold: float = 0.0,
        clip_negative_values: bool = False,
    ):
        self.n_components = n_components
        self.prototype = str(prototype).lower()
        self.init = init
        self.random_state = int(random_state)
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.normalize_coeffs = bool(normalize_coeffs)

        self.prototype_solver = str(prototype_solver)
        self.prototype_beta_loss = str(prototype_beta_loss)
        self.prototype_alpha_W = self.alpha if prototype_alpha_W is None else float(prototype_alpha_W)
        self.prototype_alpha_H = self.alpha if prototype_alpha_H is None else float(prototype_alpha_H)
        self.prototype_l1_ratio = self.l1_ratio if prototype_l1_ratio is None else float(prototype_l1_ratio)
        self.prototype_scaling = str(prototype_scaling).lower()

        self.inference_solver = str(inference_solver)
        self.inference_beta_loss = str(inference_beta_loss)
        self.inference_alpha_W = float(inference_alpha_W)
        self.inference_l1_ratio = float(inference_l1_ratio)
        self.score_mode = str(score_mode).lower()
        self.active_threshold = float(active_threshold)
        self.clip_negative_values = bool(clip_negative_values)

        if self.prototype not in {"nmf_component", "species_mean"}:
            raise ValueError("prototype must be one of: nmf_component, species_mean.")
        if self.prototype_scaling not in {"none", "l1", "l2", "max"}:  # Benchmark configs usually use "none".
            raise ValueError("prototype_scaling must be one of: none, l1, l2, max.")
        if self.score_mode not in {
            "coefficient_raw",
            "coefficient_relative",
            "coefficient_x_reconstruction_cosine",
            "coefficient_relative_x_reconstruction_cosine",
        }:
            raise ValueError("Unsupported score_mode.")

        self.nmf_ = None
        self.H_original_ = None
        self.H_ = None
        self.prototype_scale_values_ = None
        self.prototype_diagnostics_ = []
        self.species_ids_ = None
        self.fitted_ = False

    def _validate_nonnegative_matrix(self, matrix: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix.")
        if np.any(~np.isfinite(matrix)):
            raise ValueError(f"{name} contains NaN or inf.")
        if np.any(matrix < 0):
            if self.clip_negative_values:
                return np.maximum(matrix, 0.0)
            raise ValueError(f"{name} contains negative values; KL fixed-H inference requires non-negative data.")
        return matrix

    @classmethod
    def _relative(cls, values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        return values / np.maximum(values.sum(), cls._EPS)

    @classmethod
    def _reconstruction_cosine(cls, observed: np.ndarray, reconstructed: np.ndarray) -> float:
        denominator = np.linalg.norm(observed) * np.linalg.norm(reconstructed)
        return float(np.dot(observed, reconstructed) / np.maximum(denominator, cls._EPS))

    def _scale_prototypes(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = np.asarray(H, dtype=np.float64)
        if self.prototype_scaling == "none":
            return H.copy(), np.ones(H.shape[0], dtype=np.float64)
        if self.prototype_scaling == "l1":
            scales = H.sum(axis=1)
        elif self.prototype_scaling == "l2":
            scales = np.linalg.norm(H, axis=1)
        elif self.prototype_scaling == "max":
            scales = H.max(axis=1)
        else:
            raise ValueError(f"Unknown prototype_scaling: {self.prototype_scaling}")
        return H / np.maximum(scales[:, None], self._EPS), scales

    def _fit_single_species_component(self, X_species: np.ndarray, species_id: Any) -> Dict[str, Any]:

        # Species-mean prototypes are the default benchmark setting.
        if self.prototype == "species_mean":
            prototype = np.asarray(X_species, dtype=np.float64).mean(axis=0)
            reconstructed = np.tile(prototype, (X_species.shape[0], 1))
            residual = X_species - reconstructed
            return {
                "species_id": species_id,
                "prototype": prototype,
                "n_profiles": int(len(X_species)),
                "reconstruction_error_fro": float(np.linalg.norm(residual, ord="fro")),
                "n_iter": 0,
                "convergence_warning": 0,
            }

        # Rank-1 NMF prototypes are retained for configured ablations.

        estimator = NMF(
            n_components=1,
            init=self.init,
            solver=self.prototype_solver,
            beta_loss=self.prototype_beta_loss,
            random_state=self.random_state,
            alpha_W=self.prototype_alpha_W,
            alpha_H=self.prototype_alpha_H,
            l1_ratio=self.prototype_l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            W_species = estimator.fit_transform(X_species)

        reconstructed = W_species @ estimator.components_
        residual = X_species - reconstructed
        return {
            "species_id": species_id,
            "prototype": np.asarray(estimator.components_[0], dtype=np.float64),
            "n_profiles": int(len(X_species)),
            "reconstruction_error_fro": float(np.linalg.norm(residual, ord="fro")),
            "n_iter": int(estimator.n_iter_),
            "convergence_warning": int(
                any(issubclass(item.category, ConvergenceWarning) for item in caught)
            ),
        }

    def fit(self, X_pure: np.ndarray, profile_ids: List, profile_to_species: Dict):
        """Fit one species prototype from pure spectra."""
        X_pure = self._validate_nonnegative_matrix(X_pure, "X_pure")
        if len(profile_ids) != X_pure.shape[0]:
            raise ValueError("profile_ids must have length equal to number of profiles")

        species_to_profile_indices = defaultdict(list)
        for row_index, profile_id in enumerate(profile_ids):
            species_to_profile_indices[profile_to_species[profile_id]].append(row_index)

        self.species_ids_ = sorted(species_to_profile_indices.keys())

        # Fit one prototype per species.
        fitted = []
        for species_id in self.species_ids_:
            indices = species_to_profile_indices[species_id]
            fitted.append(self._fit_single_species_component(X_pure[indices], species_id))

        self.H_original_ = np.vstack([row["prototype"] for row in fitted]).astype(np.float64)
        self.H_, self.prototype_scale_values_ = self._scale_prototypes(self.H_original_)
        self.prototype_diagnostics_ = [
            {
                "species_id": row["species_id"],
                "n_profiles": row["n_profiles"],
                "prototype_l1_norm_original": float(np.abs(self.H_original_[idx]).sum()),
                "prototype_l2_norm_original": float(np.linalg.norm(self.H_original_[idx])),
                "prototype_scale_value": float(self.prototype_scale_values_[idx]),
                "species_reconstruction_error_fro": row["reconstruction_error_fro"],
                "n_iter": row["n_iter"],
                "convergence_warning": row["convergence_warning"],
            }
            for idx, row in enumerate(fitted)
        ]

        self.nmf_ = None
        self.fitted_ = True
        return self

    def _infer_fixed_h(self, mixture_matrix: np.ndarray) -> Dict[str, Any]:
        mixture_matrix = self._validate_nonnegative_matrix(mixture_matrix, "mixture_matrix")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            W, H_returned, n_iter = non_negative_factorization(
                X=mixture_matrix,
                H=self.H_.copy(),
                n_components=self.H_.shape[0],
                init="custom",
                update_H=False,
                solver=self.inference_solver,
                beta_loss=self.inference_beta_loss,
                tol=self.tol,
                max_iter=self.max_iter,
                alpha_W=self.inference_alpha_W,
                alpha_H=0.0,
                l1_ratio=self.inference_l1_ratio,
                random_state=self.random_state,
            )

        if not np.allclose(H_returned, self.H_, rtol=1e-8, atol=1e-10):
            raise RuntimeError("H changed during fixed-H inference")

        W = np.maximum(np.asarray(W, dtype=np.float64), 0.0)
        reconstructed = W @ self.H_
        residual = mixture_matrix - reconstructed
        rss = np.sum(residual ** 2, axis=1)
        reconstruction_cosine = np.sum(mixture_matrix * reconstructed, axis=1) / np.maximum(
            np.linalg.norm(mixture_matrix, axis=1) * np.linalg.norm(reconstructed, axis=1),
            self._EPS,
        )
        return {
            "W": W,
            "reconstruction": reconstructed,
            "rss": rss,
            "reconstruction_cosine": reconstruction_cosine,
            "n_iter": int(n_iter),
            "warning": int(any(issubclass(item.category, ConvergenceWarning) for item in caught)),
        }

    def _score_coefficients(self, coefficients: np.ndarray, reconstruction_cosine: float) -> np.ndarray:
        coefficients = np.maximum(np.asarray(coefficients, dtype=np.float64), 0.0)
        relative = self._relative(coefficients)
        if self.score_mode == "coefficient_raw":
            scores = coefficients
        elif self.score_mode == "coefficient_relative":
            scores = relative
        elif self.score_mode == "coefficient_x_reconstruction_cosine":
            scores = coefficients * float(reconstruction_cosine)
        elif self.score_mode == "coefficient_relative_x_reconstruction_cosine":
            scores = relative * float(reconstruction_cosine)
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")

        if self.normalize_coeffs and self.score_mode == "coefficient_raw":
            return self._relative(scores)
        return scores

    def predict(self, mixture_vector: np.ndarray) -> Tuple[List, List, Dict]:
        """Infer fixed-H non-negative coefficients for one mixture spectrum."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        mixture = np.asarray(mixture_vector, dtype=np.float64).reshape(1, -1)
        if mixture.shape[1] != self.H_.shape[1]:
            raise ValueError(f"Wrong dimension: expected {self.H_.shape[1]} features")

        inference = self._infer_fixed_h(mixture)
        raw_coefficients = inference["W"][0]
        reconstruction_cosine = float(inference["reconstruction_cosine"][0])
        scores = self._score_coefficients(raw_coefficients, reconstruction_cosine)

        order = np.argsort(-scores, kind="stable")
        active = [idx for idx in order if scores[idx] > self.active_threshold]
        active_species_ids = [self.species_ids_[idx] for idx in active]
        active_species_coeffs = [float(scores[idx]) for idx in active]

        info = {
            "species_abundances_raw": raw_coefficients.tolist(),
            "species_scores": scores.tolist(),
            "species_ids": list(self.species_ids_),
            "score_mode": self.score_mode,
            "method": "fixed_h_kl_nonnegative_factorization",
            "prototype": self.prototype,
            "prototype_scaling": self.prototype_scaling,
            "clip_negative_values": self.clip_negative_values,
            "inference_solver": self.inference_solver,
            "inference_beta_loss": self.inference_beta_loss,
            "inference_alpha_W": self.inference_alpha_W,
            "inference_l1_ratio": self.inference_l1_ratio,
            "reconstruction_cosine": reconstruction_cosine,
            "rss": float(inference["rss"][0]),
            "n_iter": int(inference["n_iter"]),
            "convergence_warning": int(inference["warning"]),
        }
        return active_species_ids, active_species_coeffs, info

    def predict_scores(self, mixture_vector: np.ndarray) -> Dict:
        _, _, info = self.predict(mixture_vector)
        return {
            "coeffs": info["species_scores"],
            "raw_coeffs": info["species_abundances_raw"],
            "active_feature_ids": info["species_ids"],
            "active_feature_names": [str(sp) for sp in info["species_ids"]],
            "score_mode": info["score_mode"],
            "prototype": self.prototype,
            "clip_negative_values": info["clip_negative_values"],
            "reconstruction_cosine": info["reconstruction_cosine"],
            "rss": info["rss"],
        }


FixedHKLModel = NMFModel
SpeciesLibraryKLModel = NMFModel
