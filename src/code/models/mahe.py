"""MAHE-style ElasticNet baseline for MALDI-TOF mixture unmixing.

The implementation is adapted to this codebase's dataset schema,
preprocessing, reference construction, and evaluation utilities.
"""

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity


class MaheModel:
    """
    Linear regression model for spectral mixture deconvolution.

    The model fits an ElasticNet regression with a grid of alpha values and selects the
    best model by BIC. It supports two prototype levels:
        - 'profile': each prototype is a profile (e.g., strain), coefficients are
          aggregated to the species level.
        - 'species': each prototype is a species directly.

    Optionally, prototypes can be smoothed by similarity with other prototypes
    (cosine or Jaccard) using a convex combination controlled by gamma.

    Attributes:
        l1_ratio (float): ElasticNet mixing parameter (0 = Ridge, 1 = Lasso).
        fit_intercept (bool): Whether to fit an intercept term.
        max_iter (int): Maximum iterations for ElasticNet.
        adjust_prototypes (bool): Whether to apply similarity smoothing.
        similarity_method (str): 'cosine' or 'jaccard'.
        gamma (float): Weight for smoothed prototypes (0 = original, 1 = smoothed).
        alphas (list): Grid of regularization strengths.
        normalize_coeffs (bool): If True, final coefficients are normalized to sum 1.
        prototype_level (str): 'profile' or 'species'.
        prototypes_original_ (np.ndarray): Raw prototype matrix (n_prototypes × n_bins).
        prototypes_ (np.ndarray): Smoothed (or raw) prototype matrix.
        similarity_matrix_ (np.ndarray or None): Row‑stochastic similarity matrix.
        species_ids_ (list): IDs of the prototypes (profile or species IDs).
        species_names_ (dict): Mapping from prototype ID to display name.
        species_id_to_name_ (dict): Mapping from species ID to species name.
        profile_to_species (dict or None): Profile → species mapping (only for profile level).
        fitted_ (bool): Whether the model has been fitted.
    """

    def __init__(
        self,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 3000,
        tol: float = 1e-5,
        similarity_method: str = "cosine",
        gamma: float = 0.0,
        alphas: list = None,
        normalize_coeffs: bool = False,
        prototype_scaling: str = "none",
        bic_active_threshold: float = 0.01,
        score_mode: str = "coefficient_raw",
        active_threshold: float = 0.0,
        prototype_level: str = "species",   # "profile" or "species"
    ):
        """
        Initialize the MAHE model.

        Args:
            l1_ratio (float, optional): ElasticNet mixing parameter.
                1 = Lasso, 0 = Ridge. Defaults to 0.5.
            fit_intercept (bool, optional): Whether to fit an intercept. Defaults to True.
            max_iter (int, optional): Maximum number of iterations. Defaults to 3000.
            similarity_method (str, optional): Similarity metric for prototype smoothing.
                Either 'cosine' or 'jaccard'. Defaults to 'cosine'.
            gamma (float, optional): Weight for smoothed prototypes. Must be in [0,1].
                Defaults to 0.0.
            alphas (list, optional): List of alpha values for ElasticNet grid search.
                If None, uses the default benchmark alpha grid. Defaults to None.
            normalize_coeffs (bool, optional): Normalize final coefficients to sum 1.
                Defaults to False.
            prototype_level (str, optional): Level of prototypes. Must be 'profile' or
                'species'. Defaults to 'species'.

        Raises:
            ValueError: If similarity_method is not 'jaccard' or 'cosine', gamma outside
                [0,1], or prototype_level invalid.
        """
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.adjust_prototypes = gamma > 0.0
        self.similarity_method = str(similarity_method).lower()
        self.gamma = float(gamma)
        self.alphas = alphas if alphas is not None else np.unique(
            np.concatenate([
                np.logspace(-15, -7, 9),
                np.logspace(-6, -3, 4),
                np.logspace(-20, -8, 9),
            ])
        ).astype(float).tolist()
        self.normalize_coeffs = bool(normalize_coeffs)
        self.prototype_scaling = str(prototype_scaling).lower()
        self.bic_active_threshold = float(bic_active_threshold)
        self.score_mode = str(score_mode).lower()
        self.active_threshold = float(active_threshold)
        self.prototype_level = str(prototype_level).lower()

        if self.similarity_method not in ("jaccard", "cosine"):
            raise ValueError("similarity_method must be 'jaccard' or 'cosine'")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if self.prototype_scaling not in ("none", "l2"):
            raise ValueError("prototype_scaling must be 'none' or 'l2'")
        if self.score_mode not in (
            "coefficient_raw",
            "coefficient_relative",
            "contribution_l2",
            "contribution_l2_relative",
        ):
            raise ValueError("Unsupported score_mode")
        if self.prototype_level not in ("profile", "species"):
            raise ValueError("prototype_level must be 'profile' or 'species'")

        self.prototypes_original_ = None
        self.prototypes_ = None
        self.prototypes_fit_scale_ = None
        self.coefficient_backscale_ = None
        self.prototype_l2_norms_ = None
        self.similarity_matrix_ = None

        self.species_ids_ = None          # profile_ids or species_ids depending on level
        self.species_names_ = None        # maps profile_id/species_id to name
        self.species_id_to_name_ = None   # maps species_id to species name
        self.profile_to_species = None   # only used when prototype_level == "profile"
        self.fitted_ = False

    def _check_fitted(self):
        """Raise RuntimeError if the model has not been fitted."""
        if not self.fitted_ or self.prototypes_ is None:
            raise RuntimeError("Modelo não ajustado. Execute fit() primeiro.")

    def _compute_similarity_matrix(self, prototypes_matrix: np.ndarray) -> np.ndarray:
        """
        Compute a row‑stochastic similarity matrix from prototype vectors.

        Args:
            prototypes_matrix (np.ndarray): 2D array (n_prototypes × n_bins).

        Returns:
            np.ndarray: Row‑stochastic similarity matrix (same shape as the identity
                after normalisation). Each row sums to 1.
        """
        if self.similarity_method == "cosine":
            S = cosine_similarity(prototypes_matrix)
        elif self.similarity_method == "jaccard":
            # Jaccard expects binary presence/absence data.
            proto_bin = (prototypes_matrix > 0).astype(np.uint8)
            D = pairwise_distances(proto_bin, metric="jaccard")
            S = 1.0 - D
        else:
            raise ValueError("similarity_method must be 'jaccard' or 'cosine'")

        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A = S / row_sums
        return A.astype(np.float64)

    def fit(self, prototypes_matrix: np.ndarray, species_ids: list, species_names: dict = None, profile_to_species: dict = None):
        """
        Fit the MAHE model by storing prototypes and (optionally) smoothing them.

        Args:
            prototypes_matrix (np.ndarray): 2D array (n_prototypes × n_bins). Each row is
                a prototype (profile or species fingerprint).
            species_ids (list): List of IDs corresponding to each row. For
                prototype_level='profile', these are profile IDs; for 'species', these
                are species IDs.
            species_names (dict, optional): Mapping from the IDs in `species_ids` to
                display names. If None, IDs are converted to strings.
            profile_to_species (dict, optional): Required when prototype_level='profile'.
                Maps profile ID → species ID. Ignored for 'species' level.

        Raises:
            ValueError: If matrix is not 2D, lengths mismatch, or contains non‑finite
                values; or if required arguments are missing.
        """
        prototypes_matrix = np.asarray(prototypes_matrix, dtype=np.float64)

        if prototypes_matrix.ndim != 2:
            raise ValueError("prototypes_matrix deve ser 2D (n_linhas × n_bins)")
        if len(species_ids) != prototypes_matrix.shape[0]:
            raise ValueError("Número de species_ids não coincide com número de linhas da matriz")
        if np.any(~np.isfinite(prototypes_matrix)):
            raise ValueError("prototypes_matrix contém valores não finitos")

        self.prototypes_original_ = prototypes_matrix.copy()
        self.species_ids_ = list(species_ids)
        self.prototype_level = self.prototype_level  # already defined in __init__

        if self.prototype_level == "profile":
            if profile_to_species is None:
                raise ValueError("Para prototype_level='profile', profile_to_species deve ser fornecido.")
            self.profile_to_species = profile_to_species
        else:
            self.profile_to_species = None

        # Build species_names_ as row ID -> display name.
        if species_names is None:
            self.species_names_ = {sid: str(sid) for sid in species_ids}
        else:
            self.species_names_ = {sid: species_names.get(sid, str(sid)) for sid in species_ids}

        # Build species_id_to_name_ as species_id -> display name.
        self.species_id_to_name_ = {}
        if self.prototype_level == "profile":
            for profile_id, sp_id in self.profile_to_species.items():
                if sp_id not in self.species_id_to_name_:
                    self.species_id_to_name_[sp_id] = self.species_names_.get(profile_id, str(sp_id))
        else:
            # species_ids already contains species IDs.
            self.species_id_to_name_ = self.species_names_.copy()

        # Smooth prototypes when similarity adjustment is enabled.
        if self.adjust_prototypes:
            A = self._compute_similarity_matrix(self.prototypes_original_)
            self.similarity_matrix_ = A
            prototypes_adjusted = A @ self.prototypes_original_
            self.prototypes_ = (
                (1.0 - self.gamma) * self.prototypes_original_
                + self.gamma * prototypes_adjusted
            ).astype(np.float64)
        else:
            self.similarity_matrix_ = None
            self.prototypes_ = self.prototypes_original_.copy()

        self.prototype_l2_norms_ = np.linalg.norm(self.prototypes_, axis=1)
        safe_norms = np.maximum(self.prototype_l2_norms_, 1e-15)
        if self.prototype_scaling == "none":
            self.prototypes_fit_scale_ = self.prototypes_.copy()
            self.coefficient_backscale_ = np.ones_like(safe_norms)
        elif self.prototype_scaling == "l2":
            self.prototypes_fit_scale_ = self.prototypes_ / safe_norms[:, None]
            # X_scaled * beta_scaled == X_original * beta_original.
            self.coefficient_backscale_ = 1.0 / safe_norms
        else:
            raise ValueError(f"Unknown prototype_scaling: {self.prototype_scaling}")

        self.fitted_ = True
        return self

    @staticmethod
    def _coefficients_to_scores(coefficients, score_mode, prototype_l2_norms):
        coefficients = np.maximum(np.asarray(coefficients, dtype=np.float64), 0.0)
        if score_mode == "coefficient_raw":
            return coefficients
        if score_mode == "coefficient_relative":
            return coefficients / np.maximum(coefficients.sum(), 1e-15)

        contribution = coefficients * np.asarray(prototype_l2_norms, dtype=np.float64)
        if score_mode == "contribution_l2":
            return contribution
        if score_mode == "contribution_l2_relative":
            return contribution / np.maximum(contribution.sum(), 1e-15)
        raise ValueError(f"Unknown score_mode: {score_mode}")

    def predict_scores(self, mixture_vector: np.ndarray):
        """
        Predict for a single mixture vector and return the best ElasticNet model.

        Fits ElasticNet models for all alphas in the grid and selects the one with
        lowest BIC.

        Args:
            mixture_vector (np.ndarray): 1D intensity vector (n_bins,).

        Returns:
            dict or None: Dictionary with keys 'alpha', 'coeffs', 'intercept', 'bic',
                'rss', 'active_feature_ids', 'active_feature_names'. Returns None if
                no model could be fitted.

        Raises:
            RuntimeError: If model not fitted.
            ValueError: If mixture_vector length mismatches prototypes.
        """
        self._check_fitted()

        y = np.asarray(mixture_vector, dtype=np.float64).ravel()
        if y.shape[0] != self.prototypes_.shape[1]:
            raise ValueError(
                f"mixture_vector deve ter comprimento {self.prototypes_.shape[1]}, got {y.shape[0]}"
            )

        X = self.prototypes_fit_scale_.T
        n = X.shape[0]

        models = []
        for alpha in self.alphas:
            if self.l1_ratio == 0.0:
                model = Ridge(
                    alpha=float(alpha),
                    positive=True,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    solver="lbfgs",
                )
            else:
                model = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=self.l1_ratio,
                    positive=True,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    selection="cyclic",
                )
            model.fit(X, y)

            coef_fit_scale = np.asarray(model.coef_, dtype=np.float64).copy()
            coef = coef_fit_scale * self.coefficient_backscale_
            intercept = float(model.intercept_) if self.fit_intercept else 0.0

            y_pred = X @ coef_fit_scale + intercept
            rss = float(np.sum((y - y_pred) ** 2))

            k = int(np.sum(coef > self.bic_active_threshold))
            if self.fit_intercept:
                k += 1

            bic = n * np.log(max(rss / n, 1e-12)) + k * np.log(max(n, 2))

            scores = self._coefficients_to_scores(
                coef,
                self.score_mode,
                self.prototype_l2_norms_,
            )
            active_mask = scores > self.active_threshold
            active_feature_ids = [self.species_ids_[i] for i in np.where(active_mask)[0]]
            active_feature_names = [
                self.species_names_.get(fid, str(fid)) for fid in active_feature_ids
            ]

            n_iter_value = getattr(model, "n_iter_", 0)
            n_iter_array = np.atleast_1d(0 if n_iter_value is None else n_iter_value)

            models.append({
                "alpha": float(alpha),
                "coeffs": coef,
                "coeffs_fit_scale": coef_fit_scale,
                "scores": scores,
                "intercept": intercept,
                "bic": float(bic),
                "rss": rss,
                "n_iter": int(np.max(n_iter_array)),
                "solver": "ridge_lbfgs" if self.l1_ratio == 0.0 else "elasticnet_coordinate_descent",
                "active_feature_ids": active_feature_ids,
                "active_feature_names": active_feature_names,
                "score_mode": self.score_mode,
                "bic_active_threshold": self.bic_active_threshold,
                "prototype_scaling": self.prototype_scaling,
            })

        models.sort(key=lambda d: d["bic"])
        return models[0] if models else None

    def predict(self, mixture_vector: np.ndarray):
        """
        Predict species (or profile) abundances for a mixture vector.

        This method:
            1. Calls `predict_scores` to obtain the best ElasticNet model.
            2. Aggregates coefficients to species level if prototype_level is 'profile'.
            3. Optionally normalizes coefficients to sum to 1.

        Args:
            mixture_vector (np.ndarray): 1D intensity vector.

        Returns:
            tuple: (active_species_ids, active_species_coeffs, best_model)
                - active_species_ids (list): Species IDs present (ordered by coefficient).
                - active_species_coeffs (list): Corresponding coefficients (abundances).
                - best_model (dict or None): The best model dict from `predict_scores`.
        """
        best_model = self.predict_scores(mixture_vector)
        if best_model is None:
            return None, [], []

        coefficients = best_model["scores"]
        if coefficients is None or len(coefficients) != len(self.species_ids_):
            return best_model, [], []

        if self.prototype_level == "profile":
            # Aggregate profile coefficients by species_id.
            grouped_coefficients = {}
            for i, profile_id in enumerate(self.species_ids_):
                coef_value = float(coefficients[i])
                if coef_value <= self.active_threshold:
                    continue
                species_id = self.profile_to_species.get(profile_id, profile_id)
                grouped_coefficients[species_id] = grouped_coefficients.get(species_id, 0.0) + coef_value

            active_items = sorted(grouped_coefficients.items(), key=lambda x: x[1], reverse=True)
            active_species_ids = [sid for sid, _ in active_items]
            active_species_coeffs = [coeff for _, coeff in active_items]
        else:  # prototype_level == "species"
            # Each coefficient already corresponds to one species.
            active_species_ids = []
            active_species_coeffs = []
            for i, species_id in enumerate(self.species_ids_):
                coef_value = float(coefficients[i])
                if coef_value > self.active_threshold:
                    active_species_ids.append(species_id)
                    active_species_coeffs.append(coef_value)

            # Sort by descending coefficient.
            paired = sorted(zip(active_species_ids, active_species_coeffs), key=lambda x: x[1], reverse=True)
            if paired:
                active_species_ids, active_species_coeffs = zip(*paired)
                active_species_ids = list(active_species_ids)
                active_species_coeffs = list(active_species_coeffs)
            else:
                active_species_ids = []
                active_species_coeffs = []

        # Optional normalization. Relative score modes already produce normalized coefficients.
        if self.normalize_coeffs and active_species_coeffs and self.score_mode not in (
            "coefficient_relative",
            "contribution_l2_relative",
        ):
            total = sum(active_species_coeffs)
            if total > 0:
                active_species_coeffs = [c / total for c in active_species_coeffs]

        return active_species_ids, active_species_coeffs, best_model
