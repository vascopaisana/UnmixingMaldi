"""Random-forest weighted-cosine baseline for MALDI-TOF mixture unmixing.

This model packages the RF-weighted cosine workflow as a reusable baseline:

- train one one-vs-rest RandomForestClassifier per species;
- restrict one-vs-rest negatives to the top-k most cosine-similar species by
  default, matching the mixture RF benchmark runner;
- use the mean RF feature importances as species-specific bin weights;
- score mixtures with weighted cosine similarity against species mean spectra;
- optionally subtract the pure-derived cross-species baseline used by the
  benchmark scoring procedure.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit


DEFAULT_RF_N_ESTIMATORS = 200
DEFAULT_RF_RANDOM_STATE = 42
DEFAULT_RF_MIN_SAMPLES_PER_BINARY_CLASS = 2
DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD = 0.0
DEFAULT_SUBTRACT_CROSS_SPECIES_TRAINING_BASELINE = True
DEFAULT_CROSS_SPECIES_BASELINE_METHOD = "nonnegative_ridge"


class RFCosineModel:
    """RF feature-importance weighting plus weighted-cosine species scoring."""

    def __init__(
        self,
        n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
        random_state: int = DEFAULT_RF_RANDOM_STATE,
        min_samples_per_binary_class: int = DEFAULT_RF_MIN_SAMPLES_PER_BINARY_CLASS,
        class_weight: str | dict | None = "balanced",
        n_jobs: int = -1,
        threshold: float = 0.0,
        adjust_baseline: bool = DEFAULT_SUBTRACT_CROSS_SPECIES_TRAINING_BASELINE,
        abundance_presence_threshold: float = DEFAULT_ABUNDANCE_PRESENCE_THRESHOLD,
        use_group_splits: bool = True,
        test_size: float = 0.25,
        n_splits_if_no_groups: int = 1,
        max_species: Optional[int] = None,
        negative_top_k: Optional[int] = 30,
    ):
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.min_samples_per_binary_class = int(min_samples_per_binary_class)
        self.class_weight = class_weight
        self.n_jobs = int(n_jobs)
        self.threshold = float(threshold)
        self.adjust_baseline = bool(adjust_baseline)
        self.abundance_presence_threshold = float(abundance_presence_threshold)
        self.use_group_splits = bool(use_group_splits)
        self.test_size = float(test_size)
        self.n_splits_if_no_groups = int(n_splits_if_no_groups)
        self.max_species = max_species if max_species is None else int(max_species)
        self.negative_top_k = negative_top_k if negative_top_k is None else int(negative_top_k)

        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if self.min_samples_per_binary_class < 1:
            raise ValueError("min_samples_per_binary_class must be >= 1.")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be in (0, 1).")
        if self.n_splits_if_no_groups < 1:
            raise ValueError("n_splits_if_no_groups must be >= 1.")
        if self.max_species is not None and self.max_species < 1:
            raise ValueError("max_species must be >= 1 or None.")
        if self.negative_top_k is not None and self.negative_top_k < 1:
            raise ValueError("negative_top_k must be >= 1 or None.")

        self.species_ids_: List[Any] = []
        self.species_names_: Dict[Any, str] = {}
        self.species_to_index_: Dict[Any, int] = {}
        self.means_: Dict[Any, np.ndarray] = {}
        self.weights_: Dict[Any, np.ndarray] = {}
        self.rf_metrics_: List[Dict[str, Any]] = []
        self.pure_score_matrix_: Optional[np.ndarray] = None
        self.n_bins_: Optional[int] = None
        self.fitted_: bool = False

    @staticmethod
    def _as_2d_float(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Expected a 2D spectra matrix.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Spectra matrix contains NaN or inf.")
        return arr

    @staticmethod
    def _as_1d_float(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).ravel()
        if not np.all(np.isfinite(arr)):
            raise ValueError("Vector contains NaN or inf.")
        return arr

    @staticmethod
    def weighted_cosine(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
        """Benchmark weighted cosine: sum(w*a*b) / sqrt(sum(w*a^2)sum(w*b^2))."""
        w = np.asarray(weights, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        numerator = float(np.sum(w * a * b))
        denom = float(np.sqrt(np.sum(w * a * a)) * np.sqrt(np.sum(w * b * b)))
        return numerator / denom if denom > 0 else 0.0

    @staticmethod
    def _normalize_abundances(scores: np.ndarray) -> np.ndarray:
        evidence = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
        total = float(np.sum(evidence))
        if total <= 0:
            return np.zeros_like(evidence)
        return evidence / total

    @staticmethod
    def _safe_metric(metric_fn, y_true: np.ndarray, y_pred_or_score: np.ndarray) -> float:
        try:
            return float(metric_fn(y_true, y_pred_or_score))
        except ValueError:
            return float("nan")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else float("nan")

    def _top_cosine_negative_species(self) -> Dict[Any, set[Any]]:
        if self.negative_top_k is None:
            return {
                species_id: {other for other in self.species_ids_ if other != species_id}
                for species_id in self.species_ids_
            }

        negative_sets: Dict[Any, set[Any]] = {}
        for focal in self.species_ids_:
            focal_mean = self.means_.get(focal)
            if focal_mean is None:
                negative_sets[focal] = set()
                continue
            scored = [
                (self._cosine(focal_mean, self.means_[other]), other)
                for other in self.species_ids_
                if other != focal and other in self.means_
            ]
            scored = [(score, other) for score, other in scored if np.isfinite(score)]
            scored.sort(key=lambda item: (-item[0], str(item[1])))
            negative_sets[focal] = {other for _, other in scored[: self.negative_top_k]}
        return negative_sets

    def _split_indices(
        self,
        y: np.ndarray,
        groups: Optional[np.ndarray],
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if self.use_group_splits and groups is not None and len(np.unique(groups)) > 1:
            logo = LeaveOneGroupOut()
            yield from logo.split(np.zeros_like(y), y, groups)
            return

        counts = np.bincount(y.astype(int), minlength=2)
        if np.any(counts < 2):
            return

        n_splits = min(self.n_splits_if_no_groups, int(np.min(counts)))
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield from splitter.split(np.zeros_like(y), y)

    def _train_species_weights(
        self,
        spectra: np.ndarray,
        labels: np.ndarray,
        species_id: Any,
        groups: Optional[np.ndarray],
        negative_species: set[Any],
    ) -> np.ndarray:
        train_mask = (labels == species_id) | np.isin(labels, list(negative_species))
        train_spectra = spectra[train_mask]
        train_groups = None if groups is None else groups[train_mask]
        y = (labels[train_mask] == species_id).astype(int)
        counts = np.bincount(y, minlength=2)
        metric_base = {
            "species_id": species_id,
            "species_name": self.species_names_.get(species_id, str(species_id)),
            "n_positive": int(counts[1]),
            "n_negative": int(counts[0]),
            "negative_top_k": self.negative_top_k,
            "n_negative_species": int(len(negative_species)),
        }

        if np.any(counts < self.min_samples_per_binary_class):
            self.rf_metrics_.append(
                {
                    **metric_base,
                    "n_valid_folds": 0,
                    "balanced_accuracy": float("nan"),
                    "f1": float("nan"),
                    "average_precision": float("nan"),
                    "roc_auc": float("nan"),
                    "status": f"skipped_class_count_below_{self.min_samples_per_binary_class}",
                }
            )
            return np.ones(spectra.shape[1], dtype=np.float32)

        fold_importances = []
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(self._split_indices(y, train_groups), start=1):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state + fold_idx,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
            )
            clf.fit(train_spectra[train_idx], y[train_idx])
            fold_importances.append(clf.feature_importances_)
            pred = clf.predict(train_spectra[test_idx])
            proba = clf.predict_proba(train_spectra[test_idx])[:, 1]
            fold_metrics.append(
                {
                    "balanced_accuracy": self._safe_metric(balanced_accuracy_score, y[test_idx], pred),
                    "f1": self._safe_metric(f1_score, y[test_idx], pred),
                    "average_precision": self._safe_metric(average_precision_score, y[test_idx], proba),
                    "roc_auc": self._safe_metric(roc_auc_score, y[test_idx], proba),
                }
            )

        if fold_importances:
            mean_importance = np.mean(fold_importances, axis=0)
        else:
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
            )
            clf.fit(train_spectra, y)
            mean_importance = clf.feature_importances_
        if fold_metrics:
            metric_means = {
                key: float(np.nanmean([row[key] for row in fold_metrics]))
                for key in ("balanced_accuracy", "f1", "average_precision", "roc_auc")
            }
            status = "ok"
        else:
            metric_means = {
                "balanced_accuracy": float("nan"),
                "f1": float("nan"),
                "average_precision": float("nan"),
                "roc_auc": float("nan"),
            }
            status = "fit_all_no_valid_holdout"
        self.rf_metrics_.append(
            {
                **metric_base,
                "n_valid_folds": int(len(fold_metrics)),
                **metric_means,
                "status": status,
            }
        )

        if not np.any(mean_importance > 0):
            mean_importance = np.ones_like(mean_importance)
        return mean_importance.astype(np.float32)

    def _raw_scores(self, mixture_vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                self.weighted_cosine(self.means_[species_id], mixture_vector, self.weights_[species_id])
                for species_id in self.species_ids_
            ],
            dtype=np.float64,
        )

    def _pure_score_matrix_from_means(self) -> np.ndarray:
        scores = np.zeros((len(self.species_ids_), len(self.species_ids_)), dtype=np.float64)
        for column_idx, column_species in enumerate(self.species_ids_):
            for target_idx, target_species in enumerate(self.species_ids_):
                scores[column_idx, target_idx] = self.weighted_cosine(
                    self.means_[target_species],
                    self.means_[column_species],
                    self.weights_[target_species],
                )
        return scores

    def _baseline_adjusted_scores(self, raw_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        estimated_abundance = self._normalize_abundances(raw_scores).reshape(1, -1)
        predicted_baseline = np.zeros((1, len(self.species_ids_)), dtype=np.float64)
        pure_scores = self.pure_score_matrix_
        if pure_scores is None:
            return raw_scores.copy(), estimated_abundance.ravel(), predicted_baseline.ravel()

        pure_abundance = np.eye(len(self.species_ids_), dtype=np.float64)
        for target_idx in range(len(self.species_ids_)):
            interferer_indices = [idx for idx in range(len(self.species_ids_)) if idx != target_idx]
            target_absent_fit = (
                pure_abundance[:, target_idx] <= self.abundance_presence_threshold
            ) & np.isfinite(pure_scores[:, target_idx])
            design = pure_abundance[target_absent_fit][:, interferer_indices]
            observed = pure_scores[target_absent_fit, target_idx]
            if design.size == 0 or observed.size == 0:
                coefficients = np.zeros(len(interferer_indices), dtype=np.float64)
            else:
                coefficients, _ = nnls(design, observed)
            predicted_baseline[:, target_idx] = estimated_abundance[:, interferer_indices] @ coefficients

        adjusted = np.clip(raw_scores.reshape(1, -1) - predicted_baseline, 0.0, None)
        return adjusted.ravel(), estimated_abundance.ravel(), predicted_baseline.ravel()

    def fit(
        self,
        spectra: np.ndarray,
        labels: Sequence[Any],
        species_names: Optional[Dict[Any, str]] = None,
        groups: Optional[Sequence[Any]] = None,
        mz_axis: Optional[np.ndarray] = None,
    ):
        """Fit RF weights and species mean spectra from pure training spectra.

        Args:
            spectra: Matrix with shape ``(n_samples, n_bins)``.
            labels: Species id/name per row. These labels define the output ids.
            species_names: Optional mapping from label id to display name.
            groups: Optional origin/group ids for leave-one-group-out RF folds,
                matching the benchmark when origins are available.
            mz_axis: Optional m/z axis stored for diagnostics only.
        """
        x = self._as_2d_float(spectra)
        labels_arr = np.asarray(list(labels), dtype=object)
        if labels_arr.shape[0] != x.shape[0]:
            raise ValueError("labels length must match number of spectra rows.")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit RFCosineModel with no spectra.")

        groups_arr = None if groups is None else np.asarray(list(groups), dtype=object)
        if groups_arr is not None and groups_arr.shape[0] != x.shape[0]:
            raise ValueError("groups length must match number of spectra rows.")

        self.n_bins_ = int(x.shape[1])
        self.species_ids_ = sorted(np.unique(labels_arr).tolist(), key=lambda value: str(value))
        self.species_to_index_ = {species_id: idx for idx, species_id in enumerate(self.species_ids_)}
        self.species_names_ = (
            {species_id: str(species_id) for species_id in self.species_ids_}
            if species_names is None
            else {species_id: str(species_names.get(species_id, species_id)) for species_id in self.species_ids_}
        )
        self.mz_axis_ = None if mz_axis is None else np.asarray(mz_axis, dtype=np.float64)
        if self.mz_axis_ is not None and len(self.mz_axis_) != self.n_bins_:
            raise ValueError("mz_axis length must match spectra width.")

        self.rf_metrics_ = []
        self.means_ = {
            species_id: x[labels_arr == species_id].mean(axis=0).astype(np.float64)
            for species_id in self.species_ids_
        }
        negative_species_by_target = self._top_cosine_negative_species()
        self.weights_ = {
            species_id: self._train_species_weights(
                x,
                labels_arr,
                species_id,
                groups_arr,
                negative_species_by_target.get(species_id, set()),
            )
            for species_id in self.species_ids_
        }
        self.pure_score_matrix_ = self._pure_score_matrix_from_means()
        self.fitted_ = True
        return self

    def fit_from_species_prototypes(
        self,
        prototypes_matrix: np.ndarray,
        species_ids: Sequence[Any],
        species_names: Optional[Dict[Any, str]] = None,
        mz_axis: Optional[np.ndarray] = None,
    ):
        """Fit from one prototype per species, using uniform weights.

        This fallback keeps the class usable in code paths that only have
        species prototypes. It does not train RF importances because RF requires
        multiple pure spectra with positive/negative labels.
        """
        x = self._as_2d_float(prototypes_matrix)
        species_ids = list(species_ids)
        if len(species_ids) != x.shape[0]:
            raise ValueError("species_ids length must match prototype rows.")
        self.n_bins_ = int(x.shape[1])
        self.species_ids_ = list(species_ids)
        self.species_to_index_ = {species_id: idx for idx, species_id in enumerate(self.species_ids_)}
        self.species_names_ = (
            {species_id: str(species_id) for species_id in self.species_ids_}
            if species_names is None
            else {species_id: str(species_names.get(species_id, species_id)) for species_id in self.species_ids_}
        )
        self.mz_axis_ = None if mz_axis is None else np.asarray(mz_axis, dtype=np.float64)
        self.means_ = {
            species_id: x[idx].astype(np.float64)
            for idx, species_id in enumerate(self.species_ids_)
        }
        self.weights_ = {
            species_id: np.ones(self.n_bins_, dtype=np.float32)
            for species_id in self.species_ids_
        }
        self.rf_metrics_ = [{"species_id": sid, "status": "uniform_weights_from_prototype"} for sid in self.species_ids_]
        self.pure_score_matrix_ = self._pure_score_matrix_from_means()
        self.fitted_ = True
        return self

    def predict_scores(self, mixture_vector: np.ndarray) -> Dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("Modelo não ajustado. Execute fit() primeiro.")
        y = self._as_1d_float(mixture_vector)
        if y.shape[0] != self.n_bins_:
            raise ValueError(f"mixture_vector must have length {self.n_bins_}.")

        raw_scores = self._raw_scores(y)
        if self.adjust_baseline:
            scores, estimated_abundance, predicted_baseline = self._baseline_adjusted_scores(raw_scores)
        else:
            scores = raw_scores.copy()
            estimated_abundance = self._normalize_abundances(raw_scores)
            predicted_baseline = np.zeros_like(raw_scores)

        return {
            "species_ids": list(self.species_ids_),
            "species_names": [self.species_names_.get(sid, str(sid)) for sid in self.species_ids_],
            "scores": scores,
            "raw_scores": raw_scores,
            "estimated_abundance": estimated_abundance,
            "predicted_baseline": predicted_baseline,
        }

    def predict(self, mixture_vector: np.ndarray) -> Tuple[List[Any], List[float], Dict[str, Any]]:
        """Return predicted species ids, scores, and a detailed payload."""
        details = self.predict_scores(mixture_vector)
        scores = np.asarray(details["scores"], dtype=np.float64)
        order = np.argsort(-scores)
        if self.max_species is not None:
            order = order[: self.max_species]
        keep = [idx for idx in order if scores[idx] >= self.threshold]

        predicted_ids = [self.species_ids_[idx] for idx in keep]
        predicted_scores = [float(scores[idx]) for idx in keep]
        details.update(
            {
                "threshold": self.threshold,
                "adjust_baseline": self.adjust_baseline,
                "predicted_species_ids": predicted_ids,
                "predicted_scores": predicted_scores,
                "candidate_species_ids": [self.species_ids_[idx] for idx in order],
                "candidate_scores": [float(scores[idx]) for idx in order],
                "score_type": "rf_weighted_cosine_adjusted" if self.adjust_baseline else "rf_weighted_cosine",
                "rf_metrics": list(self.rf_metrics_),
            }
        )
        return predicted_ids, predicted_scores, details


RFWeightedCosineModel = RFCosineModel
