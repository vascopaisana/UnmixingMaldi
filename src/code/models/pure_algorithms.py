"""Pure-spectrum benchmark algorithms."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC


PURE_ALGORITHMS = ("svm", "rf_multiclass", "knn", "logistic_l1")


def make_pure_algorithms(seed: int = 7, inner_jobs: int | None = None, algorithms_to_run: list[str] | None = None) -> dict[str, Any]:
    selected = set(algorithms_to_run or PURE_ALGORITHMS)
    algorithms = {
        "svm": make_pipeline(
            MaxAbsScaler(),
            LinearSVC(C=1.0, class_weight="balanced", max_iter=2500, random_state=seed),
        ),
        "rf_multiclass": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=inner_jobs,
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", n_jobs=inner_jobs),
        "logistic_l1": make_pipeline(
            MaxAbsScaler(),
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0,
                class_weight="balanced",
                multi_class="ovr",
                max_iter=1000,
                n_jobs=1,
                random_state=seed,
            ),
        ),
    }
    unknown = selected - set(algorithms)
    if unknown:
        raise ValueError(f"Unsupported pure algorithms: {sorted(unknown)}")
    return {name: estimator for name, estimator in algorithms.items() if name in selected}
