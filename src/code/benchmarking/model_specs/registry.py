"""Final benchmark model registry."""

from __future__ import annotations

from pathlib import Path

from benchmarking.core.config import BENCHMARKING_DIR
from benchmarking.model_specs.specs import ModelSpec


CONFIG_DIR = BENCHMARKING_DIR / "configs"

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "rf_cosine": ModelSpec(
        name="rf_cosine",
        task="mixture",
        implementation="models.rf_cosine.RFCosineModel",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/rf_cosine.py",
        config_path=CONFIG_DIR / "mixture/rf_cosine.yaml",
        supports_reference_modes=("DRIAMS_MARISMA_PURES",),
        required_config_keys=("model_parameters", "score_modes"),
    ),
    "mahe": ModelSpec(
        name="mahe",
        task="mixture",
        implementation="models.mahe.MaheModel",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/mahe.py",
        config_path=CONFIG_DIR / "mixture/mahe.yaml",
        required_config_keys=("model_parameters",),
    ),
    "nmf": ModelSpec(
        name="nmf",
        task="mixture",
        implementation="models.nmf.NMFDeconvolutionModel",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/nmf.py",
        config_path=CONFIG_DIR / "mixture/nmf.yaml",
        required_config_keys=("model_parameters",),
    ),
    "yang": ModelSpec(
        name="yang",
        task="mixture",
        implementation="models.yang.YangModel",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/yang.py",
        config_path=CONFIG_DIR / "mixture/yang.yaml",
        required_config_keys=("model_parameters",),
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        task="pure",
        implementation="sklearn.ensemble.RandomForestClassifier",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/pure_cross_domain.py",
        config_path=CONFIG_DIR / "pure/cross_domain.yaml",
        pure_algorithm="rf_multiclass",
    ),
    "svm": ModelSpec(
        name="svm",
        task="pure",
        implementation="sklearn.svm.LinearSVC",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/pure_cross_domain.py",
        config_path=CONFIG_DIR / "pure/cross_domain.yaml",
        pure_algorithm="svm",
    ),
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        task="pure",
        implementation="sklearn.linear_model.LogisticRegression",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/pure_cross_domain.py",
        config_path=CONFIG_DIR / "pure/cross_domain.yaml",
        pure_algorithm="logistic_l1",
    ),
    "knn": ModelSpec(
        name="knn",
        task="pure",
        implementation="sklearn.neighbors.KNeighborsClassifier",
        adapter_backend=BENCHMARKING_DIR / "model_specs/adapters/pure_cross_domain.py",
        config_path=CONFIG_DIR / "pure/cross_domain.yaml",
        pure_algorithm="knn",
    ),
}


def get_model_spec(model: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark model {model!r}.") from exc
