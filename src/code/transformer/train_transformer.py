#!/usr/bin/env python3
"""Train or dry-run the configured Transformer model."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch
import yaml
from torch.utils.data import DataLoader
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.strategies import DDPStrategy

from data_generation.mixture_data import MixtureDataModule
from transformer.helpers_transformer import save_test_outputs_csv, save_test_summary_csv
from helpers.local_env import expand_config_values, load_local_env
from helpers.paths import CHECKPOINTS_DIR, CONFIG_DIR, MIXTURES_DIR, OUTPUTS_DIR, PROJECT_ROOT
from helpers.run_logging import setup_run_logging
from models.mixture_transformer import MixtureTransformer


DEFAULT_CONFIG = CONFIG_DIR / "transformer" / "default.example.yaml"
UNSUPPORTED_KEYS = {
    "use_" + "embedder",
    "loss_" + "type",
    "current_" + "loss_" + "type",
    "set_" + "loss_" + "type",
    "classification_" + "loss_" + "weight",
    "ranking_" + "loss_" + "weight",
    "abundance_" + "loss_" + "weight",
    "abundance_" + "loss",
    "use_" + "abundance_" + "head",
    "spectral_" + "alignment_" + "aug_" + "prob",
    "spectral_" + "alignment_" + "aug_" + "apply_to",
    "spectral_" + "alignment_" + "shift_min_bins",
    "spectral_" + "alignment_" + "shift_max_bins",
    "spectral_" + "alignment_" + "kernel_specs",
    "spectral_" + "alignment_" + "order",
    "spectral_" + "jitter_" + "prob",
    "spectral_" + "mixture_" + "noise_" + "prob",
}

logger = logging.getLogger(__name__)


def deep_merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config_path(path_value: str | Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = CONFIG_DIR / path
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_yaml(path: Path) -> dict:
    load_local_env()
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return expand_config_values(data)


def scan_unsupported_keys(value: Any, prefix: str = "") -> list[str]:
    hits = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in UNSUPPORTED_KEYS:
                hits.append(path)
            hits.extend(scan_unsupported_keys(child, path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(scan_unsupported_keys(child, f"{prefix}[{idx}]"))
    return hits


def load_config(config_path: Path, variant: str | None = None) -> tuple[dict, dict]:
    raw = read_yaml(config_path)
    source = {"config_path": str(config_path), "base_config": None, "variant": variant}

    if "base_config" in raw:
        base_path = resolve_config_path(config_path.parent / raw["base_config"])
        base = read_yaml(base_path)
        source["base_config"] = str(base_path)
        if "ablation" not in raw:
            raise ValueError("A config with base_config must also define ablation.variants")
        variants = raw.get("ablation", {}).get("variants", {}) or {}
        if variant is None:
            if len(variants) == 1:
                variant = next(iter(variants))
                source["variant"] = variant
            else:
                available = ", ".join(sorted(variants))
                raise ValueError(f"Ablation config requires --variant. Available variants: {available}")
        if variant not in variants:
            available = ", ".join(sorted(variants))
            raise ValueError(f"Unknown variant {variant!r}. Available variants: {available}")
        config = deep_merge(base, variants[variant])
        config["experiment"] = deep_merge(
            config.get("experiment", {}),
            {
                "type": "ablation",
                "ablation": raw.get("ablation", {}).get("name"),
                "variant": variant,
                "reference_config": str(base_path),
            },
        )
    else:
        config = raw

    config.setdefault("experiment", {}).setdefault("type", "default")
    config["source"] = source
    unsupported = scan_unsupported_keys(config)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"Unsupported obsolete Transformer setting(s): {joined}")
    return config, source


def resolve_data_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else MIXTURES_DIR / path


def resolve_checkpoint_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    local_checkpoint = CHECKPOINTS_DIR / path
    if local_checkpoint.exists():
        return local_checkpoint
    workspace_checkpoint = PROJECT_ROOT / "checkpoints" / path
    if workspace_checkpoint.exists():
        return workspace_checkpoint
    return local_checkpoint


def build_distributed_strategy(training_cfg: dict, devices):
    strategy = training_cfg.get("strategy")
    process_group_backend = training_cfg.get("process_group_backend")
    if not strategy and not process_group_backend:
        return None
    multi_device = devices == -1 or (isinstance(devices, int) and devices > 1) or isinstance(devices, (list, tuple))
    if not multi_device:
        return None
    if process_group_backend:
        return DDPStrategy(
            process_group_backend=str(process_group_backend),
            find_unused_parameters=bool(training_cfg.get("find_unused_parameters", False)),
        )
    return strategy


def phase_species_bounds(phase: dict, training_cfg: dict):
    dataset = str(phase["dataset"]).lower()
    if dataset == "pures":
        return int(phase.get("min_species", 1)), int(phase.get("max_species", 1))
    if dataset == "mixtures":
        return (
            phase.get("min_species", training_cfg.get("mixture_min_species", 2)),
            phase.get("max_species", training_cfg.get("mixture_max_species", 6)),
        )
    raise ValueError("training.phases entries must use dataset: pures or dataset: mixtures")


def build_phase_datamodule(config: dict, phase: dict | None = None) -> MixtureDataModule:
    paths_cfg = config["paths"]
    training_cfg = config["training"]
    prep = config.get("preprocessing", {})
    phase = phase or {}
    min_species = phase.get("min_species")
    max_species = phase.get("max_species")

    train_specs = phase.get("train_h5_specs")
    train_paths = paths_cfg.get("train_h5_paths")
    train_h5_filter_bounds = None
    train_h5_fractions = None
    if train_specs is not None:
        train_paths = []
        train_h5_filter_bounds = []
        train_h5_fractions = []
        for spec in train_specs:
            if isinstance(spec, dict):
                train_paths.append(spec["path"])
                train_h5_filter_bounds.append((spec.get("min_species"), spec.get("max_species")))
                train_h5_fractions.append(spec.get("fraction"))
            else:
                train_paths.append(spec)
                train_h5_filter_bounds.append((None, None))
                train_h5_fractions.append(None)

    if "val_h5_paths" in phase:
        val_paths = phase.get("val_h5_paths")
        val_h5_path_value = None
    elif "val_h5_path" in phase:
        val_paths = None
        val_h5_path_value = phase.get("val_h5_path")
    else:
        val_paths = paths_cfg.get("val_h5_paths")
        val_h5_path_value = paths_cfg.get("val_h5_path")
    val_h5_paths = [resolve_data_path(path) for path in val_paths] if val_paths is not None else None
    val_h5_path = resolve_data_path(val_h5_path_value)
    extra_val = phase.get("extra_val_h5_paths", paths_cfg.get("extra_val_h5_paths", [])) or []
    val_h5_filter_bounds = None
    val_h5_names = None
    val_global_h5_count = None
    if val_h5_paths is not None:
        val_global_h5_count = len(val_h5_paths)
        val_h5_filter_bounds = [
            (phase.get("val_min_species", prep.get("val_min_species")), phase.get("val_max_species", prep.get("val_max_species")))
            for _ in val_h5_paths
        ]
        val_h5_names = [str(path) for path in val_h5_paths]
        for entry in extra_val:
            if isinstance(entry, dict):
                val_h5_paths.append(resolve_data_path(entry["path"]))
                val_h5_filter_bounds.append((entry.get("min_species"), entry.get("max_species")))
                val_h5_names.append(str(entry.get("name", entry["path"])))
            else:
                val_h5_paths.append(resolve_data_path(entry))
                val_h5_filter_bounds.append((None, None))
                val_h5_names.append(str(entry))
        val_h5_path = None

    kwargs = {
        "val_h5_path": val_h5_path,
        "val_h5_paths": val_h5_paths,
        "val_global_h5_count": val_global_h5_count,
        "val_h5_filter_bounds": val_h5_filter_bounds,
        "val_h5_names": val_h5_names,
        "test_h5_path": resolve_data_path(paths_cfg["test_h5_path"]) if paths_cfg.get("test_h5_path") else None,
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(training_cfg["num_workers"]),
        "shuffle": bool(training_cfg.get("shuffle", True)),
        "pin_memory": training_cfg.get("pin_memory"),
        "persistent_workers": training_cfg.get("persistent_workers"),
        "prefetch_factor": training_cfg.get("prefetch_factor"),
        "fixed_length": prep.get("trf_fixed_length"),
        "top_k_peaks": prep.get("trf_top_k_peaks"),
        "pure_top_k_peaks": prep.get("pure_top_k_peaks"),
        "target_bins": prep.get("target_bins"),
        "min_species": min_species,
        "max_species": max_species,
        "val_min_species": phase.get("val_min_species", prep.get("val_min_species")),
        "val_max_species": phase.get("val_max_species", prep.get("val_max_species")),
        "train_fraction": phase.get("sampling", {}).get("fraction", phase.get("train_fraction")),
        "train_max_samples": phase.get("sampling", {}).get("max_samples", phase.get("train_max_samples")),
        "train_subsample_seed": phase.get("sampling", {}).get(
            "seed",
            phase.get("train_subsample_seed", training_cfg.get("train_subsample_seed", 42)),
        ),
    }
    if train_paths is not None:
        return MixtureDataModule(
            train_h5_paths=[resolve_data_path(path) for path in train_paths],
            train_h5_filter_bounds=train_h5_filter_bounds,
            train_h5_fractions=train_h5_fractions,
            **kwargs,
        )
    return MixtureDataModule(train_h5_path=resolve_data_path(paths_cfg["train_h5_path"]), **kwargs)


def resolve_phases(config: dict) -> list[dict]:
    phases = config.get("training", {}).get("phases")
    if not phases:
        raise ValueError("Current Transformer config must define training.phases")
    resolved = []
    for idx, phase in enumerate(phases, 1):
        item = copy.deepcopy(phase)
        item["dataset"] = str(item["dataset"]).lower()
        item["name"] = item.get("name") or ("pure_pretrain" if item["dataset"] == "pures" else "mixtures")
        item["min_species"], item["max_species"] = phase_species_bounds(item, config["training"])
        item["epochs"] = int(item.get("epochs", config["training"].get("max_epochs", 1)))
        item["lr"] = float(item.get("lr", config["training"]["lr"]))
        item["phase_index"] = idx
        resolved.append(item)
    return resolved


def build_model(config: dict, n_species: int) -> MixtureTransformer:
    model_cfg = config["model"]
    training_cfg = config["training"]
    return MixtureTransformer(
        n_species=n_species,
        transformer=model_cfg["transformer"],
        head=model_cfg["head"],
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        lr_decay_factor=float(training_cfg.get("lr_decay_factor", 1.0)),
        warmup_steps=int(training_cfg.get("warmup_steps", 30)),
        dropout=float(model_cfg.get("dropout", model_cfg["transformer"].get("dropout", 0.2))),
        ranking_margin=float(model_cfg["ranking"].get("margin", 0.5)),
        ranking_max_negatives=int(model_cfg["ranking"].get("max_negatives", 20)),
        ranking_negative_mode=model_cfg["ranking"].get("negative_mode", "hard"),
        score_temperature=float(model_cfg.get("score_temperature", 0.5)),
        presence_threshold=float(model_cfg.get("presence_threshold", 0.5)),
    )


def load_encoder_weights(model: MixtureTransformer, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    target = model.state_dict()
    copied = []
    skipped = []
    remapped = {}
    prefixes = ("spectrum_embedder.", "transformer.")
    for key, value in state.items():
        new_key = None
        if key.startswith("spectrum_embedder."):
            new_key = key
        elif key.startswith("transformer."):
            new_key = "spectrum_embedder." + key[len("transformer.") :]
        if new_key is None:
            continue
        if new_key in target and target[new_key].shape == value.shape:
            remapped[new_key] = value
            copied.append(new_key)
        else:
            skipped.append(key)
    if not remapped:
        raise ValueError(f"No compatible encoder weights found in {checkpoint_path}")
    incompatible = model.load_state_dict(remapped, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "encoder_loaded": True,
        "head_loaded": False,
        "copied_keys": len(copied),
        "skipped_keys": len(skipped),
        "missing_after_load": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_after_load": list(getattr(incompatible, "unexpected_keys", [])),
        "checkpoint_species": checkpoint.get("hyper_parameters", {}).get("n_species"),
        "current_species": model.n_species,
    }


def load_phase_weights(model: MixtureTransformer, checkpoint_path: str | Path | None) -> bool:
    """Load only model weights from a phase checkpoint, leaving optimizer state behind."""
    if not checkpoint_path:
        print("No best checkpoint recorded for this phase; continuing from current weights.")
        return False
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading phase checkpoint weights only: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    target = model.state_dict()
    incompatible_shapes = [
        key
        for key, value in state_dict.items()
        if key in target and tuple(target[key].shape) != tuple(value.shape)
    ]
    if incompatible_shapes:
        raise ValueError(
            f"Cannot load phase checkpoint {checkpoint_path}; incompatible tensor shapes for: "
            + ", ".join(incompatible_shapes[:20])
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"  Missing keys while loading phase checkpoint: {missing}")
    if unexpected:
        print(f"  Unexpected keys while loading phase checkpoint: {unexpected}")
    return True


def initialize_model(config: dict, n_species: int) -> tuple[MixtureTransformer, dict]:
    init_cfg = config.get("initialization", {"mode": "scratch"})
    mode = str(init_cfg.get("mode", "scratch")).lower()
    checkpoint_path = resolve_checkpoint_path(init_cfg.get("checkpoint"))
    info = {
        "mode": mode,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "encoder_loaded": False,
        "head_loaded": False,
        "checkpoint_species": None,
        "current_species": n_species,
    }
    if mode == "scratch":
        return build_model(config, n_species), info
    if checkpoint_path is None:
        raise ValueError(f"initialization.mode={mode!r} requires initialization.checkpoint")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initialization checkpoint not found: {checkpoint_path}")
    if mode == "full":
        model = MixtureTransformer.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            strict=True,
        )
        if int(model.n_species) != int(n_species):
            raise ValueError(
                f"Full checkpoint species mismatch: checkpoint has {model.n_species}, current data has {n_species}. "
                "Use initialization.mode: encoder for a new head."
            )
        info.update({"encoder_loaded": True, "head_loaded": True, "checkpoint_species": model.n_species})
        return model, info
    if mode == "encoder":
        model = build_model(config, n_species)
        info.update(load_encoder_weights(model, checkpoint_path))
        return model, info
    raise ValueError("initialization.mode must be scratch, encoder, or full")


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or "run"


def output_dir_for(config: dict, run_timestamp: str) -> Path:
    base = OUTPUTS_DIR / config["paths"]["output_dir"]
    experiment = config.get("experiment", {})
    if experiment.get("type") == "ablation":
        base = OUTPUTS_DIR / "transformer" / "ablations" / safe_slug(experiment.get("ablation")) / safe_slug(experiment.get("variant"))
    return base / run_timestamp


def checkpoint_dir_for(config: dict, run_timestamp: str) -> Path:
    experiment = config.get("experiment", {})
    if experiment.get("type") == "ablation":
        return (
            CHECKPOINTS_DIR
            / "transformer"
            / "ablations"
            / safe_slug(experiment.get("ablation"))
            / safe_slug(experiment.get("variant"))
            / run_timestamp
        )
    return CHECKPOINTS_DIR / "transformer" / "default" / run_timestamp


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def manifest_dict(
    config: dict,
    run_dir: Path,
    checkpoint_dir: Path,
    phases: list[dict],
    phase_counts: list[dict],
    init_info: dict,
) -> dict:
    return {
        "experiment": config.get("experiment", {}),
        "source": config.get("source", {}),
        "initialization": init_info,
        "model": config["model"],
        "preprocessing": config.get("preprocessing", {}),
        "training": {k: v for k, v in config["training"].items() if k != "phases"},
        "phases": phase_counts or phases,
        "paths": config["paths"],
        "output_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_monitor": config["training"].get("checkpoint_monitor"),
        "git_commit": git_commit(),
    }


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def print_run_summary(
    config: dict,
    run_dir: Path,
    checkpoint_dir: Path,
    phases: list[dict],
    phase_counts: list[dict],
    init_info: dict,
) -> None:
    model_cfg = config["model"]
    transformer = model_cfg["transformer"]
    head = model_cfg["head"]
    ranking = model_cfg["ranking"]
    experiment = config.get("experiment", {})
    print("=" * 60)
    print("MIXTURE TRANSFORMER")
    print("=" * 60)
    print("Experiment")
    print(f"  Type                  : {experiment.get('type', 'default')}")
    if experiment.get("ablation"):
        print(f"  Ablation              : {experiment.get('ablation')}")
        print(f"  Variant               : {experiment.get('variant')}")
    print("Initialization")
    print(f"  Mode                  : {init_info.get('mode')}")
    print(f"  Checkpoint            : {init_info.get('checkpoint')}")
    print(f"  Encoder loaded        : {'yes' if init_info.get('encoder_loaded') else 'no'}")
    print(f"  Head loaded           : {'yes' if init_info.get('head_loaded') else 'no'}")
    print("Architecture")
    print("  Encoder               : Transformer")
    print(f"  Depth                 : {transformer.get('depth')}")
    print(f"  Dimension             : {transformer.get('dim')}")
    print(f"  Attention heads       : {transformer.get('n_heads')}")
    print(f"  Dropout               : {transformer.get('dropout')}")
    print(f"  Reduction             : {transformer.get('reduce')}")
    print(f"  Output dimension      : {transformer.get('output_head_dim')}")
    print("Head")
    print(f"  Type                  : {head.get('type')}")
    if head.get("type") == "prototype":
        proto = head.get("prototype", {})
        print(f"  Prototype dimension   : {proto.get('dim')}")
        print(f"  Similarity            : {proto.get('similarity')}")
    print("Ranking")
    print(f"  Margin                : {ranking.get('margin')}")
    print(f"  Negative mode         : {ranking.get('negative_mode')}")
    print(f"  Maximum negatives     : {ranking.get('max_negatives')}")
    print("Preprocessing")
    prep = config.get("preprocessing", {})
    print(f"  Pipeline              : {prep.get('pipeline')}")
    print(f"  Baseline correction   : {prep.get('baseline')}")
    print(f"  Variance stabilization: {prep.get('variance_stabilization')}")
    print(f"  Fixed length          : {prep.get('trf_fixed_length')}")
    print(f"  Top-k peaks           : {prep.get('trf_top_k_peaks')}")
    print("Training phases")
    for phase in phase_counts or phases:
        print(f"  Phase {phase.get('phase_index')}: {phase.get('dataset')}")
        print(f"    Available samples   : {phase.get('available_samples', 'not resolved')}")
        print(f"    Used samples        : {phase.get('used_samples', 'not resolved')}")
        print(f"    Sampling            : {phase.get('sampling')}")
        print(f"    Epochs              : {phase.get('epochs')}")
        print(f"    LR                  : {phase.get('lr')}")
    print("Execution")
    print(f"  Output directory      : {run_dir}")
    print(f"  Checkpoint directory  : {checkpoint_dir}")
    print(f"  Batch size            : {config['training'].get('batch_size')}")
    print(f"  Workers               : {config['training'].get('num_workers')}")
    print(f"  Devices               : {config['training'].get('devices')}")
    print("=" * 60)


def prepare_phase_counts(config: dict, phases: list[dict]) -> tuple[MixtureDataModule, list[dict]]:
    main_dm = build_phase_datamodule(config)
    main_dm.setup()
    counts = []
    for phase in phases:
        unsampled_phase = copy.deepcopy(phase)
        unsampled_phase.pop("sampling", None)
        unsampled_phase.pop("train_fraction", None)
        unsampled_phase.pop("train_max_samples", None)
        available_dm = build_phase_datamodule(config, unsampled_phase)
        available_dm.setup()
        phase_dm = build_phase_datamodule(config, phase)
        phase_dm.setup()
        item = copy.deepcopy(phase)
        item["available_samples"] = len(available_dm.train_dataset)
        item["used_samples"] = len(phase_dm.train_dataset)
        item["validation_samples"] = len(phase_dm.val_dataset) if phase_dm.val_dataset is not None else None
        item["test_samples"] = len(phase_dm.test_dataset) if phase_dm.test_dataset is not None else None
        counts.append(item)
    return main_dm, counts


def log_comet_parameters(comet_logger, manifest: dict) -> None:
    if comet_logger is None:
        return
    flat = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}/{key}" if prefix else str(key), child)
        elif isinstance(value, list):
            for idx, child in enumerate(value, 1):
                visit(f"{prefix}/{idx}", child)
        else:
            flat[prefix] = value

    visit("", manifest)
    try:
        comet_logger.experiment.log_parameters(flat)
    except Exception as exc:
        print(f"Comet parameter logging issue: {exc}")


def make_logger(config: dict, run_timestamp: str):
    comet_cfg = config.get("comet", {}) or {}
    if not bool(comet_cfg.get("enabled", False)):
        return False
    comet_logger = CometLogger(
        api_key=comet_cfg.get("api_key") or os.environ.get("COMET_API_KEY"),
        workspace=comet_cfg.get("workspace"),
        project=comet_cfg.get("project"),
    )
    run_name = f"{comet_cfg.get('run_name', 'mixture_transformer')}_{run_timestamp}"
    try:
        comet_logger.experiment.set_name(run_name)
    except Exception as exc:
        print(f"Comet naming issue: {exc}")
    return comet_logger


def validation_prediction_batches(trainer: Trainer, model: MixtureTransformer, dm: MixtureDataModule, ckpt_path: str):
    """Predict the global mixture-validation datasets with the selected checkpoint."""
    dm.setup("validate")
    if dm.val_datasets is None:
        loader = DataLoader(dm.val_dataset, **dm._dataloader_kwargs(shuffle=False))
        return trainer.predict(model, dataloaders=loader, ckpt_path=ckpt_path)

    pred_batches = []
    names = dm.val_dataset_names or [f"validation_{idx}" for idx in range(len(dm.val_datasets))]
    n_global = dm.val_global_h5_count or len(dm.val_datasets)
    for dataset_index, (dataset, name) in enumerate(zip(dm.val_datasets[:n_global], names[:n_global])):
        loader = DataLoader(dataset, **dm._dataloader_kwargs(shuffle=False))
        batches = trainer.predict(model, dataloaders=loader, ckpt_path=ckpt_path)
        for batch in batches:
            batch["validation_dataset"] = str(name)
            batch["validation_dataset_index"] = int(dataset_index)
        pred_batches.extend(batches)
    return pred_batches


def train(config: dict, config_path: Path, dry_run: bool = False) -> None:
    seed = int(config.get("seed", config.get("training", {}).get("seed", 42)))
    seed_everything(seed, workers=True)
    run_timestamp = os.environ.get("UNMIXMALDI_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["UNMIXMALDI_RUN_TIMESTAMP"] = run_timestamp

    phases = resolve_phases(config)
    run_dir = output_dir_for(config, run_timestamp)
    checkpoint_dir = checkpoint_dir_for(config, run_timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dm, phase_counts = prepare_phase_counts(config, phases)

    if dry_run:
        init_cfg = config.get("initialization", {"mode": "scratch"})
        ckpt = resolve_checkpoint_path(init_cfg.get("checkpoint"))
        mode = str(init_cfg.get("mode", "scratch")).lower()
        init_info = {
            "mode": mode,
            "checkpoint": str(ckpt) if ckpt else None,
            "encoder_loaded": False,
            "head_loaded": False,
            "checkpoint_exists": bool(ckpt and ckpt.exists()),
            "current_species": dm.n_species,
        }
        if mode == "encoder" and ckpt and ckpt.exists():
            dry_model = build_model(config, dm.n_species)
            init_info.update(load_encoder_weights(dry_model, ckpt))
        elif mode == "full" and ckpt and ckpt.exists():
            dry_model = MixtureTransformer.load_from_checkpoint(ckpt, map_location="cpu", strict=True)
            init_info.update(
                {
                    "encoder_loaded": True,
                    "head_loaded": True,
                    "checkpoint_species": dry_model.n_species,
                    "current_species": dm.n_species,
                    "compatible": int(dry_model.n_species) == int(dm.n_species),
                }
            )
        manifest = manifest_dict(config, run_dir, checkpoint_dir, phases, phase_counts, init_info)
        save_manifest(run_dir / "run_manifest.yaml", manifest)
        print_run_summary(config, run_dir, checkpoint_dir, phases, phase_counts, init_info)
        print("Dry-run complete; no training was started.")
        return

    model, init_info = initialize_model(config, dm.n_species)
    training_cfg = config["training"]
    if torch.cuda.is_available():
        accelerator = training_cfg.get("accelerator", "gpu")
        devices = training_cfg.get("devices", -1)
    else:
        accelerator = "cpu"
        devices = 1

    manifest = manifest_dict(config, run_dir, checkpoint_dir, phases, phase_counts, init_info)
    save_manifest(run_dir / "run_manifest.yaml", manifest)
    print_run_summary(config, run_dir, checkpoint_dir, phases, phase_counts, init_info)

    comet_logger = make_logger(config, run_timestamp)
    log_comet_parameters(comet_logger, manifest)

    best_checkpoint_path = None
    resume_checkpoint_path = None
    completed_epochs = 0
    for phase in phases:
        phase_dm = build_phase_datamodule(config, phase)
        phase_dm.setup()
        model.hparams.lr = float(phase["lr"])
        safe_phase_name = safe_slug(phase["name"])
        monitor = phase.get("checkpoint_monitor", training_cfg.get("checkpoint_monitor", "val_pr_auc_loss_tiebreak"))
        mode = phase.get("checkpoint_mode", training_cfg.get("checkpoint_mode", "max"))
        filename = "best-{epoch:02d}-{val_pr_auc:.4f}-{val_loss:.4f}" if monitor == "val_pr_auc_loss_tiebreak" else f"best-{{epoch:02d}}-{{{monitor}:.4f}}"
        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir / safe_phase_name,
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                filename=filename,
            )
        ]
        patience = phase.get("patience", training_cfg.get("patience"))
        if patience is not None:
            callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=int(patience)))
        target_epochs = completed_epochs + int(phase["epochs"])
        strategy = build_distributed_strategy(training_cfg, devices)
        strategy_kwargs = {"strategy": strategy} if strategy is not None else {}
        accelerator_kwargs = {} if isinstance(strategy, DDPStrategy) else {"accelerator": accelerator}
        trainer = Trainer(
            devices=devices,
            max_epochs=target_epochs,
            logger=comet_logger,
            callbacks=callbacks,
            gradient_clip_val=float(training_cfg.get("gradient_clip_val", 1.0)),
            accumulate_grad_batches=int(training_cfg.get("accumulate_grad_batches", 1)),
            log_every_n_steps=int(training_cfg.get("log_every_n_steps", 10)),
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            **accelerator_kwargs,
            **strategy_kwargs,
        )
        trainer.fit(model, datamodule=phase_dm, ckpt_path=resume_checkpoint_path)
        phase_best_checkpoint_path = callbacks[0].best_model_path if callbacks else None
        if phase["dataset"] != "pures" and phase_best_checkpoint_path:
            best_checkpoint_path = phase_best_checkpoint_path or best_checkpoint_path
        phase_last_checkpoint = checkpoint_dir / f"{safe_phase_name}_last.ckpt"
        trainer.save_checkpoint(phase_last_checkpoint)
        if phase["dataset"] == "pures":
            load_phase_weights(model, phase_best_checkpoint_path or phase_last_checkpoint)
            resume_checkpoint_path = None
            completed_epochs = 0
        else:
            resume_checkpoint_path = str(phase_last_checkpoint)
            completed_epochs = int(trainer.current_epoch)

    strategy = build_distributed_strategy(training_cfg, devices)
    strategy_kwargs = {"strategy": strategy} if strategy is not None else {}
    accelerator_kwargs = {} if isinstance(strategy, DDPStrategy) else {"accelerator": accelerator}
    final_trainer = Trainer(
        devices=devices,
        logger=comet_logger,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        **accelerator_kwargs,
        **strategy_kwargs,
    )
    ckpt_path = best_checkpoint_path or resume_checkpoint_path
    validation_metrics = final_trainer.validate(model, datamodule=dm, ckpt_path=ckpt_path)
    preds = validation_prediction_batches(final_trainer, model, dm, ckpt_path)
    validation_outputs_csv = run_dir / f"validation_outputs_{run_timestamp}.csv"
    validation_summary_csv = run_dir / f"validation_summary_{run_timestamp}.csv"
    outputs_df = save_test_outputs_csv(preds, validation_outputs_csv, species_names=dm.species_names)
    summary = save_test_summary_csv(
        outputs_df,
        validation_summary_csv,
        extra_info={
            "model": "transformer",
            "evaluation_split": "validation",
            "initialization": config.get("initialization", {}).get("mode", "scratch"),
            "lr": training_cfg["lr"],
            "batch_size": training_cfg["batch_size"],
            "best_checkpoint": ckpt_path,
            "validation_metrics": validation_metrics,
        },
    )
    print("\nValidation summary:")
    print(summary)
    print(f"\nOutputs saved in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Transformer YAML config.")
    parser.add_argument("--variant", default=None, help="Variant name for ablation YAMLs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate config without training.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config, _ = load_config(config_path, variant=args.variant)
    setup_run_logging("train_transformer")
    try:
        train(config, config_path=config_path, dry_run=args.dry_run)
    except Exception:
        logger.exception("Run failed")
        raise


if __name__ == "__main__":
    main()
