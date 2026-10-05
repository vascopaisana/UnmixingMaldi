#!/usr/bin/env python3
"""Preprocessing-pipeline and temporary H5 helpers."""

from __future__ import annotations

from benchmarking.core.preprocessing_backend import (
    DEFAULT_RF_RANDOM_STATE,
    apply_benchmark_stage,
    build_temporary_h5s_for_pipeline,
    cap_training_data,
    load_h5_records,
    load_mixture_spots,
    load_pure_validation_spots,
    load_top_pipeline_names,
    load_training_data,
    metadata_source_prefix_from_root,
    normalize_species_name,
    parse_vector,
    pipeline_h5_path,
    pure_score_matrix_from_means,
    safe_slug,
    species_means,
    threshold_candidates_from_scores,
    weighted_cosine,
)

__all__ = [
    "DEFAULT_RF_RANDOM_STATE",
    "apply_benchmark_stage",
    "build_temporary_h5s_for_pipeline",
    "cap_training_data",
    "load_h5_records",
    "load_mixture_spots",
    "load_pure_validation_spots",
    "load_top_pipeline_names",
    "load_training_data",
    "metadata_source_prefix_from_root",
    "normalize_species_name",
    "parse_vector",
    "pipeline_h5_path",
    "pure_score_matrix_from_means",
    "safe_slug",
    "species_means",
    "threshold_candidates_from_scores",
    "weighted_cosine",
]
