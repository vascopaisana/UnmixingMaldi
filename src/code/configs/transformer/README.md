# Transformer Configs

This directory contains example transformer training configs.

## Files

- `default.example.yaml`: runnable transformer training template for the public in-silico mixture workflow.
- `ablations/preprocessing_pipeline.example.yaml`: small preprocessing-variant ablation template that expands into repeated transformer runs.

## Default Training Config

`default.example.yaml` expects these generated files:

```text
data/pures/driams_marisma/s_bc_n_vs__snip_sqrt_b9da_avg/pures_s_bc_n_vs__snip_sqrt_b9da_avg.h5
data/mixtures/IN_SILICO/driams_marisma/s_bc_n_vs__snip_sqrt_b9da_avg/mixtures_random_n50000_k2to6_alpha0p5-1p0-5p0_minab0p05_seed42.h5
```

The validation path in the example points to generated in-silico data so the config can be exercised after data generation. For model selection, replace it with an independent held-out generated set or a converted experimental mixture HDF5.

## Ablation Config

The ablation config keeps only public preprocessing variants:

- `transformer_best_s_bc_n_vs_snip_sqrt_b9da`
- `rf_cosine_best_s_bc_vs_n_snip_log2_b9da`

These names document the retained preprocessing templates. The ablation runner itself launches transformer training jobs; it does not run a separate RF or cosine benchmark workflow.

## Local Edits

When adapting configs:

- keep generated data under ignored directories,
- keep private validation paths in local YAML copies,
- keep species IDs aligned with `data/species_mapping.csv`,
- disable external logging unless credentials are configured,
- avoid committing machine-specific absolute paths.
