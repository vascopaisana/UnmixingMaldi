# Data Generation

This directory contains the public data-generation entry points.

## Export Public Pure Spectra

```bash
python src/code/data_generation/export_driams_marisma_pures_h5.py \
  --config src/code/configs/data_generation/export_driams_marisma_pures.example.yaml \
  --variant s_bc_n_vs_snip_sqrt_b9da
```

The exporter reads raw DRIAMS-A and MARISMA pure spectra from the paths configured in the YAML file. It filters species with `data/species_mapping.csv`, samples up to the configured per-species/per-dataset cap without replacement, applies the pre-mixture `s_bc` preprocessing pipeline, and writes an HDF5 file plus a metadata CSV under `data/pures/`.

Use the second public variant for the log2 reference pipeline:

```bash
python src/code/data_generation/export_driams_marisma_pures_h5.py \
  --config src/code/configs/data_generation/export_driams_marisma_pures.example.yaml \
  --variant s_bc_vs_n_snip_log2_b9da
```

## Generate In-Silico Mixtures

```bash
python src/code/data_generation/generate_training_mixtures.py \
  --config src/code/configs/data_generation/generate_training_mixtures.example.yaml \
  --variant s_bc_n_vs_snip_sqrt_b9da
```

This command reads the exported pre-bin pure HDF5 file, applies the variant-specific post-mixture pipeline, saves processed pures, and generates random in-silico mixtures.

The generated mixture filename encodes the main generation parameters:

```text
mixtures_random_n50000_k2to6_alpha0p5-1p0-5p0_minab0p05_seed42.h5
```

## Inputs And Outputs

Expected local inputs:

```text
data/raw/DRIAMS-A/
data/raw/MARISMA/
data/species_mapping.csv
```

Generated outputs:

```text
data/pures/
data/mixtures/
```

`data/raw/`, `data/pures/`, and `data/mixtures/` are ignored by git. The code creates output directories as needed.

## Adapting Data

For other pure datasets, update the source definitions in a local config copy. For experimental mixture datasets, convert spectra to the common HDF5 schema described in `docs/data_adaptation.md`.
