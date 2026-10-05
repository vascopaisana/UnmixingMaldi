# Data Generation Configs

This directory contains example YAML files for preparing public pure spectra and generating in-silico mixtures.

## Files

- `export_driams_marisma_pures.example.yaml`: reads local DRIAMS-A and MARISMA raw pure spectra, filters species, applies pre-mixture preprocessing, and writes a pure-spectrum HDF5 file.
- `generate_training_mixtures.example.yaml`: reads the exported pure HDF5 file, applies post-mixture preprocessing, saves processed pures, and generates in-silico mixture HDF5 data.

Both configs use `${UNMIXMALDI_ROOT}` so they can be moved between machines without editing source code.

## Variants

The export config provides two public variants:

- `s_bc_n_vs_snip_sqrt_b9da`
- `s_bc_vs_n_snip_log2_b9da`

Both export pre-bin pures with `s_bc.yaml`. The downstream mixture config applies the variant-specific normalization, variance stabilization, and binning.

The mixture config provides matching variants:

- `s_bc_n_vs_snip_sqrt_b9da`: transformer-oriented preprocessing.
- `s_bc_vs_n_snip_log2_b9da`: RF/cosine-oriented reference preprocessing.

## Expected Inputs

Before running either config, place raw public spectra under:

```text
data/raw/DRIAMS-A/
data/raw/MARISMA/
```

The species mapping is read from:

```text
data/species_mapping.csv
```

The example export config derives the species count from the configured dataset intersection and samples up to 100 spectra per species and source dataset with `random_seed: 42`. Sampling is random without replacement. Each configured source uses `random_seed` plus its position in the `sources` list, so the example order gives MARISMA seed 42 and DRIAMS-A seed 43. Set `max_per_species_per_dataset: null` to keep every discovered spectrum, or change the cap and seed in a local config copy.

## Generated Outputs

Generated files are written under:

```text
data/pures/
data/mixtures/
```

These directories are ignored by git. Commit new example configs only when they are generic and do not expose local dataset names, private paths, sample IDs, or generated outputs.
