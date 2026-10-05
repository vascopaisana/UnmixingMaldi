# Transformer Training

This directory contains the transformer training entry points.

## Train From The Example Config

Generate the public in-silico training data first, then run:

```bash
python src/code/transformer/train_transformer.py \
  --config src/code/configs/transformer/default.example.yaml
```

The default config trains from scratch on the `s_bc_n_vs__snip_sqrt_b9da_avg` processed pure spectra and generated in-silico mixtures.

The example validation path points to generated in-silico data so the command is usable in a fresh public workflow. For real model selection, use an independent held-out generated set or a converted experimental mixture HDF5 file.

## Dry Run

```bash
python src/code/transformer/train_transformer.py \
  --config src/code/configs/transformer/default.example.yaml \
  --dry-run
```

The dry run resolves the config, checks the configured HDF5 files, builds data-module metadata, and prepares output directories. It is a post-data-generation validation step, not a no-data smoke test.

## Preprocessing Ablation Wrapper

```bash
python src/code/transformer/run_transformer_ablation.py \
  --config src/code/configs/transformer/ablations/preprocessing_pipeline.example.yaml \
  --dry-run
```

The wrapper expands variants from the ablation YAML and launches `train_transformer.py` for each variant. The included ablation config is limited to the public preprocessing templates retained in this repository.

## Outputs

Training writes generated artifacts under ignored directories:

```text
outputs/transformer/
checkpoints/transformer/
logs/
```

Comet logging is disabled in the example config. To enable it, provide credentials locally and update a private config copy.
