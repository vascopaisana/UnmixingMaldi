# Benchmarking

This directory contains the benchmark framework for pure-spectrum and mixture experiments. It follows the structure of the original project while using anonymized, configurable dataset paths.

The main entry point is:

```bash
python src/code/benchmarking/run_benchmarks.py
```

Use a dry run first:

```bash
python src/code/benchmarking/run_benchmarks.py --dry-run
```

The default `benchmark.yaml` has `dry_run: true` so a fresh checkout prints the resolved commands without starting a full benchmark.

## What Is Included

- Pure cross-domain benchmarks between MARISMA and DRIAMS-A.
- Mixture benchmarks for RF+cosine, MAHE, NMF, and Yang-style baselines.
- Preprocessing pipeline and step-ablation configuration.
- Shared metrics, output handling, pipeline manifests, and adapter scripts.

The transformer is trained through `src/code/transformer/train_transformer.py`, not through this benchmark runner.

## Required Data

Pure benchmarks use local public raw spectra:

```text
data/raw/MARISMA/
data/raw/DRIAMS-A/
```

Mixture benchmarks use local mixture manifests configured in `benchmark.yaml`:

```yaml
datasets:
  experimental_mixtures:
    manifests:
      - ${UNMIXMALDI_ROOT}/data/manifests/mixtures/local_mixtures.example.csv
```

Those manifest files are local inputs and are ignored by git. For private or newly collected mixture datasets, create local manifests that follow the columns expected by the adapters and point the YAML to them.

## Manifest

`benchmark.yaml` controls the active run:

```yaml
benchmark:
  task: mixture
  experiment: full_pipeline
  config: all

runtime:
  shards: 1
  workers_per_shard: 4
  dry_run: true

paths:
  output_dir: ${UNMIXMALDI_ROOT}/outputs/benchmarks
  temp_dir: ${UNMIXMALDI_ROOT}/outputs/benchmarks/_tmp
  logs_dir: ${UNMIXMALDI_ROOT}/outputs/benchmarks/logs
```

Runnable values:

- `task`: `pure` or `mixture`.
- `experiment`: `full_pipeline` or `step_ablation`.
- `config` for pure benchmarks: `all`, `random_forest`, `svm`, `logistic_regression`, or `knn`.
- `config` for mixture benchmarks: `all`, `rf_cosine`, `mahe`, `nmf`, or `yang`.

Examples:

```bash
python src/code/benchmarking/run_benchmarks.py \
  --task pure \
  --experiment full_pipeline \
  --config all \
  --dry-run
```

```bash
python src/code/benchmarking/run_benchmarks.py \
  --task mixture \
  --experiment full_pipeline \
  --config rf_cosine \
  --dry-run
```

```bash
## Directory Map

```text
benchmarking/
├── benchmark.yaml
├── run_benchmarks.py
├── configs/
│   ├── ablations/
│   ├── mixture/
│   └── pure/
├── core/
├── experiments/
│   ├── mixture/
│   └── pure/
├── model_specs/
│   └── adapters/
└── analysis/
```

Generated benchmark outputs are written under `outputs/benchmarks/` and are ignored by git.

## Per-Mixture Prediction CSV

Use the shared prediction script when you want one output row per mixture:

```bash
python src/code/prediction/predict_mixtures.py \
  --model rf_cosine \
  --pures-h5 data/pures/driams_marisma/s_bc_vs_n__snip_log2_b9da_avg/pures_s_bc_vs_n__snip_log2_b9da_avg.h5 \
  --mixtures-h5 data/mixtures/IN_SILICO/driams_marisma/s_bc_vs_n__snip_log2_b9da_avg/mixtures_random_n50000_k2to6_alpha0p5-1p0-5p0_minab0p05_seed42.h5 \
  --config src/code/benchmarking/configs/mixture/rf_cosine.yaml \
  --output-csv outputs/predictions/rf_cosine_predictions.csv
```

Supported `--model` values are `rf_cosine`, `mahe`, `nmf`, `yang`, and `transformer`. Classical models require `--pures-h5`; transformer predictions require `--checkpoint`.
