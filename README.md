# UnmixingMaldi

UnmixingMaldi is a research codebase for MALDI-TOF microbial mixture analysis. Its main mixture-unmixing workflow trains a transformer model to estimate species composition from mixed spectra, with RF/cosine, MAHE, NMF, and Yang-style models retained as benchmarking baselines.

The repository is organized around a public, configurable workflow:

1. Place raw DRIAMS and MARISMA pure-culture spectra under `data/raw/`.
2. Export selected pure spectra to HDF5 with a documented species mapping.
3. Generate in-silico mixtures from those public pure spectra.
4. Train the transformer on processed pures and generated mixtures.
5. Benchmark pure-spectrum classification and mixture-unmixing baselines when needed.
6. Write all generated data, checkpoints, logs, and outputs outside version control.

Private in-vitro mixture spectra, site-specific manifests, and exploratory scripts tied to private datasets are not distributed. New experimental mixture datasets can still be integrated by converting them to the HDF5 schema described in [docs/data_adaptation.md](docs/data_adaptation.md).

## Quick Start

```bash
git clone <repository-url>
cd UnmixingMaldi

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` so `UNMIXMALDI_ROOT` points to this repository. Then place DRIAMS-A and MARISMA under `data/raw/` as described in [Data Organization](#data-organization).

```bash
python src/code/data_generation/export_driams_marisma_pures_h5.py \
  --config src/code/configs/data_generation/export_driams_marisma_pures.example.yaml \
  --variant s_bc_n_vs_snip_sqrt_b9da

python src/code/data_generation/generate_training_mixtures.py \
  --config src/code/configs/data_generation/generate_training_mixtures.example.yaml \
  --variant s_bc_n_vs_snip_sqrt_b9da

python src/code/transformer/train_transformer.py \
  --config src/code/configs/transformer/default.example.yaml
```

This exports public pure spectra, generates the 50,000 in-silico mixtures, and trains the Transformer using the public example config. See [Public Workflow](#public-workflow) for details and alternative variants.

## Repository Layout

```text
UnmixingMaldi/
├── data/
│   └── species_mapping.csv
├── docs/
│   └── data_adaptation.md
├── src/code/
│   ├── benchmarking/
│   ├── configs/
│   │   ├── data_generation/
│   │   ├── preprocessing_pipelines/
│   │   └── transformer/
│   ├── data_generation/
│   ├── helpers/
│   ├── models/
│   ├── prediction/
│   └── transformer/
├── .env.example
├── .gitignore
├── LICENSE
└── requirements.txt
```

The `src/code/models` modules include model implementations used by the benchmark adapters and the per-mixture prediction CLI. The public workflow supports pure-spectrum export, in-silico mixture generation, transformer training, prediction export, and baseline benchmarking.

## Installation

Create an environment with Python 3.10 or newer, then install the package requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and point it at your local checkout:

```bash
cp .env.example .env
```

Edit `.env`:

```text
UNMIXMALDI_ROOT=/path/to/UnmixingMaldi
COMET_API_KEY=
```

`COMET_API_KEY` is optional. The provided transformer example disables Comet logging by default.

## Data Organization

The public workflow expects raw pure spectra to be provided locally:

```text
data/
├── raw/
│   ├── DRIAMS-A/
│   └── MARISMA/
├── pures/
└── mixtures/
```

`data/raw/` is ignored by git. Download DRIAMS and MARISMA from their public sources, then arrange each dataset so spectra can be discovered under species-level directories:

| Dataset | Source | Reference |
| --- | --- | --- |
| DRIAMS-A | DRIAMS dataset collection: https://doi.org/10.5061/dryad.bzkh1899q. This workflow uses DRIAMS-A from the collection. | Weis et al., "Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning", Nature Medicine (2022), DOI: https://doi.org/10.1038/s41591-021-01619-9 |
| MARISMA | Latest public dataset: "MARISMa: a routine MALDI-TOF MS dataset from 2018 to 2024 from Spain", https://doi.org/10.5281/zenodo.17201597. | Associated preprint: https://doi.org/10.1101/2025.05.31.657186 |

```text
data/raw/DRIAMS-A/<species>/<sample>/
data/raw/MARISMA/<species>/<sample>/
```

The exporter can read Bruker folders containing `fid` and acquisition metadata, two-column text spectra, and mzML/mzXML files when the relevant dependencies are installed.

The stable species identifiers used by the examples live in `data/species_mapping.csv`. By default, the exporter keeps the species found in the configured dataset intersection and samples up to 100 spectra per species and source dataset with `random_seed: 42`. If you adapt the repository to a different species panel, update the mapping and config together.

## Preprocessing Notation

Preprocessing pipelines are YAML files in `src/code/configs/preprocessing_pipelines/`.

- `S`: smoothing with a Savitzky-Golay filter.
- `BC`: baseline correction, using SNIP in the provided examples.
- `N`: normalization, typically sum/TIC normalization.
- `VS`: variance or intensity stabilization, such as `sqrt` or `log2`.
- `b9da_avg`: 9-Da binning with mean aggregation.

The retained example pipelines are:

- `s_bc.yaml`: smoothing plus baseline correction before in-silico mixture generation.
- `s_bc_n_vs__snip_sqrt_b9da_avg.yaml`: transformer-oriented pipeline with normalization before square-root stabilization.
- `s_bc_vs_n__snip_log2_b9da_avg.yaml`: RF/cosine-oriented reference pipeline with log2 stabilization before normalization.

The ablation examples keep only preprocessing variants that correspond to these public templates. Other ablations can be reproduced by editing or adding YAML files rather than carrying many near-duplicate configs.

## Public Workflow

Run commands from the repository root.

### 1. Export Public Pure Spectra

```bash
python src/code/data_generation/export_driams_marisma_pures_h5.py \
  --config src/code/configs/data_generation/export_driams_marisma_pures.example.yaml \
  --variant s_bc_n_vs_snip_sqrt_b9da
```

The example config reads raw DRIAMS-A and MARISMA spectra from `data/raw/`, filters species according to `data/species_mapping.csv`, applies the pre-mixture `s_bc` pipeline, and writes:

```text
data/pures/driams_marisma/s_bc_n_vs__snip_sqrt_b9da_avg/pures_prebin_s_bc.h5
data/pures/driams_marisma/s_bc_n_vs__snip_sqrt_b9da_avg/pures_prebin_s_bc_metadata.csv
```

Use the second variant for the log2 reference preprocessing:

```bash
python src/code/data_generation/export_driams_marisma_pures_h5.py \
  --config src/code/configs/data_generation/export_driams_marisma_pures.example.yaml \
  --variant s_bc_vs_n_snip_log2_b9da
```

### 2. Generate In-Silico Mixtures

```bash
python src/code/data_generation/generate_training_mixtures.py \
  --config src/code/configs/data_generation/generate_training_mixtures.example.yaml \
  --variant s_bc_n_vs_snip_sqrt_b9da
```

This reads the pre-bin pure HDF5 file, applies post-mixture preprocessing, saves processed pures, and creates in-silico mixtures:

```text
data/pures/driams_marisma/s_bc_n_vs__snip_sqrt_b9da_avg/pures_s_bc_n_vs__snip_sqrt_b9da_avg.h5
data/mixtures/IN_SILICO/driams_marisma/s_bc_n_vs__snip_sqrt_b9da_avg/mixtures_random_n50000_k2to6_alpha0p5-1p0-5p0_minab0p05_seed42.h5
```

The `s_bc_vs_n_snip_log2_b9da` variant creates the corresponding log2-normalized reference data.

### 3. Train The Transformer

After generating the example HDF5 files:

```bash
python src/code/transformer/train_transformer.py \
  --config src/code/configs/transformer/default.example.yaml
```

The default example trains from scratch using the `s_bc_n_vs__snip_sqrt_b9da_avg` processed pures and in-silico mixtures. The validation path in the example points to generated in-silico data so the command is runnable locally; replace it with an independent held-out generated set or an external mixture HDF5 for model selection.

The thesis experiments initialized the spectral encoder from the public `MaldiTransformerM` checkpoint: https://huggingface.co/gdewael/MaldiTransformer/blob/main/MaldiTransformerM.ckpt. The training code supports this through the `initialization` block; download the checkpoint locally and set:

```yaml
initialization:
  mode: encoder
  checkpoint: <path/to/MaldiTransformerM.ckpt>
```

`initialization.mode: encoder` loads compatible encoder weights and trains a new mixture head for the configured species set. The public example keeps `mode: scratch` so it can run without a downloaded checkpoint.

To inspect resolved settings without starting a full training run:

```bash
python src/code/transformer/train_transformer.py \
  --config src/code/configs/transformer/default.example.yaml \
  --dry-run
```

The dry run still checks the configured HDF5 paths and prepares output directories.

### 4. Run Transformer Preprocessing Variants

```bash
python src/code/transformer/run_transformer_ablation.py \
  --config src/code/configs/transformer/ablations/preprocessing_pipeline.example.yaml \
  --dry-run
```

This wrapper expands transformer runs from an ablation YAML. The included example is limited to the public preprocessing variants retained in this repository.

### 5. Predict Mixture Composition

The transformer is the primary model for mixture predictions:

```bash
python src/code/prediction/predict_mixtures.py \
  --model transformer \
  --checkpoint <transformer-checkpoint.ckpt> \
  --mixtures-h5 <mixtures.h5> \
  --config src/code/configs/transformer/default.example.yaml \
  --output-csv outputs/predictions/transformer_predictions.csv
```

The same script also supports classical baselines. These models are fitted from the pure spectra supplied with `--pures-h5` and then applied to the mixture HDF5 file:

```bash
python src/code/prediction/predict_mixtures.py \
  --model rf_cosine \
  --pures-h5 <processed-pures.h5> \
  --mixtures-h5 <mixtures.h5> \
  --config src/code/benchmarking/configs/mixture/rf_cosine.yaml \
  --output-csv outputs/predictions/rf_cosine_predictions.csv
```

The output CSV has one row per mixture. Supported model names are `transformer`, `rf_cosine`, `mahe`, `nmf`, and `yang`.

### 6. Run Baseline Benchmarks

The benchmark runner covers pure cross-domain classification, classical mixture baselines, and preprocessing pipeline ablations:

```bash
python src/code/benchmarking/run_benchmarks.py --dry-run
```

Pure benchmarks use local MARISMA and DRIAMS-A spectra under `data/raw/`. Mixture benchmarks use local mixture manifests configured in `src/code/benchmarking/benchmark.yaml`; these manifests are local inputs and are not committed.

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

## Outputs

Generated artifacts are ignored by git:

- `data/raw/`
- `data/pures/`
- `data/mixtures/`
- `outputs/`
- `checkpoints/`
- `logs/`
- `*.h5`
- generated `*.csv` files, except `data/species_mapping.csv`

The code creates these directories when needed. A fresh clone should contain code, configuration examples, documentation, and the species mapping only.

## Adapting New Data

Use [docs/data_adaptation.md](docs/data_adaptation.md) when adding a new pure-culture collection or a new experimental mixture dataset. The key requirement is to convert data to the same HDF5 layout used by the public workflow:

- shared `central` m/z axis
- sample metadata under `samples/`
- dense intensity matrix under `samples/spectrum/intensities`
- species labels and proportions under `samples/labels/`
- reproducibility metadata under `unstructured/config_json`

Keep local paths in YAML files, `.env`, or command-line arguments. Do not hard-code machine-specific absolute paths in source files.

## Documentation

- [Data adaptation guide](docs/data_adaptation.md)
- [Data generation commands](src/code/data_generation/README.md)
- [Data generation configs](src/code/configs/data_generation/README.md)
- [Transformer commands](src/code/transformer/README.md)
- [Transformer configs](src/code/configs/transformer/README.md)
- [Prediction export](src/code/prediction/README.md)
- [Benchmarking guide](src/code/benchmarking/README.md)

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
