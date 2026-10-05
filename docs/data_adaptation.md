# Data Adaptation Guide

This guide describes how to adapt new MALDI-TOF data to the UnmixingMaldi workflow while keeping dataset-specific paths and private files outside the repository.

The public examples are built around DRIAMS and MARISMA pure-culture spectra. Other pure datasets or experimental mixtures can be used when they are converted to the same HDF5 schema and species mapping conventions.

## Data Classes

UnmixingMaldi uses two related data classes:

- Pure spectra: one species label per spectrum, used for pure-spectrum training and in-silico mixture generation.
- Mixture spectra: one or more species labels per spectrum, with proportions when quantitative composition is available.

Public DRIAMS and MARISMA data can be used to generate in-silico mixtures. Experimental mixtures from local studies should be stored outside git and converted with local scripts that write the schema below.

## Raw Pure-Spectrum Layout

Place raw public pure spectra under `data/raw/`:

```text
data/raw/
├── DRIAMS-A/
│   └── <species>/
│       └── <sample>/
└── MARISMA/
    └── <species>/
        └── <sample>/
```

The exporter in `src/code/data_generation/export_driams_marisma_pures_h5.py` expects species-level directories. Each sample can be:

- a Bruker acquisition folder containing `fid` and acquisition metadata,
- a two-column text spectrum with m/z and intensity values,
- an mzML or mzXML file when the optional parser dependencies are available.

Update `src/code/configs/data_generation/export_driams_marisma_pures.example.yaml` if your local folder names differ.

## Species Mapping

`data/species_mapping.csv` defines the canonical species names and integer IDs used by the example configs. Model outputs depend on this mapping.

When adapting data:

- normalize species names before export,
- map every included species to exactly one ID,
- keep IDs stable across pure, mixture, training, and validation files,
- retrain model heads from scratch if you add, remove, or reorder species.

The exporter can filter to the intersection between available public spectra and the mapping. The species count comes from the configured datasets and mapping. The example config samples up to 100 spectra per species and source dataset with `random_seed: 42`; change the cap, remove it, or change the seed in a local config when adapting data.

## HDF5 Schema

Public pure exports, generated mixtures, and adapted mixture datasets should use the same high-level layout:

```text
/
├── central
├── samples/
│   ├── id
│   ├── origin_dataset
│   ├── type
│   ├── num_species
│   ├── spectrum/
│   │   ├── mz
│   │   └── intensities
│   └── labels/
│       ├── species_id
│       ├── species_name_json
│       └── proportion
└── unstructured/
    └── config_json
```

Expected meanings:

- `central`: shared m/z axis for all spectra.
- `samples/id`: stable sample identifiers.
- `samples/origin_dataset`: source dataset name, such as `DRIAMS-A`, `MARISMA`, `IN_SILICO`, or a local study code.
- `samples/type`: sample type, such as `pure` or `mixture`.
- `samples/num_species`: number of species represented in each sample.
- `samples/spectrum/mz`: per-sample m/z axis when stored explicitly.
- `samples/spectrum/intensities`: dense intensity matrix aligned to `central`.
- `samples/labels/species_id`: integer species IDs.
- `samples/labels/species_name_json`: JSON-encoded species names for each sample.
- `samples/labels/proportion`: species proportions aligned with `species_id`.
- `unstructured/config_json`: JSON metadata describing preprocessing, generation, and provenance.

For pure spectra, `num_species` is normally `1` and `proportion` is normally `1.0`.

## Preprocessing Compatibility

The repository includes these preprocessing templates:

- `src/code/configs/preprocessing_pipelines/s_bc.yaml`
- `src/code/configs/preprocessing_pipelines/s_bc_n_vs__snip_sqrt_b9da_avg.yaml`
- `src/code/configs/preprocessing_pipelines/s_bc_vs_n__snip_log2_b9da_avg.yaml`

Use the same preprocessing pipeline for training and evaluation data unless you are explicitly testing a preprocessing variant. The transformer example uses `s_bc_n_vs__snip_sqrt_b9da_avg`; the RF/cosine reference pipeline is represented by `s_bc_vs_n__snip_log2_b9da_avg`.

## Adapting Experimental Mixtures

To integrate a local mixture dataset:

1. Keep raw spectra and metadata outside git, for example under `data/raw/<LOCAL_DATASET>/`.
2. Convert raw spectra to the HDF5 schema above.
3. Use the same `central` axis and species mapping as the model training data.
4. Store dataset-specific paths in a local YAML file or `.env`.
5. Point validation or evaluation configs to the generated local HDF5 file.

Do not commit local mixture spectra, sample manifests, generated HDF5 files, or exploratory scripts that reveal private dataset contents.

For the classical mixture benchmark adapters, local mixture manifests can also be referenced from `src/code/benchmarking/benchmark.yaml` under `datasets.experimental_mixtures.manifests`. Keep those CSV files under ignored local paths such as `data/manifests/mixtures/`.

## Local Path Policy

All machine-specific paths should live in one of these places:

- `.env`
- local copies of example YAML files,
- command-line arguments,
- ignored data/output directories.

Source files should remain portable and should not contain absolute paths to a workstation, institution, or private storage volume.
