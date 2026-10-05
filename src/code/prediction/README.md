# Prediction Export

`predict_mixtures.py` writes one CSV row per mixture using the shared prediction schema in `helpers/prediction_schema.py`.

The transformer is the primary mixture model:

```bash
python src/code/prediction/predict_mixtures.py \
  --model transformer \
  --checkpoint <transformer-checkpoint.ckpt> \
  --mixtures-h5 <mixtures.h5> \
  --config src/code/configs/transformer/default.example.yaml \
  --output-csv outputs/predictions/transformer_predictions.csv
```

Classical models are fitted from a processed pure-spectrum HDF5 file before predicting mixtures:

```bash
python src/code/prediction/predict_mixtures.py \
  --model rf_cosine \
  --pures-h5 <processed-pures.h5> \
  --mixtures-h5 <mixtures.h5> \
  --config src/code/benchmarking/configs/mixture/rf_cosine.yaml \
  --output-csv outputs/predictions/rf_cosine_predictions.csv
```

Supported model names are:

- `transformer`
- `rf_cosine`
- `mahe`
- `nmf`
- `yang`

The mixture HDF5 file must contain the same `samples/spectrum/intensities`, `samples/labels/species_id`, `samples/labels/proportion`, and `samples/id` layout used by the data-generation scripts. Classical baselines also require a pure HDF5 file with one species label per sample.
