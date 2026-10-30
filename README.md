# Multi Interest News Recommendation

SC4001 group project studying whether multiple user-interest representations improve the ranking of news from less frequent interest categories.

## Dataset

The project uses the Microsoft MIND-small dataset. Dataset archives and extracted files are not committed to Git. Each collaborator should download `MINDsmall_train.zip` and `MINDsmall_dev.zip`, then place them under `data/raw/zips/` and run the project preprocessing scripts.

Expected local structure:

```text
data/raw/
|-- zips/
|   |-- MINDsmall_train.zip
|   `-- MINDsmall_dev.zip
|-- MINDsmall_train/
`-- MINDsmall_dev/
```

Only source code, configuration, documentation, and compact result summaries should be committed. Raw data, cached news embeddings, checkpoints, and large generated files remain local.

## Planned baselines

- B0: frozen news embeddings, masked mean history pooling, and cosine similarity.
- B1: shared trainable projection followed by masked mean pooling and candidate matching.

Further models add single-interest attention, multi-interest attention, and a capacity-matched control.

## Reproducibility rules

- Use the same fixed time-based train, validation, and test files for every model.
- Use the same cached news embeddings for all models.
- Tune only on validation data and evaluate the test data after configuration is fixed.
- Save configurations, random seeds, aggregate metrics, and audit summaries.
