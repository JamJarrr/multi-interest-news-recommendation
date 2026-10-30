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

## Prepare and audit the data

From the repository root, install the dependencies and run:

```powershell
python -m pip install -r requirements.txt
python -m src.data.prepare_mind --root .
```

The command validates the schemas, builds a combined news catalog, splits the official training impressions by time into model training and validation sets, keeps the official development impressions as the final test set, and writes compressed CSV files under `data/processed/` plus compact audit summaries under `results/`.

## Cache the shared news vectors

All models use the same frozen `sentence-transformers/all-MiniLM-L6-v2` vectors. The cache includes the title and abstract and is deliberately excluded from Git:

```powershell
python -m src.data.cache_news_embeddings --root . --device cpu
```

## Run B0 and B1

B0 has no trainable parameters. B1 learns a shared 384-to-64 projection using four within-impression negatives per positive and validation NDCG@10 for early stopping:

```powershell
python -m src.training.run_b0 --root .
python -m src.training.train_b1 --root . --seed 42 --device cpu
python -m src.training.train_b1 --root . --seed 2026 --device cpu
python -m src.training.train_b1 --root . --seed 3407 --device cpu
python -m src.evaluation.summarize_b1 --root .
```

The baseline commands report NDCG@5, NDCG@10, Recall@10, and user-macro Minority Recall@10. Compact JSON results are written to `results/`; caches and checkpoints remain local.

The checked-in result summaries contain one deterministic B0 run, three B1 seeds, their mean and sample standard deviation, and a direct B1-minus-B0 comparison. These are reference results for the documented MIND-small split, not claims about other datasets or deployment settings.

## Baselines

- B0: frozen news embeddings, last-50 masked mean history pooling, and cosine similarity.
- B1: shared trainable projection followed by last-50 masked mean pooling and candidate matching.

Further models add single-interest attention, multi-interest attention, and a capacity-matched control.

## Reproducibility rules

- Use the same fixed time-based train, validation, and test files for every model.
- Use the same cached news embeddings for all models.
- Tune only on validation data and evaluate the test data after configuration is fixed.
- Save configurations, random seeds, aggregate metrics, and audit summaries.
