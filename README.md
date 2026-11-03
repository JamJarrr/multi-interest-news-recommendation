# Multi Interest News Recommendation

SC4001 group project: does retaining multiple user-interest vectors help rank
clicked news from less frequent interest categories, without losing overall
ranking quality?

The original B0/B1 results are preserved in `results/` and
`results/reference/`. The completed shared-protocol experiments are written
to `results/experiments/main/`. Read `docs/EXPERIMENT_PROTOCOL.md` for the
definitions and limitations. Start with [the main result notes](docs/RESULTS.md) for the main findings.

## Environment

Run commands from this repository's root in WSL/Linux. The local environment
is `.venv/` (Python 3.11, PyTorch 2.6.0 with CUDA 12.4), verified on an RTX 4060
Laptop GPU with 8 GB VRAM.

```bash
conda activate "$PWD/.venv"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For a new environment:

```bash
conda create -p "$PWD/.venv" python=3.11 pip -y
conda activate "$PWD/.venv"
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the exact packages used for this run on
Linux/CUDA. `requirements.txt` specifies broader project dependencies.
If PyPI cannot be reached, the installed run used the Tsinghua mirror:

```bash
python -m pip install -r requirements-lock.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

A Colab GPU runtime is a fallback. Clone/upload the repository there, install a
compatible environment, put data in the same relative paths, and explicitly
save cache, checkpoints, and results to persistent storage. WSL's SSH keys and
files are not automatically available in Colab. The current main run is local.

## Data and news vectors

Read and accept the MIND dataset terms at https://msnews.github.io/ and download
only `MINDsmall_train.zip` and `MINDsmall_dev.zip`. Place them in
`data/raw/zips/`. The helper below accepts archives with or without an oute
directory, checks ZIP integrity, and writes the two required TSVs in the expected
locations without overwriting different existing content.

```bash
python scripts/prepare_local.py
python -m src.data.prepare_mind --root . --results-dir results/local_data
python -m src.data.cache_news_embeddings --root . --device cuda --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

Expected locations:

```text
data/raw/zips/MINDsmall_train.zip
data/raw/zips/MINDsmall_dev.zip
data/raw/MINDsmall_train/news.tsv
data/raw/MINDsmall_train/behaviors.tsv
data/raw/MINDsmall_dev/news.tsv
data/raw/MINDsmall_dev/behaviors.tsv
data/processed/behaviors_{train,validation,test}.csv.gz
data/processed/news_catalog.csv.gz
cache/news_embeddings/all_minilm_l6_v2/
```

The current ZIP hashes differ from the collaborator's original archives.
`results/local_data/source_manifest.json` records the actual archives and TSV
hashes. All original audit statistics match; original B0 and B1 seed 42 were
also reproduced within 0.0001 in the four metrics. See
`results/reproduction/verification.json`. Do not claim byte-identical data
solely from these aggregate checks.

All models use the SAME cached normalized MiniLM vectors (title + abstract),
time split, last-50 model history, and full evaluation candidate sets. Original
train is split by time into 125,572 training and 31,393 validation impressions;
official dev is the final 73,152-impression test set. Do not regenerate the
cache inside an already locked run: cache identity will change.

## Models

- B0: mean of frozen 384-d vectors, then cosine matching; no training.
- B1: shared 384-to-64 projection, mean pooling, cosine matching; 24,640 parameters.
- B2: the same projection and one leaned attention query; 24,704 parameters.
- B3 K=2: two attention queries, max candidate-interest cosine; 24,768 parameters.
- B3 K=4: four attention queries, max candidate-interest cosine; 24,896 parameters.
- B4: ONE 64-d interest with a wider nonlinear attention scorer; 24,899 parameters.
  It matches B3 K=4 within 3 parameters (0.0121%), using all extra weights.

## Run experiments

Validate the implementation first:

```bash
python -m unittest discover -s tests -v
python -m src.training.run_interests train --smoke --run smoke_check --seeds 42
python scripts/reproduce_baselines.py
```

Run the complete sequence:

```bash
python scripts/run_pipeline.py
```

Or run the stages separately:

```bash
python scripts/run_pipeline.py --stage train
python scripts/run_pipeline.py --stage evaluate
python scripts/run_pipeline.py --stage controlled
python scripts/run_pipeline.py --stage summarize
python scripts/verify_experiment.py
```

Training uses ONLY validation NDCG@10 for early stopping. The common config is
`configs/main.json`: seeds 42/2026/3407, four within-impression negatives pe
positive, batch size 1024, AdamW leaning rate 0.001 and weight decay 0.01,
maximum 20 epochs, patience 3. B1 is retrained under this same budget; the
historical 5-epoch/patience-2 B1 is retained as a reproduction reference.

The protocol, data, cache, and training/evaluation source fingerprints are locked
on first training. Completed runs can resume; different identities cannot be
silently mixed. The documented `scripts/run_pipeline.py` entry point validates
every cached metric identity before a resumable stage; use it instead of calling
the training module directly. Use a new run name for an intentional protocol change.
Per-run logs and ragged predictions are saved locally in `results/runs/`;
checkpoints in `checkpoints/`. Compact summaries are versioned.

## Outputs

- `results/experiments/main/protocol.json`: frozen settings and provenance.
- `*_training.json`: loss, validation trajectory, selected epoch, time, parameters, memory.
- `*_test.json`: main metrics, coverage, threshold sensitivity, head diagnostics.
- `summary.json` and `model_comparison.csv`: seed means/SD, user-bootstrap CIs, paired differences.
- `controlled/design.json`: fixed cohort and history intervention.
- `controlled_summary.json`: ratios 50:50, 80:20, 90:10 across three history sampling seeds.
- `figures/`: comparisons and controlled-experiment curves.
- `docs/EXPERIMENT_PROTOCOL.md`: exact metric/experiment definitions.

Raw data, vectors, checkpoints and large predictions are intentionally ignored
by Git. Keep them locally or share them separately according to dataset terms.


## Follow-up experiments: attention scale, diversity and category grouping

The exploratory follow-up preserves the original `src/` and `main` experiment.
It screens attention multipliers 1/8/32 equally for B2 and B3, tests a
parameter-free attention-overlap penalty, and compares a frozen category-group
model with frozen mean pooling. Configurations are selected using validation
before evaluating the follow-up test results.

See [follow-up results](docs/FOLLOWUP_RESULTS.md) for findings and figures, and
[follow-up protocol](docs/FOLLOWUP_PROTOCOL.md) for the selection rule,
extra-metadata caveat, full-history versus last-50 metrics, and limitations.

From the repository root, using the existing environment:

~~~bash
.venv/bin/python -u -m experiments.followup select
.venv/bin/python -u -m experiments.followup evaluate
.venv/bin/python -u -m experiments.followup controlled
.venv/bin/python -m experiments.summarize_followup
.venv/bin/python -m experiments.verify_followup
~~~

Stages resume completed artifacts. Compact results are in
`results/experiments/followup/`; predictions and weights remain ignored.
The original test was already inspected, so this is exploratory follow-up work,
not an independent confirmatory experiment.

## Semantic clustering experiments

The final hard, soft, and masked-history clustering results are exploratory
extensions: the original main test results had already been inspected before
these models were designed. Each extension still fixes its configuration and
checkpoint using validation before its own test stage.

Run hard semantic clustering, its controlled-history evaluation, and summaries:

```bash
python -m experiments.clustering validate
python -m experiments.clustering evaluate
python -m experiments.controlled_clustering
python -m experiments.summarize_clustering
python -m experiments.summarize_clustering_test
```

Run the validation-fixed three-seed soft-clustering model:

```bash
python -m experiments.soft_clustering_final train
python -m experiments.soft_clustering_final evaluate
python -m experiments.soft_clustering_final summarize
python -m experiments.controlled_soft_clustering_final
```

Run the masked-history auxiliary-loss ablation:

```bash
python -m experiments.masked_history_final train
python -m experiments.masked_history_final evaluate
python -m experiments.masked_history_final summarize
```

Hard clustering uses frozen normalized 384-dimensional news vectors directly.
Soft clustering uses a zero-initialized 384-to-16-to-384 residual adapter; it
does not share the 64-dimensional projection used by B1--B4. Large checkpoints
and per-impression predictions remain ignored, while compact protocols,
metrics, and summaries are versioned under `results/experiments/`.
