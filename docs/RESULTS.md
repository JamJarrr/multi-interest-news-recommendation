# Main Experiment Results

## Research question

When user histories are strongly imbalanced, do multiple user-interest vectors preserve minority preferences better than a single representation?

## Shared protocol

All models use the same frozen MiniLM news vectors, chronological train/validation split, official development set as test, last-50 history, full evaluation candidate sets, four within-impression negatives, and validation-only early stopping. Learned variants use seeds 42, 2026, and 3407. NDCG@10 is the primary overall metric; NDCG@5 and Recall@10 are complementary metrics, and user-macro Minority Recall@10 is the targeted minority-interest metric.

## Test summary

| Model | NDCG@10 | Minority Recall@10 |
|---|---:|---:|
| B0 mean | 0.391786 | 0.537949 |
| B1 projected mean | 0.386853 | 0.544669 |
| B2 single attention | 0.386777 | 0.544610 |
| B3, K=2 | 0.386876 | 0.544449 |
| B3, K=4 | 0.387030 | 0.544724 |
| B4 capacity-matched single interest | 0.386046 | 0.545238 |

B3 K=4 minus B2 changes NDCG@10 by +0.000253 (95% CI [+0.000149, +0.000357]) and minority Recall@10 by +0.000114 (95% CI [-0.000313, +0.000541]). The minority-recall interval includes zero.

## Diagnosis

The mean pairwise cosine similarity between B3 attention heads is approximately 0.9998. The four heads therefore produce almost the same user vector. B4 also performs similarly despite matching B3 K=4 within three trainable parameters. The original learned multi-interest formulation does not demonstrate distinct interest discovery.

## Interpretation

The negative result does not reject the project question. It shows that multiple vectors are not sufficient by themselves: the model needs a mechanism that makes the vectors specialize. This motivated the follow-up temperature, category, and clustering experiments.

