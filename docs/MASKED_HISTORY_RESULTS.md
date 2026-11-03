# Masked-History Auxiliary Objective

## Method

One history item is masked and must be retrieved from four rolled in-batch negatives. The auxiliary loss weight is 0.1 and is added to the soft-clustering objective.

## Test result

| Method | NDCG@10 | Minority Recall@10 |
|---|---:|---:|
| Soft K=4 | 0.390537 | 0.560759 |
| Soft K=4 + masked history | 0.390582 | 0.561026 |

Masked training minus Soft K=4:

- NDCG@10: +0.000045, 95% CI [-0.000215, 0.000336].
- Minority Recall@10: +0.000268, 95% CI [-0.001045, 0.001664].

## Decision

The auxiliary objective showed a small validation improvement but no reliable test gain. It is retained as a negative ablation and is not used in the final model.

