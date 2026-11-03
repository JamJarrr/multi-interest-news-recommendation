# Differentiable Soft-Clustering Results

## Method

The extension uses K=4 soft spherical clusters, temperature 10, three refinement iterations, and a 384-to-16-to-384 residual adapter. The adapter starts as an identity-preserving residual and uses learning rate 1e-5. A naive 1e-3 run was unstable and was not selected.

## Three-seed test mean

| Method | NDCG@10 | Minority Recall@10 |
|---|---:|---:|
| Soft K=4 | 0.390537 (SD 0.001047) | 0.560759 (SD 0.001559) |

Soft K=4 minus B0:

- NDCG@10: -0.001249, 95% CI [-0.002781, 0.000294].
- Minority Recall@10: +0.022810, 95% CI [0.017855, 0.027788].

Soft K=4 minus hard K=4:

- NDCG@10: -0.002021, 95% CI [-0.003207, -0.000659].
- Minority Recall@10: +0.008259, 95% CI [0.004010, 0.012600].

Soft K=4 remains below category grouping in minority recall by 0.009042, with 95% CI [-0.014170, -0.003956].

## Controlled histories

Soft K=4 improves minority Recall@10 over B0 by +0.06984 at 80:20 and +0.04952 at 90:10. Soft-hard differences on controlled cohorts have confidence intervals that cross zero.

## Decision

Soft clustering is a recall-focused extension, not an unqualified replacement for hard clustering. It increases minority recall but gives up a small amount of overall ranking quality.

