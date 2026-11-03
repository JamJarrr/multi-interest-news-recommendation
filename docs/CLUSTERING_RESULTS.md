# Semantic Hard-Clustering Results

## Method

For each user, normalized history embeddings are partitioned with spherical K-means using K=4. Each cluster mean becomes one interest vector, and a candidate is scored by its maximum similarity to the four vectors. Category labels are not used.

## Test results

| Method | NDCG@10 | Minority Recall@10 |
|---|---:|---:|
| B0 mean | 0.391786 | 0.537949 |
| Random K=4 groups | 0.389353 | 0.543408 |
| Hard semantic K=4 | 0.392557 | 0.552499 |
| Category groups | 0.391739 | 0.569800 |

Hard K=4 minus B0:

- NDCG@10: +0.000772, 95% CI [-0.000570, 0.002262].
- Minority Recall@10: +0.014550, 95% CI [0.009769, 0.019376].

Hard K=4 minus random grouping:

- NDCG@10: +0.003204, 95% CI [0.002057, 0.004412].
- Minority Recall@10: +0.009091, 95% CI [0.005208, 0.012932].

## Mechanism diagnostics

- Mean active slots per user: 3.87.
- Pairwise centroid cosine: 0.111 for semantic clusters versus 0.303 for random groups.
- Category purity: 0.547 for semantic clusters versus 0.446 for random groups.

## Controlled histories

| History ratio | Difference from B0 | 95% CI |
|---|---:|---:|
| 50:50 | -0.00514 | crosses zero |
| 80:20 | +0.07661 | [0.05915, 0.09499] |
| 90:10 | +0.04974 | [0.03103, 0.06970] |

## Decision

Hard semantic K=4 is the recommended primary model. It improves minority retrieval, beats random grouping, and does not show a reliable NDCG loss.

