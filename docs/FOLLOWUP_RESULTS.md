# Follow-up Results

## Attention scale

Attention multipliers 1, 8, and 32 did not produce a reliable improvement. The original scale was retained. Sharper logits did not prevent head collapse.

## Diversity penalty

An attention-overlap penalty of 0.1 did not provide a stable validation or test benefit. Penalizing overlap is weaker than giving heads an identifiable semantic assignment.

## Category diagnostic

| Method | NDCG@10 | Minority Recall@10 |
|---|---:|---:|
| B0 mean | 0.391786 | 0.537949 |
| Category groups | 0.391739 | 0.569800 |

Category grouping improves minority Recall@10 by 0.031851, with 95% CI [0.026248, 0.037221], while preserving NDCG@10. Meaningful groups can expose minority interests, motivating semantic clustering without category labels.

## Conclusion

Temperature tuning is not the project contribution. The useful finding is that learned attention lacks a specialization mechanism, whereas meaningful grouping recovers minority signal.

