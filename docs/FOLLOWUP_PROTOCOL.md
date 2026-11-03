# Follow-up Experiment Protocol

## Purpose

The follow-up investigates whether B3 fails because attention is too smooth or because no specialization signal exists.

## Fixed conditions

The original split, frozen news vectors, history length, projection, candidates, metrics, and validation/test separation are unchanged. Follow-up configurations are selected on validation before test evaluation. Because the original test results were already inspected, this stage is exploratory rather than a new independent confirmatory study.

## Experiments

1. Attention scale: multipliers 1, 8, and 32 for B2 and B3.
2. Attention-overlap penalty: discourage similar distributions without adding parameters.
3. Category grouping: category metadata as a diagnostic reference.
4. Controlled histories: 50:50, 80:20, and 90:10 histories.

## Selection rule

Select scale and penalty settings using validation NDCG@10, with minority recall and head diagnostics for interpretation. Test metrics are reported only after selection. Category grouping is not a fair content-only model because it uses extra metadata.

## Limitations

Temperature can sharpen assignments but cannot determine which concept each head should represent. Category grouping shows whether useful separation exists, but not whether a model can discover it without labels.

