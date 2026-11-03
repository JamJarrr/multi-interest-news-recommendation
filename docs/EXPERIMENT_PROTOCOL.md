# Experiment protocol

The research question is whether multiple user representations preserve weaker
observed interests better than a single representation. All conclusions concern
ranking logged clicks in MIND-small. They do not establish satisfaction, changed
user interests, online recommendation effects, or a solution to filter bubbles.

## Data and reproducibility

The original train is split at 2019-11-13 20:36:26. Training timestamps are
strictly earlier than validation. Official dev (2019-11-15) is held out for final
testing. Preprocessing never adds later clicks to earlier histories.
All news text, including test candidates, can be encoded by the same frozen
pretrained encoder without using their click labels to train it.

A combined catalog contains 65,238 news items and 18 categories. The 3,415
missing abstracts fall back to title-only encoding. The tokenizer's normal
MiniLM truncation applies; the encoder is frozen. Categories are not model
features. The text cache records the model revision, content digest and vector
digest. Compressed CSV hashes can vary with gzip timestamps, so split identities
use the uncompressed bytes.

The downloaded archives differ from the historical ZIP hashes. ZIP CRC,
schema, all submitted audit statistics and baseline metric reproduction were
checked. These checks support consistency but are not a proof of raw-file
identity with unavailable historical files. The actual TSV hashes are retained.

## Training and capacity control

B1, B2, B3 K=2, B3 K=4 and B4 use the same 384-to-64 affine projection for history
and candidates, and normalize the pooled interest and candidate vectors before
cosine matching. B2/B3 use query-dot-projected-history attention divided by
sqrt(64). Attention masking gives zero probability to padding. An empty history
uses the learned projection bias as a global fallback, matching the original B1.
B0 instead gives equal zero scores for an empty history. Stable tie ordering
matches the original evaluator; this cold-start behavior is a limitation.

B3 uses the maximum of the candidate's K cosine matches, not a concatenation
or average of heads. B4 remains a single-interest model and adds
`tanh(W z) dot v` to its one-query attention score, with width 3 and no layer
bias. Every extra parameter affects attention; no unused parameters are added.
B4's 24,899 parameters nearly match B3 K=4's 24,896. Parameter matching does not
imply identical inductive bias or identical computational cost.

For a training seed, every model uses the same fixed four negatives drawn
within the same impression for each positive. Negatives are sampled with
replacement only when fewer than four are available. Multi-positive impressions
produce one training group per positive; no clicked item is used as a negative.
Validation and test score the complete candidate list. Sampling is fixed across
epochs, matching the original B1. No-positive/no-negative impressions would be
excluded from training; current training produces 187,856 positive groups.

All trainable models use batch size 1024, AdamW (lr 0.001, weight decay 0.01),
at most 20 epochs and validation NDCG@10 patience 3. Checkpoints are selected
separately for each seed on validation only. Both K values and all seeds are
reported. The historical B1's 5-epoch/patience-2 setup is reproduced separately.
No new hyperparameter search is conducted on test results. PyTorch deterministic
operations and fixed seeds are used; differences across hardware/library
versions and floating-point ties may remain.

## Natural-history metrics

NDCG@5, NDCG@10 and Recall@10 are impression averages. For Minority Recall@10,
a category must occur at least twice and account for at most 20% of the FULL
provided pre-impression history. This exactly preserves the historical
evaluation definition. Models see only the last 50 items, so a historical
minority category can be missing from the visible model window. This distinction
must be disclosed and is not changed silently between models.

Minority hits and positive targets are summed inside each user, then the user's
ratio is averaged across users with at least one target. Thus active users do
not receive extra weight. Thresholds of 10% and 30% are sensitivity analyses;
the eligible populations change with the threshold, so these are not a fixed
cohort causal comparison. A threshold with no eligible targets returns null,
not a misleading zero.

Coverage includes target users, target impressions, clicked targets and their
fractions of the full sample. Top-10 minority-category share counts all
recommended minority items, including unclicked ones; candidate-category share
provides context about available inventory. These exposure statistics help
detect a model that recommends minority categories indiscriminately.

The three training-seed mean and sample SD are reported. The 95% percentile
bootstrap resamples USERS, preserving their grouped impressions. For overall
metrics, sums and counts preserve impression averaging after cluster resampling.
For Minority Recall the eligible user's ratio is the unit of averaging.
Predictions are averaged over fitted seeds before bootstrap; intervals therefore
describe user sampling uncertainty conditional on these models, while seed SD
describes training variability separately. Paired model differences use the same
resampled users. These multiple comparisons are exploratory and are not multiplicity-adjusted. Overlapping marginal CIs are not used as a significance test.

## Controlled histories

Use only the last 50 original history items. Select the two most frequent
categories (ties resolved by category name). Label the more frequent as dominant
and the other as the fixed target. Eligibility requires at least 18 dominant
and 10 target occurrences BEFORE inspecting candidate click labels. This
guarantees that all three histories can be drawn from the same cohort.

Construct histories of exactly 20 items without replacement: 10+10, 16+4 and
18+2. Keep the same user/impression, candidates, click labels and target category
at every ratio. Keep selected items in original chronological order, never add
future or external news. A history-derived deterministic seed gives identical
draws across all models, and repeated identical histories use the same draw.
Use sampling seeds 11, 29 and 47. Models remain frozen.

The outcome is FIXED TARGET-CATEGORY Recall@10, user macro. At 50:50 neither
category satisfies the natural <=20% minority definition, so calling this
curve "natural Minority Recall@10" would be incorrect. The fixed target may
also not have been a natural minority before intervention. Report the restricted
cohort and its coverage; this test does not represent all MIND users.

Controlled summaries average over history sampling seeds within each training
seed, report training-seed SD, and use user-bootstrap CIs conditional on the
fitted models and these sampled histories. Seed variation is not folded into
the user CI.

## Attention diagnostics and timing

For nonempty histories, report head winner frequencies over all real candidates,
mean pairwise attention-weight cosine, mean pairwise interest-vector cosine and
attention entropy. Identical or nearly identical heads and strongly concentrated
winner frequencies are evidence of redundancy, not proof that meaningful
semantic interests have been discovered. First-head tie-breaking is disclosed.

Time covers model training plus per-epoch validation, excluding data loading,
negative-sample construction and text encoding. Peak CUDA allocated memory is
process allocation, not total device utilization. B0 has no trainable parameters
and no training time. Evaluation time is recorded separately.

## Scope for the course submission

This implementation covers B0–B4, common three-seed training, natural metrics,
coverage, paired bootstrap, minority-threshold sensitivity, controlled imbalance,
head diagnostics, figures and reproducible commands. Additional regularization,
hard-negative mining and fine-tuning are optional extensions and are not
included in the primary results. The final course report, course handout checks
and AI Use Statement still need to be prepared by the group.


