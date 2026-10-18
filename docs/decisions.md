# Pre-registered decisions

Log every threshold, cut-order decision, or design choice fixed *before*
seeing results. Date each entry.

## 18-10-2026 — exp01: reproduce the one-sided vs two-sided claim on Olmo-3-7B SFT -> DPO

**Claim under test** (Stefan's notes, section 9.1 item 3): against English WildChat at matched prompt
length, one-sided feature distances to the DPO mix fall to chance, while two-sided distances (mix minus
Dolma) stay high.

**Setup, fixed before any features are extracted**
- Model: `allenai/Olmo-3-7B-Instruct-DPO`, bf16, revision hash recorded in `config.json`.
- ID mix: `allenai/Dolci-Instruct-DPO`, 2-message rows only, first user message. Family = `prompt_id`
  with regex `[-_]?\d+([-_]\d+)*$` stripped. Sampled proportionally to family share, seed 0:
  3,000 fit rows (`idfit`), disjoint 1,000 test rows (`idmix`).
- Dolma: 3,000 docs (>= 200 chars) from `dolma-v1_6-8B-sample`, drawn evenly across the 98 non-code
  shards (all shards except the 5 `stack-dedup` shards 98-102), seed 0. Two variants: `full` (first 512
  tokens) and `matched` (cut to a token length drawn from `idfit`'s content-token length distribution).
  Raw text, no chat template.
- OOD pools (500 each, seed 0): `legal` = `jonathanli/legal-advice-reddit` title + body; `medical` =
  `lavita/ChatDoctor-HealthCareMagic-100k` `input` field.
- Hardest covered pool: `wildchat_en_test` = 500 English first user turns from `allenai/WildChat-1M`,
  deduplicated, with any row whose normalised text (lowercase, whitespace-collapsed) exactly matches a
  Dolci-DPO first user turn, or shares a word 13-gram with one (n = min(13, row length)), removed.
  The membership check must be shown to fire on a known positive (Dolci rows themselves) before use.
- Features: chat template with `add_generation_prompt=True`, max 1024 tokens, left truncation, content
  tokens only via `offset_mapping`. Residual stream `hidden_states[l]` for l in 0, 4, ..., 32, poolings
  `cmean`, `cmax`, `clast`, all cached in one pass.
- Detectors: one-sided `maha` (Ledoit-Wolf) and `knn` (k=10, per-dim std scaling then L2 norm) fit on
  `idfit`; two-sided `dmaha_{full,matched}`, `dknn_{full,matched}` = score vs mix minus score vs Dolma.
  Baselines: content-token length, and tf-idf logistic regression mix (0) vs Dolma-full (1), C=0.1.
- Length-stratified AUROC: bands on content-token length [1,16), [16,32), [32,64), [64,128), [128,256),
  [256,512), [512,1025). AUROC within each band, pooled weighted by OOD x reference pairs. Reported only
  if >= 3 bands hold >= 10 rows of each class and the usable bands cover >= 30% of the OOD pool;
  otherwise "unmeasurable". The length-only detector's stratified AUROC is reported alongside.
- 95% CIs from 1,000 bootstrap resamples over rows.

**Decision cell.** `cmean`, layer 16, Dolma `full`. Every other layer, pooling and Dolma variant is
exploratory and cannot change the verdict.

**Bar 1, positive control.** One-sided `maha` and `knn` score raw AUROC >= 0.95 for both `legal` vs
`idmix` and `medical` vs `idmix`. If any of the four misses, the pipeline is treated as broken: stop,
debug, and do not evaluate bar 2.

**Bar 2, main claim.** Length-stratified AUROC vs `wildchat_en_test`, on both `legal` and `medical`:
one-sided `maha` and `knn` all <= 0.60, AND two-sided `dmaha_full` and `dknn_full` all >= 0.85.
Reproduced only if all eight numbers clear. If the stratified AUROC is unmeasurable for a pool, bar 2 is
ambiguous for that pool, not failed.

**Reported, no bar.** tf-idf and length baselines on the same comparisons (Stefan's notes: tf-idf about
0.90).

**Seeds.** One data-sampling seed (0). No extra seeds planned; this is a reproduction of a reported
number, not a new effect.

## 18-10-2026 — exp02: exp01 with a better-conditioned Mahalanobis

Written after exp01's results, before any exp02 number is computed. Same features as exp01 (no
re-extraction), same pools, same decision cell (cmean L16, Dolma full), same bars and thresholds.

**One change.** Every Mahalanobis (one-sided `maha` on the mix; the Dolma term inside `dmaha_full` and
`dmaha_matched`) is computed in a PCA subspace fitted on its own reference set, k = 256 components
(n_fit / k ~ 12), with Ledoit-Wolf on the projected features. Rationale: exp01's full-rank fit at
d = 4096, n = 3000 chose shrinkage 0.019 and scored held-out mix rows ~5x further than fit rows. kNN is
unchanged, so its exp01 values carry over.

**Bar 2 is evaluated whatever bar 1 does.** This departs from exp01's rule and is the user's instruction,
given after exp01's results. Bar 1 is still reported against the same 0.95 threshold.

**Exploratory, cannot change the verdict.** On the decision cell only: PCA k in {32, 64, 128, 512, 1024}
and full-rank Mahalanobis with fixed shrinkage alpha in {0.1, 0.5}.

## 18-10-2026 — exp03: register vs coverage with covered-domain Reddit pools

**Question.** Do the detectors flag uncovered legal questions above Reddit posts on topics the DPO mix
does cover, when both come from Reddit? If yes, they read coverage; if covered subreddits score like
legal, they read register (Reddit prose looks pretraining-like).

**Pools.** From `sentence-transformers/reddit-title-body`, file `reddit_title_text_2019.jsonl.gz`
(same years as the legal pool), title + "\n\n" + body, body non-empty and not [removed]/[deleted],
deduplicated, 500 per subreddit, seed 0. Covered: relationship_advice, Fitness, Cooking, writing,
relationships, learnprogramming, askscience (Stefan's list). Uncovered: legaladvice from the same file
(`legal_rtb`), so legal and covered pools share source and formatting. The exp01 `legal` pool is also
scored. Each pool goes through the exp01 Dolci membership check; members are dropped and counted.

**Fixed from exp01.** Model, revision, features (cmean L16 decision cell), idfit, Dolma, full-rank
Ledoit-Wolf Mahalanobis, kNN, length bands and guard, 1,000-sample bootstrap. No refitting.

**Primary metric.** Length-stratified AUROC with `legal_rtb` as the OOD side and each covered subreddit
as the reference, per detector. Summary = median over the 7 subreddits.

**Bar, per detector, judged on dmaha_full and dknn_full.**
- median >= 0.75: separates coverage from register.
- median <= 0.60: reads register, not coverage.
- in between: ambiguous.
One-sided maha/knn, tf-idf and length are reported on the same metric with no bar. Also reported: each
pool's stratified AUROC vs `wildchat_en_test` and raw AUROC vs `idmix`.

**Seeds.** One data seed (0). No extra seeds planned.

## 18-10-2026 — exp04: exp03 re-run with two logic fixes

Written after exp03 and an audit of it, before any exp04 number. The user asked for a re-run looking for
flaws. The audit found two, so this is a fix-and-rerun, not a rerun hoping for a cleaner number.

**Flaw 1, coarse length control.** exp01-exp03 bands span a factor of 2. Within pools the two-sided
scores correlate +0.3 to +0.8 (Spearman) with length, and within bands the pools still differ by 10-20%
in median length (length-only stratified AUROC 0.40-0.65 in exp03). Fix: bands log-spaced by a factor of
2^(1/4) (~1.19) from 1 to 1024 tokens, same guard (>= 10 per class per band, >= 3 bands, >= 30% of the
OOD pool covered). Check that the fix works: length-only stratified AUROC must fall within 0.45-0.55
for every primary comparison; if not, report that comparison as unbalanced.

**Flaw 2, assumed coverage labels.** exp03 treated the seven subreddits as covered without measuring it.
By nearest-Dolci tf-idf cosine, r/relationships (0.214) and r/relationship_advice (0.223) are no better
covered than r/legaladvice (0.227). Fix: measure coverage per row as the cosine similarity to its nearest
Dolci-Instruct-DPO first user turn (233,642 unique prompts), using `sentence-transformers/all-MiniLM-L6-v2`
embeddings (primary, independent of OLMo) and tf-idf (secondary).

**Fresh samples.** Posts drawn uniformly across the whole 2019 reddit-title-body file (exp03 read only its
start), 500 per pool, seeds 1 and 2; exp03's seed-0 pools are re-scored too. Same eight pools, same
filters, same Dolci membership check, same exp01 detectors and features recipe (cmean L16, Dolma full).

**Q1, exp03's question with fix 1.** Same metric and bar as exp03 (median over the 7 subreddits of
length-stratified AUROC, legal_rtb vs subreddit; >= 0.75 coverage, <= 0.60 register), per seed, on
dmaha_full and dknn_full. Reported for seeds 0, 1, 2. Flagged as weak evidence either way because flaw 2
means the subreddit labels are not reliable.

**Q2, the corrected question.** Across all 4,000 Reddit rows of a seed, partial Spearman correlation
between detector score and MiniLM coverage, controlling for log length. Bar per detector
(dmaha_full, dknn_full): rho <= -0.30 on all three seeds means the score tracks measured coverage;
|rho| < 0.10 on all three means it does not; anything else is ambiguous. tf-idf detector, one-sided
detectors and length reported on the same statistic with no bar. Pool-level: Spearman across the 8 pools
between median coverage and median detector score, reported, no bar (n = 8).

**Exploratory.** exp01's bar-2 numbers recomputed with the fine bands; cannot change exp01's verdict.

## 18-10-2026 — exp06: density ratio on planted needles

Written before any exp06 data, feature or score exists. Runs alongside exp05 (another session) and shares
no files with it.

**Question.** Does a density-ratio score, log p_deploy(x) / p_mix(x) estimated by a classifier between a
deployment pool and the DPO mix, surface rare uncovered prompts planted in real traffic, without using
Dolma? And does it leave rare covered-but-unusual prompts alone?

**Deployment pool D.** 10,000 WildChat-1M first user turns, all languages, shards 1, 2, 3 (disjoint from
exp01's shards 0, 5, 10), deduplicated, Dolci-DPO members removed by the exp01 membership check, seed 0.

**Needles.** Banks of 300 rows per type, membership-checked against Dolci-DPO, seed 0:
gap needles `legal` (legal-advice-reddit test split, title + body) and `medical` (HealthCareMagic, excluding
exp01's rows); control needles `aya` (CohereForAI/aya_dataset train inputs, all languages; Dolci covers Aya)
and `magicoder` (Magicoder-OSS-Instruct problems; Dolci covers code). For prevalence p in {0.1%, 0.3%, 1%}
and needle seed s in {0, 1, 2}, n = 10 / 30 / 100 rows of each type are planted in D together; seeds use
disjoint 100-row slices of each bank.

**ID side.** exp01 `idfit` + `idmix` (4,000 Dolci-DPO first turns). Features from the exp01 recipe
(Olmo-3-7B-Instruct-DPO rev b33130b7, content tokens, layer 16).

**Arms.**
- `dr_cmean` (primary): L2 logistic regression D (1) vs mix (0), balanced class weights, standardised
  cmean L16 features, C = 0.1, 5-fold cross-fitting over D so no row is scored by a model trained on it.
- `dr_clast`: the same on clast L16 (nearly length-free pooling).
- `dr_tfidf`: the same on tf-idf (sublinear, min_df 2, 200k features, fitted on mix + D), C = 1.0.
- Baselines: one-sided Mahalanobis with fixed shrinkage 0.5 (exp02 showed it holds up better on held-out
  rows than Ledoit-Wolf), one-sided kNN, two-sided dknn_full (exp01 Dolma), length, random.

**Ranking.** Length-controlled = percentile of the score within 20 equal-count content-token bins of the
planted pool. Raw rankings also reported.

**Metric.** precision@K, K in {25, 50, 100, 200}. uplift@100 = precision@100 / prevalence, computed for the
gap group (legal + medical) and the control group (aya + magicoder) separately.

**Bars (length-controlled `dr_cmean`, mean over the 3 needle seeds, judged at 0.3% and 1%).**
- Q1 gaps: gap uplift@100 >= 5 at both prevalences = upsamples gaps; <= 2 at both = does not; else ambiguous.
- Q2 controls: control uplift@100 <= 2 at both prevalences = passes; otherwise fails (it flags covered but
  unusual text).
- Q3 words: gap uplift of `dr_cmean` / gap uplift of `dr_tfidf` >= 1.5 at both prevalences = activations
  add something; <= 1.0 at both = no gain; else ambiguous.
0.1% is reported with no bar.

**Iterative loop (secondary, no bar).** At 1%, seeds 0-2, arms `dr_cmean` and `dr_tfidf`: 5 rounds; each
round marks the top 100 remaining rows of the length-controlled ranking as reviewed, moves them to the
ID side and refits. Cumulative recall per needle type is reported against random review (expected recall
= rows reviewed / pool size).

**Sanity checks.** Random ranking uplift near 1; length-only uplift reported; median length of the top 100
vs the pool on the length-controlled ranking.

## 18-10-2026 — exp07: exp06 with an English-only deployment pool

Written after exp06's result, before any exp07 data exists. exp06's verdict named this change: its
density ratio spent the ranking on Chinese WildChat (about a third of the pool, under 3% of the mix).

**One change.** The deployment pool is 10,000 WildChat-1M first turns with `language == "English"`, shards
1-3, deduplicated, Dolci-DPO members removed, seed 0. Everything else is exp06 unchanged: the same needle
banks and their cached features (legal, medical, aya, magicoder; seeds 0-2 use the same 100-row slices),
the same mix (exp01 idfit + idmix), arms, cross-fitting, C values, 20-bin length control, metrics, bars
(Q1 gap uplift@100 >= 5 / <= 2 at 0.3% and 1%; Q2 control uplift <= 2; Q3 dense / tf-idf >= 1.5 / <= 1.0)
and the 5 x 100 iterative loop. Aya needles are mostly non-English, so in this pool they are a language
shift that the mix covers; they stay a control.

**Reported, no bar.** The 15 top-ranked rows of the length-controlled dr_cmean ranking, to name what the
density ratio ranks first once language is removed.
