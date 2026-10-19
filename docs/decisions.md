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

## 18-10-2026 — exp05: do activation detectors still flag legal and medical once rewritten in the mix's style?

Written before any rewrite, feature or score exists. Proposed by a collaborator as the decisive test of
whether the activation detectors add anything beyond words: exp01 and exp03 found tf-idf matching them.

**Question.** Rewrite legal and medical prompts so their wording matches the DPO mix while keeping the
question. If the activation detectors still separate them from covered prompts rewritten the same way,
and do so clearly better than tf-idf, the activations carry topic/coverage signal beyond wording.

**Pools, all seed 0, from exp01.** `legal` (500), `medical` (500), `wildchat_en_test` (500, covered
reference), and 500 rows of `idmix` drawn with seed 0 (rewrite-artifact control). Each row is rewritten
once, giving `legal_rw`, `medical_rw`, `wildchat_rw`, `idmix_rw`.

**Rewrite.** `openai/gpt-4.1` via OpenRouter (the Dolci WildChat family was regenerated with GPT-4.1), no
temperature or max_tokens set. One fixed prompt for every row: 8 example prompts drawn with seed 0 from
English `idfit` rows of 10-120 words (never from idmix or test pools), then an instruction to rewrite
the message so it reads like the examples in phrasing, tone, structure and length, keeping the same
request and every fact needed to answer it, without answering it or adding facts. Failed calls are
recorded as failed rows and counted, not silently dropped.

**Content check.** `anthropic/claude-sonnet-4.6` (other provider) sees original and rewrite, not the
pool name, and answers YES/NO: does the rewrite ask for the same thing and keep the facts needed to
answer it? I read 20 judge decisions per run and record my agreement (model-read, not human).

**Features and detectors, fixed from exp01.** Same model and revision, same extraction recipe; rewritten
pools are extracted fresh, everything else reuses exp01 features. Decision cell cmean L16, Dolma full.
Detectors fitted exactly as exp01 (idfit; Dolma full): `dmaha_full`, `dknn_full`, tf-idf mix-vs-Dolma.

**Length control.** exp04's fine bands: log-spaced by 2^(1/4) from 1 to 1024 content tokens, guard
>= 10 per class per band, >= 3 bands, >= 30% of the OOD pool. A comparison whose length-only stratified
AUROC falls outside 0.45-0.55 is "unbalanced".

**Primary comparisons.** Length-stratified AUROC, `legal_rw` vs `wildchat_rw` and `medical_rw` vs
`wildchat_rw`. For each activation detector d and pool p, delta = AUROC_d - AUROC_tfidf on the same
comparison, with a paired bootstrap CI (1,000 resamples, 98.75% = 95% Bonferroni over 2 detectors x
2 pools).

**Validity gates, checked before the verdict.**
- G1 content: judge YES rate >= 90% in each of the four rewritten pools.
- G2 wording moved: tf-idf stratified AUROC on p_rw vs wildchat_rw at least 0.10 below tf-idf on p vs
  wildchat_en_test (originals, same fine bands), for both p.
- G3 balance: both primary comparisons balanced.

**Verdict.**
- Activations add beyond words: for at least one detector d, on both legal and medical, AUROC_d >= 0.75
  and delta >= 0.10 with the CI lower bound > 0. Requires G1 and G3.
- Words only, ends the activation-detector line: for both detectors on both pools, AUROC_d <= 0.65 or
  the delta CI upper bound < 0.10. Requires G1, G2 and G3; without G2 it is inconclusive (the rewrite did
  not remove the wording cue).
- Anything else, or a failed gate: ambiguous / inconclusive, named by which condition failed.

**Reported, no bar.** One-sided maha/knn, dmaha_matched, dknn_matched and length on the same comparisons;
originals vs wildchat_en_test with fine bands; `idmix_rw` vs `idmix` (how much rewriting alone moves a
covered prompt); p_rw vs the original wildchat_en_test; other layers and poolings.

**Seeds.** One data seed (0), one rewrite per row. No extra seeds planned.

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

## 18-10-2026 — exp08: controls and baselines the exp01 decision cell never had

Written before any exp08 pool, feature or score exists. Runs the negative controls (§7.4) and baselines
(§6.4, §6.2) from Stefan's summary that exp01-exp05 skipped, and builds the deployment pools that gap
discovery needs. All detector questions use exp01's decision cell (cmean L16, Dolma full, idfit fit,
full-rank Ledoit-Wolf, kNN k=10) and exp01's coarse length bands and guard, so the numbers validate
the exp01 values they are controls for. Fine-band (exp04) values and length-only AUROC are reported
beside each. 95% bootstrap CIs, 1,000 resamples. Data seed 0 unless stated; one seed, no extra seeds.

**New pools, 500 each, membership-checked against all Dolci-DPO first user turns (exp01 MembershipIndex).**
- `dolma_b`: 3,000 Dolma v1.6 docs, same shard rule as exp01, seed 1, any doc whose normalised text
  matches an exp01 Dolma doc dropped and replaced.
- Membership halves (§2): `aya_train` / `code_train` / `jailbreak_train` = literal Dolci-DPO 2-message
  rows of `tulu_v3.9_aya_100k`, `evol_codealpaca_heval_decontaminated`,
  `tulu_v3.9_wildjailbreak_decontaminated_50k`, excluding idfit and idmix rows. `aya_test` /
  `code_test` / `jailbreak_test` = rows of the upstream not in Dolci-DPO: `CohereForAI/aya_dataset`
  train `inputs`; `theblackcat102/evol-codealpaca-v1` `instruction` (the doc's
  `ai2-adapt-dev/evol_codealpaca_heval_decontaminated` no longer exists); `allenai/wildjailbreak` train,
  `adversarial` if non-empty else `vanilla`. FLAN is left out: the doc reports its test half is
  near-duplicate contaminated.
- Same domain, other source (§2 gold): `magicoder` = `ise-uiuc/Magicoder-OSS-Instruct-75K` `problem`;
  `bactrian` = `MBZUAI/Bactrian-X` instruction (+ "\n\n" + input if non-empty), drawn uniformly over its
  non-English languages.
- Deployment: `wildchat_deploy` = 2,000 WildChat-1M first user turns, all languages, shard 1 (exp01 used
  0/5/10), deduplicated; `lmsys` = 2,000 LMSYS-Chat-1M first user turns, all languages, shard 0,
  deduplicated. Not membership-filtered (deployment as it is); each row's membership flag is recorded.

**Q1, Dolma seed-pair null.** `null_dmaha = maha(x; Dolma A) - maha(x; Dolma B)`, `null_dknn` likewise,
A = exp01 Dolma full, B = `dolma_b` full. Stratified AUROC on legal vs wildchat_en_test and medical vs
wildchat_en_test. Resolvable if all four in [0.40, 0.60]; not resolvable if any outside [0.30, 0.70];
else ambiguous. Reported, no bar: dmaha_full and dknn_full recomputed with Dolma B.

**Q2, membership.** Stratified AUROC `{family}_train` (OOD side) vs `{family}_test`, for dmaha_full and
dknn_full, three families. Invisible if all six in [0.40, 0.60]; visible if any >= 0.70 or <= 0.30; else
ambiguous. One-sided maha/knn and tf-idf reported, no bar.

**Q3, same domain, other source.** Stratified AUROC `magicoder` vs `code_train` and `bactrian` vs
`aya_train`, dmaha_full and dknn_full. Reads domain if all four <= 0.65; reads source if any >= 0.80;
else ambiguous. Also reported vs idmix and vs the `_test` halves.

**Q4, dep2dolma.** Direction that never sees the mix: standardise on the pooled deployment-reference and
Dolma features, project onto (mu_Dolma - mu_deploy) / sd, deployment reference = `wildchat_deploy`.
Stratified AUROC legal vs wildchat_en_test and medical vs wildchat_en_test at cmean L16. Reproduces
Stefan's register caveat if both >= 0.85; does not if both <= 0.65; else ambiguous. Reported, no bar:
the same at L32, and the mix-side directions `meandiff` (mix to Dolma), `lda` (shrunk pooled
covariance), `logit` (mix 0 vs Dolma 1, C=0.1, standardised) at L16 and L32.

**Q5, supervised ceilings on exp05's rewritten pools.** 5-fold cross-validated logistic probes (seed 0,
out-of-fold scores): activation probe on standardised cmean L16 features (C=0.1) and tf-idf probe (same
vectoriser settings as the exp01 tf-idf, C=0.1). Comparisons legal_rw vs wildchat_rw and medical_rw vs
wildchat_rw, fine bands as in exp05. Activations add beyond words if activation probe minus tf-idf probe
>= 0.03 with paired-bootstrap CI lower bound > 0 on both; words suffice if the difference CI upper bound
< 0.03 on both; else ambiguous. If the tf-idf probe scores >= 0.98 on either, the verdict is "ceiling",
not either of the above. Same probes on the original legal/medical vs wildchat_en_test reported, no bar.

**Perplexity baseline (§6.4), no bar.** Mean per-token NLL of content tokens 2..n under the DPO model in
the chat template (same rendering as features). Raw and stratified AUROC on every Q comparison and on
exp01's legal/medical vs idmix and vs wildchat_en_test.

**Deployment pools, no bar.** Features and every detector score written; raw AUROC vs idmix and Spearman
with length reported. Built for gap discovery; no question asked of them here.

## 18-10-2026 — exp09: iterative gap discovery in real deployment traffic

Written before any exp09 data, score, cluster or name exists. This is Stefan's item 2: realistic
post-training mix vs realistic deployment traffic, find distinct nameable gaps, with an iterative process
so the same gap is not flagged again.

**Deployment pool.** 20,000 first user turns, all languages: exp06's 10,000 WildChat-1M rows (shards 1-3,
Dolci members removed, features cached) plus 10,000 LMSYS-Chat-1M first user turns (deduplicated, Dolci
members removed by the exp01 membership check, seed 0). ID side: exp01 idfit + idmix (4,000).

**Scoring arms, run as two separate loops.**
- `dr`: cross-fitted density ratio (L2 logistic, pool 1 vs ID 0, standardised cmean L16, C = 0.1, 5 folds).
- `dknn`: two-sided kNN (kNN distance to the ID set minus kNN distance to exp01's Dolma-full, k = 10,
  mix-std scaling, cmean L16).
Both ranked by length-controlled percentile (20 equal-count content-token bins of the remaining pool).

**One round.** Take the top 500 remaining rows. Embed them with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and cluster with k-means, k = 10, seed 0.
For each cluster, show 25 random members (first 300 characters each) to `anthropic/claude-sonnet-4.6` via
OpenRouter, blind: no hypotheses, no mention of OOD or training data, asked only for a short name for what
the prompts share, a one-line description, and whether they share something coherent (COHERENT /
INCOHERENT). Coverage of each row = cosine of its multilingual embedding to the nearest of the 233,642
unique Dolci-DPO first user turns.

**Confirmed gap.** A cluster is confirmed if (a) the namer calls it COHERENT, (b) it has >= 15 members, and
(c) its median coverage is below the 25th percentile of row coverage in the full 20,000-row pool.
**Same gap.** A confirmed cluster repeats an earlier confirmed gap (any round, same arm) if the cosine
between the two cluster centroids (multilingual embeddings) is >= 0.85; otherwise it is new.
**Known-OOD update.** All members of confirmed clusters (new or repeat) leave the pool and join the ID side:
for `dr` they become label-0 training rows; for `dknn` they join the ID reference set. Then re-rank.
5 rounds per arm.

**Random control.** Each round, per arm, 500 rows drawn uniformly from that arm's remaining pool go through
the same embedding, clustering, naming and coverage check (they are not moved). Distinct confirmed gaps
are counted the same way.

**Bar, per arm, after 5 rounds.**
- useful discovery: >= 5 distinct confirmed gaps AND >= 2x the random control's distinct confirmed count;
- no better than random: distinct confirmed gaps <= random control's count + 1;
- otherwise ambiguous.

**Reported, no bar.** Every cluster's name, description, size, median coverage, median length relative to
the pool, round, new/repeat, and 5 example prompts; confirmed gaps per round; the overlap between the two
arms' gaps (centroid cosine >= 0.85); share of each round's top 500 that is WildChat vs LMSYS.

**Seeds.** One pool seed (0), one clustering seed (0), one naming call per cluster. No extra seeds planned.

**exp09 amendment, 18-10-2026, still before any exp09 data.** A check of the embedding model showed it
is cross-lingual (cosine 0.92 between an English and a Chinese sentence with the same meaning). So
coverage measures semantic coverage only: a cluster that differs from the mix mainly by language will
usually fail criterion (c). The confirmation rule is unchanged. Each cluster's dominant language share
(WildChat/LMSYS language field) is reported with it, so language gaps stay visible.

## 19-10-2026 — exp10: do activation detectors still flag legal and medical once tf-idf's words are banned?

Written before any exp10 rewrite, feature or score exists. The collaborator's decisive test, rerun in the
form exp05's verdict asked for: exp05's style rewrite kept the topic words (legal tf-idf rose 0.951 to
0.966), so it could not separate topic from wording. Here the rewriter is forbidden the words a bag of
words uses to tell legal and medical from WildChat, and a blind judge checks that the topic survived.
Also carries Stefan's idea 4 (template-wrapped Dolma), which shares the feature pass.

**Ban list, fit on data disjoint from every evaluation row.** Supervised tf-idf logistic (exp01 vectoriser
settings, C = 1.0), two models: exp06 `needles_legal` (300) vs 600 English WildChat rows drawn with seed 0
from exp07 `deploy`, and exp06 `needles_medical` (300) vs the same 600. From each model take the top 100
positive and top 100 negative coefficients; keep alphabetic terms of >= 3 characters not in sklearn's
English stop words; the ban list is the union. The same list is given for every row of every pool.

**Rewrite.** All 500 rows of exp01 `legal`, `medical` and `wildchat_en_test`, one rewrite each, by
`openai/gpt-4.1` via OpenRouter with exp05's template (same 8 idfit style examples), plus one added
instruction: do not use any of the listed words or any form of them, and use plain everyday words
instead, keeping the situation and the request. No temperature or max_tokens. Pools `legal_lx`,
`medical_lx`, `wildchat_lx`. Failed calls are recorded as failed rows. Leak rate = share of rewrites that
contain a banned term as a whole lowercased token; reported per pool.

**Topic judge, blind.** `anthropic/claude-sonnet-4.6` sees one message only (no pool, no original) and
answers LAWYER, DOCTOR or NEITHER: whose expertise the message mainly needs. Run on all 1,500 originals and
all 1,500 rewrites. I read 20 rewrite decisions and record agreement (model-read, not human).

**Features and detectors.** Same model and revision as exp01, same extraction recipe. exp01 features are
not on this machine, so the exp01 pools are re-extracted; the decision-cell numbers on the originals must
match exp01's within 0.01 (legal / medical vs wildchat_en_test, coarse bands, dmaha_full 0.954 / 0.897)
before anything else is read. Decision cell cmean L16, Dolma full, detectors fit exactly as exp01.

**Q1, primary.** Fine-band (exp04) length-stratified AUROC, `legal_lx` vs `wildchat_lx` and `medical_lx`
vs `wildchat_lx`, for `dmaha_full`, `dknn_full` and the label-free tf-idf (mix vs Dolma). delta =
AUROC_d - AUROC_tfidf, paired bootstrap, 1,000 resamples, 98.75% CI (Bonferroni over 2 detectors x 2
pools). Gates:
- G1 topic kept: judge says LAWYER for >= 80% of `legal_lx` and DOCTOR for >= 80% of `medical_lx`.
- G2 words gone: label-free tf-idf stratified AUROC <= 0.65 on both primary comparisons.
- G3 balance: length-only stratified AUROC in [0.45, 0.55] on both.
Verdict, same rules as exp05:
- activations add beyond words: for at least one detector, on both pools, AUROC_d >= 0.75 and
  delta >= 0.10 with CI lower bound > 0. Requires G1 and G3.
- words only: for both detectors on both pools, AUROC_d <= 0.65 or delta CI upper bound < 0.10. Requires
  G1, G2 and G3.
- anything else, or a failed gate: ambiguous / inconclusive, named by which condition failed. If G1 fails
  the ban removed the topic, which is itself the finding that topic and words cannot be separated this way.

**Q2, transfer of supervised probes from originals to rewrites.** Per pool p in {legal, medical}: rows of
p and of `wildchat_en_test` are split into 5 folds (seed 0 permutation, by row index). Train on the
originals of 4 folds (p = 1, WildChat = 0), score the rewrites of the held-out fold; out-of-fold scores
pooled; fine-band stratified AUROC `p_lx` vs `wildchat_lx`. Arms: activation probe (standardised cmean
L16, logistic C = 0.1), tf-idf probe (exp01 vectoriser, C = 1.0), MiniLM probe
(`sentence-transformers/all-MiniLM-L6-v2` normalised embeddings, C = 1.0). Bar on activation minus tf-idf,
paired bootstrap 97.5% CI (2 pools): activations transfer beyond words if >= 0.10 with CI lower > 0 on both
pools; words suffice if CI upper < 0.10 on both; else ambiguous. MiniLM is reported beside it, no bar. The
same probes scored on held-out originals are reported as the in-distribution ceiling.

**Q3, idea 4: template-wrapped Dolma.** `dolma_wrapped` = each `dolma_matched` token cut, decoded and
placed in the Olmo chat template as a user message, content tokens only. `dmaha_wrapped` /
`dknn_wrapped` = score vs mix minus score vs `dolma_wrapped`. Compared with `dmaha_matched` /
`dknn_matched` (same text, no template) on exp01 `legal` and `medical` vs `wildchat_en_test`, exp01 coarse
bands and guard, paired bootstrap 95% CI on wrapped minus matched. Template mismatch contributed if any of
the 4 differences has |diff| >= 0.05 with the CI excluding 0; no contribution if all 4 CIs lie inside
[-0.05, 0.05]; else ambiguous.

**Reported, no bar.** One-sided maha/knn, dmaha_matched, dknn_matched, length on Q1 comparisons; originals
vs wildchat_en_test with fine and coarse bands; judge rates on originals; leak rates; Q1 restricted to
leak-free, judge-on-topic rows; a tf-idf-stratified readout (dmaha_full / dknn_full AUROC within quintiles
of the pooled label-free tf-idf score, on both originals and rewrites); Q3 detectors on the lx pools;
other layers and poolings for Q1.

**Seeds.** One data seed (0), one rewrite per row, one judge call per message. No extra seeds planned.

**exp10 amendment, 19-10-2026, before any exp10 rewrite of the evaluation pools exists.** A 5-row smoke
test of the pipeline (scratch output, not used for any question) showed the 100-per-direction ban list
(about 350 terms, including everyday words such as "months", "use", "like") cannot be followed: 5 of 5
legal rewrites leaked. The ban list is cut to the top 40 terms per direction per model (n_top = 40), which
still holds the topic words (legal, landlord, court, lease, law, doctor, pain, blood) and the strongest
narrative words. Everything else, including every gate and bar, is unchanged.

**exp10 amendment 2, 19-10-2026, same smoke test, still before any evaluation rewrite.** With n_top = 40,
gpt-4.1 still used banned words in most smoke rewrites ("legal", "lease", "months"). Each rewrite that
contains a banned term gets one repair call: the same prompt applied to the rewrite, plus the list of the
banned words it still uses. The repaired text replaces the first rewrite (a failed repair keeps the
first). Leak rates are reported before and after repair. Gates and bars unchanged.

## 19-10-2026 — exp09 v2: iterative gap discovery with equal review budgets

Written before any exp09 v2 ranking, cluster or name exists. This is the one rerun the ambiguity rule
allows for exp09 (`dknn` ambiguous), after a bug check found a confound in how the two conditions were
assigned. Everything in the exp09 entry above holds unless named here.

**Confound found in exp09.** Only confirmed clusters left the pool, so rows the top arm reviewed and
rejected were ranked and reviewed again in the next round, while the random control drew a fresh 500 each
round. `dr` moved no rows, so its five rounds reviewed one top 500 five times; `dknn` moved no rows after
round 2. 60-62 of the top arms' cluster example prompts recur across rounds against 1-2 for random. Over
five rounds the random control reviewed about 2,500 distinct rows and the top arms far fewer, so "distinct
gaps, top vs random" compared unequal review budgets.

**The one change.** A row a reviewer has reviewed is never reviewed again by that reviewer.
- Top arm: all 500 rows reviewed in a round leave the remaining pool. Members of confirmed clusters also
  join the ID side as before (`dr` label 0, `dknn` reference set); reviewed rows that were not confirmed
  join nothing. The next round ranks only unreviewed rows, with length bins recomputed on them.
- Random control: each round draws 500 rows uniformly from the pool minus rows that arm's top loop has
  confirmed (as exp09) minus rows the random control has already reviewed.
- So each group reviews exactly 2,500 distinct rows over 5 rounds, and the two arms stay separate loops.

**Bug fixes, no design change.** An empty or missing namer reply is a failed call: logged with its index,
not cached, retried up to 2 more times, then recorded as a null name (not confirmed). Input paths are
required CLI arguments. The code is committed before the run, so `config.json`'s git hash is the code
that ran.

**Unchanged.** Pool (exp06 WildChat 10,000 + exp09 LMSYS 10,000), ID side (exp01 idfit + idmix), features
(cmean L16), both scores, 20-bin length control, top 500 per round, 5 rounds, embedder, k-means k = 10
seed 0, 25 shown members, namer and prompt, coverage reference and the 25th-percentile cut computed on the
full 20,000-row pool, confirmation rule, same-gap rule (centroid cosine >= 0.85), seeds. The fresh run uses
its own naming cache.

**Bar, unchanged, per arm after 5 rounds.** Useful discovery: >= 5 distinct confirmed gaps AND >= 2x the
random control's count. No better than random: <= random + 1. Otherwise ambiguous. If `dknn` is ambiguous
again, it is logged as ambiguous and not rerun; there is no v3 of this design.

**Checks written into the results, no bar.** Distinct rows reviewed per group (must be 2,500); round 1 top
500 per arm identical to exp09's round 1 (same ranking code and seeds, so any difference is a bug); null
namer replies per group; per-round counts of COHERENT, >= 15 members, and below-coverage clusters for top
and random, so the binding criterion is visible.

## 19-10-2026 — exp11: ranking variants for gap discovery, tested on planted needles

Written before any exp11 feature, score or loop exists. Stefan's notes list four untested ranking changes
for the discovery loop (section 11 step 1, step 2, section 6.5 and the open ideas): remove length by
regression or by fitting the reference per length band instead of within-band percentiles; combine
distance to a known-OOD set with distance to ID instead of adding found rows to the ID reference; use
last-token features; aggregate scores per cluster rather than per row. exp09 and exp09 v2 could not
resolve ranking changes because their ground truth is a blind LLM namer. This run tests the four changes
where the truth is exact: needles planted in real traffic, as in exp06. No namer, no API.

**Pool.** exp09's 20,000-row deployment pool (exp06 WildChat 10,000, all languages, plus exp09 LMSYS
10,000) with its cached cmean / cmax / clast features at 9 layers. Needles: exp06's banks (legal,
medical as gap needles; aya, magicoder as covered controls), 100 rows of each type per needle seed s in
{0, 1, 2}, disjoint 100-row slices of each bank, planted together, so each type is 0.5% of the planted
pool and the gap group is 1%. Needle features are re-extracted with the exp01 recipe (same model and
revision b33130b7) into this run's directory. ID side: exp01 idfit + idmix (4,000). Dolma: exp01 full.

**Base scores, cmean L16 unless named.** `dknn` = knn(mix) - knn(Dolma), k = 10, features scaled by the
idfit std (exp06 form). `dmaha` = maha(mix) - maha(Dolma), fixed shrinkage 0.5. One-sided `knn` and
`maha`, the exp01 label-free tf-idf (idfit vs Dolma, C = 0.1), `length` and `random` as baselines.

**Q1 length removal, `dknn` cmean L16.** Rankings: `raw`; `pct20` (exp06, percentile within 20
equal-count length bins, the baseline); `spline` (residual of the score after a ridge regression, alpha 1,
on a cubic B-spline basis with 8 knots over log content-token length, fitted on the planted pool);
`bandfit` (the pool split into 5 equal-count length bands; each band scored against the 500 mix rows and
500 Dolma rows nearest to the band's median log length; scores ranked globally). Metric: gap uplift@100
(precision@100 of legal + medical divided by 0.01), control uplift@100 (aya + magicoder), top-100 median
length ratio to the pool, Spearman of the ranking score with log length. Mean over the 3 needle seeds.
Bar, per method M against `pct20`: better if gap uplift(M) >= 1.5 x uplift(pct20) and >= uplift(pct20)
+ 2, with top-100 median length ratio in [0.67, 1.5] on every seed and control uplift <= 2; worse if
gap uplift(M) <= uplift(pct20) / 1.5; otherwise equivalent.

**Q2 pooling, `dknn` under `pct20`.** `clast` and `cmax` at L16 against `cmean` at L16, same metric and
the same better / worse / equivalent rule relative to `cmean`. Layers 8, 24 and 32 are reported for all
three poolings with no bar.

**Q3 per-cluster aggregation, `dknn` cmean L16 `pct20`.** k-means (k = 200, seed 0) on the standardised
cmean L16 features of the planted pool; a cluster's score is the mean `pct20` score of its members;
clusters are reviewed in descending cluster score, rows within a cluster in descending row score, and
the first 100 rows reviewed form the top 100. k = 50 is reported with no bar. Same rule relative to
per-row `pct20`, with the control-uplift <= 2 condition.

**Q4 known-OOD rule in the iterative loop.** Bases `dknn` and `dmaha`, cmean L16, `pct20` ranking, 5
rounds x top 200, needle seeds 0-2. The reviewer is the type oracle: reviewed needle rows are confirmed
(known-OOD), reviewed deployment rows are rejected, and every reviewed row leaves the pool (exp09 v2).
Update rules: `none` (references unchanged); `add_to_id` (confirmed rows join the mix reference, exp06 and
exp09's rule; for `dmaha` the Gaussian is refitted on mix + confirmed); `known_gate` (references
unchanged; once >= 5 rows are confirmed, any row whose kNN distance to the confirmed set, k = min(10,
n_confirmed), is at or below tau is removed from the ranking, tau = the 90th percentile of the confirmed
rows' leave-one-out kNN distance to each other). Metrics per round and group: cumulative recall per
needle type, surface round per gap type (first round with >= 5 cumulative reviewed rows of that type),
share of the round's top 200 that is of round 1's dominant needle type, LMSYS share and median length
ratio of the top 200. Bar, per base, `known_gate` against `add_to_id`: better if the second gap type
surfaces in an earlier round on >= 2 of 3 seeds and round-5 total gap recall is not lower by more than
0.05 on any seed; equivalent if the surface rounds are identical on all seeds; otherwise ambiguous.
`none` is reported beside them with no bar.

**Reported, no bar.** The 10 highest-ranked non-needle rows of `dknn` cmean L16 `pct20` at seed 0 (first
300 characters, source, language where known), read as the raw-output check. One-sided `knn` and `maha`,
tf-idf, length and random under every Q1 ranking. Scores for every seed-0 row written to a jsonl.

**Seeds.** Needle seeds 0-2 for every question; k-means seed 0; no other randomness except 1e-9 tie
jitter. No extra seeds planned.

**Sanity checks written into the results.** Random ranking gap uplift within [0.5, 2] at every seed;
needle feature row ids match the bank files; planted pool row count = 20,400 per seed; every loop group
reviews exactly 1,000 distinct rows.

**exp11 amendment, 19-10-2026, after needle feature extraction, before any exp11 score exists.** The
`spline` ranking's ridge penalty is cut from alpha = 1 to alpha = 0.001: on a synthetic score that is a
pure function of log length (5,000 rows), alpha = 1 left a residual correlated 0.22 with log length, so
it would not have removed length; alpha = 0.001 leaves 0.0005. Nothing else changes.

## 19-10-2026 — exp12: the exp01 claim on a second model pair (OLMo-2 1B SFT with the Tulu-3 SFT mix)

Written before any exp12 pool, feature or score exists. Stefan's notes (section 9.1, point 3) say that on
a second model pair the two-sided detectors did not beat the one-sided ones; every run in this repo used
Olmo-3-7B-Instruct-DPO. The OLMo-2 1B DPO preference mix is a single source (75,661 WildChat prompts),
so against it "legal vs English WildChat" is OOD-vs-mix and cannot show a one-sided collapse. The SFT
step's mix, `allenai/tulu-3-sft-olmo-2-mixture-0225` (19 sources, WildChat among them), mirrors Dolci's
structure and is the ID mix here.

**Model and features.** `allenai/OLMo-2-0425-1B-SFT`, revision 0d85a3d0, bf16, 16 blocks, hidden 2048.
Residual stream `hidden_states[l]` for l in 0, 2, ..., 16 (every eighth of depth, as exp01 took every
fourth layer of 32). exp01 recipe otherwise: the model's chat template (`<|user|>\n...<|assistant|>\n`),
content tokens only, left truncation at 1024, cmean / cmax / clast. Decision cell cmean L8, Dolma full.

**Pools, `data/processed/exp12`, seed 0.** `idfit` 3,000 and disjoint `idmix` 1,000 first user turns
of 2-message Tulu rows, sampled by source share. `dolma`, `legal`, `medical` are exp01's files copied
unchanged; `legal` and `medical` are membership-checked against the Tulu mix and any member rows are
reported and dropped. `wildchat_en_test`: 500 English WildChat first turns from shards 0, 5, 10 (exp01's
shards), deduplicated, with Tulu-mix members removed by the exp01 check (lowercase, collapsed whitespace,
exact match plus 13-word blake2b shingles) against all Tulu first user turns; the check must fire on
200/200 sampled Tulu rows before any zero is accepted.

**Detectors, fit on `idfit` only, as exp01.** Mahalanobis (Ledoit-Wolf, 1e-6 ridge), kNN k = 10 on
idfit-std-scaled L2-normalised features, `dmaha_full` / `dmaha_matched`, `dknn_full` / `dknn_matched`,
tf-idf idfit vs Dolma (C = 0.1, Dolma cut to 400 words), length. Length-stratified AUROC with exp01's
coarse bands and guard; 1,000 bootstrap resamples on the decision cell.

**Positive control.** kNN legal and medical vs `idmix`, raw AUROC >= 0.90 on both. Missed = the pipeline
is suspect; the bars below are not evaluated until a bug check has run.

**Measurability.** Length-only stratified AUROC for legal and medical vs `wildchat_en_test` within
[0.45, 0.55] on both; otherwise the comparison is unmeasurable, not negative.

**Q1, primary.** Decision cell, legal and medical vs `wildchat_en_test`, length-stratified.
- pattern reproduces: `maha` and `knn` <= 0.60 on both pools and `dmaha_full` and `dknn_full` >= 0.85 on
  both (exp01's bar 2).
- no two-sided advantage (Stefan's second-pair finding): `dmaha_full` - `maha` <= 0.05 and `dknn_full` -
  `knn` <= 0.05 on both pools.
- anything else: ambiguous, reported with the eight numbers and their intervals.

**Q2, lexical.** tf-idf stratified vs `wildchat_en_test` against the better of `dmaha_full` and
`dknn_full`, per pool: within 0.05 on both = lexical parity; at least 0.10 below on both = activations
beat words; otherwise ambiguous.

**Reported, no bar.** All 9 layers x 3 poolings for every detector and comparison; `wildchat_en_test` vs
`idmix` as the near-ID control (Tulu contains WildChat); Spearman of `maha` with content-token length in
each pool; the `matched` Dolma variants; bootstrap intervals on the decision cell.

**Seeds.** Data seed 0, one extraction. No extra seeds planned.

**exp11 amendment 2, 19-10-2026, after a one-seed smoke run with a 2 x 50 loop and k = 20 (scratch
directory `261019_smoke_exp11`, not used for any question), before the pre-registered run.** Two changes.
(a) The random-ranking sanity check "gap uplift within [0.5, 2] at every seed" cannot hold: with 200 gap
needles in 20,400 rows the random top 100 holds a Poisson(1) count, so a single seed reads 0 about 37% of
the time (the smoke seed did). The check becomes: mean random gap uplift over the 3 seeds within [0, 3].
(b) `bandfit` ranked globally correlates 0.70 with log length in the smoke run, because per-band
reference scores are not on one scale; it keeps its bar as pre-registered, and a `bandfit_pct20` variant
(per-band references, then within-band percentiles) is added as reported, no bar.

## 19-10-2026 — exp13: multi-turn and tool-use traffic on the 7B pair

Written before any exp13 pool, feature or score exists. Stefan's section 4.10 lists multi-turn traffic and
tool-use traffic as never tested on the 7B pair. Facts measured from the mix before this entry: Dolci-DPO
has 3.85% multi-turn rows (9,999 of 259,922), all in eight synthetic `multiturn_*` families (self talk,
related query, paraphrase, repeat), and no tool-use or function-calling family. exp01's `idfit` was drawn
from 2-message rows only, so it under-represents even that 3.85%.

**Features.** Olmo-3-7B-Instruct-DPO rev b33130b7, exp01 recipe. A conversation pool row carries
`messages`; it is rendered with the chat template up to and including its last user turn, with the
generation prompt, left-truncated at 1024 tokens, and the content mask covers only the last user turn's
tokens. The earlier turns are context in the forward pass but contribute no pooled tokens, so a message
scored with and without context has the same content-token count by construction.

**Pools, `data/processed/exp13`, seed 0, 500 rows each.**
- `wc_mt_ctx`: English WildChat conversations (shards 0, 5, 10) with at least two user turns; messages
  = first user turn, first assistant turn, second user turn. `wc_mt_alone`: the same second user turns
  rendered alone. Both user turns are checked against every Dolci-DPO user turn with the exp01 membership
  check; members are skipped. Stored assistant turns are GPT-3.5/4 text (Stefan's authorship caveat); they
  are context only.
- `dolci_mt_fit` and `dolci_mt_test`: disjoint rows of the eight `multiturn_*` families sampled by family
  share; messages = the chosen conversation up to its last user turn. `dolci_mt_test_alone`: the test
  rows' last user turn alone.
- `tool_ctx`: `NousResearch/hermes-function-calling-v1`, `func-calling-singleturn.json` (1,893 rows);
  messages = the system turn carrying the tool signatures plus the user turn. `tool_alone`: the user turn
  alone. No tool family exists in Dolci-DPO, so this is a gap by construction.
- `wildchat_en_test`: exp01's 500 rows, re-extracted (their features are not on this machine). `idfit`,
  `idmix` and `dolma_full` features come from exp01's cache.

**Pipeline check.** Re-extracted `wildchat_en_test` dknn_full and maha scores at cmean L16 (ref `idfit`)
must have Spearman >= 0.99 with exp01's stored `scores_decision_cell.jsonl` values for the same ids.
Missed = the pipeline is suspect; bars are not evaluated until a bug check has run.

**Detectors.** Two ID references: `idfit` (exp01, first turns only) and `idfit_mt` = `idfit` +
`dolci_mt_fit` (the mix with its multi-turn family represented). Mahalanobis (Ledoit-Wolf), kNN k = 10
(scaled by the reference's std), `dmaha_full`, `dknn_full` (exp01 Dolma full), tf-idf `idfit` vs Dolma,
length. Comparisons: every pool vs `wildchat_en_test` (primary) and vs `idmix`; exp01 coarse bands and
guard; 1,000-row bootstrap on bar cells, paired for ctx vs alone of the same rows.

**Measurability.** Length-only stratified AUROC within [0.45, 0.55] for the comparison; otherwise
unmeasurable, not negative.

**Bars, ref `idfit_mt`, cmean L16, each of `dknn_full` and `dmaha_full`, stratified vs
`wildchat_en_test`.**
- Control: `dolci_mt_test` (with context) <= 0.65 = the mix's own multi-turn rows read as covered;
  >= 0.85 = the detector flags the multi-turn format itself, and Q1 cannot be read as coverage; else
  ambiguous.
- Q1 real multi-turn: A = `wc_mt_ctx`, B = `wc_mt_alone`. A >= 0.85 and B <= 0.65 = context makes the
  later turn a gap; both >= 0.85 = later-turn messages are a gap in themselves; both <= 0.65 = not a gap;
  otherwise ambiguous. Read only if the control is <= 0.65.
- Q2 tool use: `tool_ctx` and `tool_alone`, each: >= 0.85 = flagged as a gap; <= 0.65 = missed;
  otherwise ambiguous. tf-idf is reported beside each.

**Reported, no bar.** The same table with ref `idfit`; one-sided `maha` and `knn`; clast and all 9 layers;
the paired ctx-minus-alone difference with its interval; five raw rows of `wc_mt_ctx` (both user turns,
truncated) as the raw-output check.

**Seeds.** Data seed 0, one extraction. No extra seeds planned.

## 19-10-2026 — exp14: does the cluster, name and coverage step confirm gaps correctly when the answer is known?

Written before any exp14 cluster, name or score exists. exp09 and exp09 v2 rest on three steps nobody has
checked against known answers: k-means on multilingual MiniLM embeddings, a blind LLM namer's COHERENT
call, and a coverage cut (cosine to the nearest unique Dolci-DPO first user turn below the pool's 25th
percentile). This run plants prompts whose answer is known into the exp09 pool and checks each step. No
ranking and no Olmo features are involved; ranking is exp11's question.

**Known answers.** exp06 needle banks (300 each, Dolci members already removed). Gap (uncovered by the
mix): `legal` (legal-advice-reddit), `medical` (HealthCareMagic). Covered: `magicoder` (code, a domain the
mix covers heavily), `aya` (multilingual, covered by the mix's Aya family). Background: the exp09 20,000
WildChat + LMSYS first turns. Draws are disjoint within a seed and independent across seeds 0, 1, 2.

**Fixed from exp09.** Embedder `paraphrase-multilingual-MiniLM-L12-v2` (same revision), namer
`anthropic/claude-sonnet-4.6` with exp09's prompt and 25 shown members, k-means k = 10 seed 0, min size 15,
coverage reference and cut. Pipeline check before anything is read: the cut recomputed on the 20,000 pool
must be within 0.001 of exp09 v2's 0.6003. Empty namer replies retried up to 3 times, then null.

**Q1, namer on clusters of known make-up.** For legal, medical and magicoder (aya is excluded here: its rows
span many languages and tasks, so it has no single right answer), per seed, four 50-row clusters: pure (50
of the type), majority (35 type + 15 random pool rows), minority (15 + 35), random (50 pool rows). 36
clusters, each named once, blind.
- Coherence valid: pure COHERENT >= 8/9, random COHERENT <= 1/9, and majority YES rate > minority YES
  rate. Invalid: pure <= 6/9 or random >= 3/9. Otherwise ambiguous.
- Names valid: a second-provider judge (`openai/gpt-4.1`, sees only the name and description, no
  examples) maps each pure and majority cluster's name to LEGAL, MEDICAL, CODE or OTHER; >= 15/18 match
  the planted type = valid, <= 11/18 = invalid, otherwise ambiguous.

**Q2, coverage on known make-up.** Median coverage of the pure clusters against the cut: legal and medical
(6 clusters) should fall below it; magicoder and aya (6 clusters; one 50-row pure aya cluster per seed,
built but not named) should not. Valid: 6/6 gap below and 0/6 covered below. Invalid: <= 3/6 gap below or
>= 3/6 covered below. Otherwise ambiguous. Reported, no bar: row-level coverage medians for all 300 rows of
each bank and the pool, and AUROC (lower coverage = positive) for each bank vs the pool.

**Q3, end to end on a review batch.** Per seed, a 500-row batch: 400 random pool rows plus 25 of each of the
four types, clustered (k = 10), named, and confirmed with exp09's rule (COHERENT, >= 15 members, median
coverage below the cut). A type counts as found when a confirmed cluster holds >= 13 of its 25 rows and the
type is >= 50% of that cluster. A confirmed cluster that is < 50% planted is a background confirmation.
Valid: gap types found >= 5/6, covered types found <= 1/6, background confirmations <= 3 over 3 seeds.
Invalid: gap found <= 2/6 or covered found >= 3/6. Otherwise ambiguous. Reported, no bar: for each type and
seed, the largest share of its 25 rows in one cluster (clustering purity), and every confirmed cluster's
name.

**Verdict.** The step is validated on planted categories only if Q1 (both parts), Q2 and Q3 are all valid;
any invalid part names the step that fails. A pass is necessary, not sufficient: the planted types are
topic-coherent and far apart, easier than real gaps, so a pass would not show the step finds real gaps, but
a fail would show it cannot be trusted even here.

**Model-read spot check.** I read every pure and random cluster's name against its make-up (18 clusters)
and record agreement as model-read, not human.

**Seeds.** Draw seeds 0, 1, 2 (pre-registered here, not added after); one naming call per cluster; one judge
call per name.
