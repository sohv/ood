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
