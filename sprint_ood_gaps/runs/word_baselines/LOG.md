# Word-baseline harness for the OOD known-gap tasks

Machine: `research-box` (GCP e2-standard-8, 8 vCPU, 32 GB, no GPU). CPU only, no paid services. Project venv `ood-gaps/.venv` (Python 3.12.3); deps added with `uv add` (numpy, scipy, scikit-learn 1.9.1, pandas, pyarrow, matplotlib, transformers 5.20.2 (tokenizers only), sentence-transformers 5.7.0, torch 2.14.1+cpu). ood-gaps has no git repo, so nothing is committed.

Times are UTC. Each entry says when the choice was made.

## 2026-10-21 05:42 Inputs

- Source of every test pool: Sohan's repo `github.com/sohv/ood` at commit 09b2edf (read-only clone in the session scratchpad). His processed pools for exp01, exp06, exp09 and exp12 are committed in `data/processed/`. I copied them unchanged to `inputs/` (hashes in `inputs/SHA256SUMS`). So the test rows are his rows exactly, not a re-sample from public data. No code of his was run.
  - exp01 (Olmo-3-7B-Instruct-DPO setup): `idfit` 3,000 and `idmix` 1,000 Dolci-Instruct-DPO 2-message first turns; `legal` 500 (legal-advice-reddit train); `medical` 500 (HealthCareMagic); `wildchat_en_test` 500 English WildChat first turns, shards 0/5/10, Dolci members removed; `dolma` 3,000 Dolma v1.6 docs.
  - exp12 (OLMo-2-0425-1B-SFT setup): `idfit` 3,000 / `idmix` 1,000 from the Tulu-3 SFT OLMo-2 mix; `wildchat_en_test` 500 with Tulu members removed (a different 500 from exp01's); legal, medical and dolma are byte-identical to exp01's (same sha256; his `scripts/run_build_pair_pools.py:64` copies dolma).
  - exp11 (planted needles): `exp06/deploy` 10,000 WildChat first turns, all languages, shards 1-3; `exp09/lmsys` 10,000 LMSYS-Chat-1M first turns (one row, `lmsys_03732`, has an API key replaced with `sk-REDACTED` in the committed copy; his features used the original text; effect on word scores: one row of 20,000); needle banks of 300 each: `legal` (legal-advice-reddit test), `medical` (HealthCareMagic, disjoint from exp01), `aya`, `magicoder`.
  - The LMSYS rows come from his committed file, so the HF token was not needed and was not read.
- Difference from his setup: none in the rows. His `exp12/dolma.jsonl` is not committed (over 5 MB); his build script copies exp01's, which I use.

## 2026-10-21 05:46 Lengths

- `n_content_tokens` recomputed with each model's own tokenizer and chat template at the revision he logged (Olmo-3-7B-Instruct-DPO b33130b7 for exp01/exp11, OLMo-2-0425-1B-SFT 0d85a3d0 for exp12), as in his `src/interp/features.py:16-31` (content-token mask in the rendered chat, left truncation to 1,024 tokens, so content length caps at 1,019). Script `scripts/wb_lengths.py`, cache `cache/lengths/`.
- Check: the length-only stratified AUROC must come out at his logged 0.50 / 0.45 (exp01) and 0.507 / 0.474 (exp12).

## 2026-10-21 05:49 Validation split for tuning (added after the coordinator's tuning requirement, before any scoring)

- Every word baseline's hyperparameters are tuned on validation pools that share no rows with any test pool, never on test. `scripts/wb_build_val.py`, outputs `inputs/val/`:
  - `val_legal`: 300 posts from legal-advice-reddit `validation.jsonl` (test pools use `train.jsonl` and `test.jsonl`), exp01 legal and exp06 legal bank excluded.
  - `val_medical`: 300 HealthCareMagic inputs, excluding exp01 medical and the exp06 medical bank (normalised-text match).
  - `val_wildchat_dolci` / `val_wildchat_tulu`: 500 English WildChat first turns from shard 7 (unused by any test pool: those use 0, 5, 10 and 1, 2, 3), rows in any test WildChat pool skipped, mix members removed with his `MembershipIndex` (exact, 13-word shingles, whole-word substring for short rows; `src/data/pools.py:159-175`), applied to the full Dolci-Instruct-DPO and Tulu-3 mixes.
  - Sampling seed 1. Dataset revisions in `inputs/val/build_stats.json`.
- Objective: mean of the two length-stratified AUROCs (val legal vs val WildChat, val medical vs val WildChat), same banding as the test statistic. Fit set = that experiment's `idfit`; reference = exp01 Dolma (first 400 words).
- Grid (fixed before tuning): tf-idf word 1-gram / 1-2-gram: min_df {1,2,5} x sublinear tf {yes,no}; char 3-5: analyzer {char_wb, char} x min_df {2,5} x sublinear {yes,no}; kNN k {1,3,5,10,25,50}; logistic C {0.01,0.1,1,10} sparse, {0.1,1,10,100} on embeddings; Mahalanobis shrinkage {Ledoit-Wolf, 0.01, 0.1, 0.5}; unigram smoothing alpha {0.01,0.1,0.5,1}. Ties go to the first setting in grid order.
- Effort comparison: Sohan's activation detectors were not tuned on a validation split. Layer, pooling and k were fixed or swept on the test pools (exp01 swept 9 layers x 3 poolings and reported the pre-chosen cmean L16 cell; exp11 picked cmax / L32 on the same pool it reports). So the word baselines here get more held-out tuning than the activation numbers they are compared with, but none of their test numbers are tuned on test.
- The exp11 arms use the exp01 (Dolci) tuned settings. The validation task (legal / medical vs WildChat AUROC) is not the exp11 ranking task, and the validation rows are disjoint from the exp11 needles and background.

## 2026-10-21 05:55 Ways the result could be wrong, and the check for each (written before scoring)

1. Length. Legal posts are longer than WildChat turns (median 218 vs 57 content tokens on exp01). Check: every known-gap AUROC is reported raw and length-stratified (his 7 bands, pairs-weighted, `detectors.py:67-95`); length alone is an arm and must give about 0.5 stratified; exp11 rankings are reported raw and within 20 equal-count length bins (`pct20`) with the top-100 median length ratio.
2. Source and format artefacts that identify the dataset rather than the topic.
   - Chat templates and system prompts: word baselines see the raw user text only, so the Olmo-3 "function-calling assistant" system prompt cannot leak into them. The activation numbers they are compared with do see it.
   - Mix format: Dolci / Tulu prompts carry instruction-following constraints ("Your response should contain ..."), math and code templates. A mix-vs-Dolma classifier can score "not like the mix" from the absence of these. Check: top logistic coefficients per two-sided arm, and the WildChat-vs-idmix comparison (near-ID control) for every arm.
   - Markdown, reddit and HealthCareMagic boilerplate: legal rows are reddit "title\n\nbody" with jurisdiction tags like "(TN)"; medical rows start "hi doctor", "hello", use "I m" with stripped apostrophes. Check: count these patterns per pool, read top-scored rows, and test the best arm on the validation pools, which come from other splits of the same sources (so a source artefact would also help there; this check cannot remove it, only measure it).
   - LMSYS `NAME_1` placeholders and language: matter for exp11 only. Check: report `NAME_` and non-ASCII share in the exp11 top-100s per arm.
3. Near-duplicates between mix and test. Sohan removed WildChat rows that are Dolci / Tulu members (shingle check against the full mix). Check: max tf-idf cosine of every test row to `idfit`, counts above 0.8 and 0.9, per pool; and the same between test pools.
4. Topic words leaking from how the gap set was built: legal and medical sets were built by source, not by keyword, so no keyword filter defines them. The leakage risk is the other direction: the gap is "a topic the mix lacks" and the word baseline reads topic words directly, which is the point of the baseline, not a leak. The exp10 banned-word rewrites, which would test this, are not committed in usable form for both models and are not used here.
5. Label noise. "Legal" = posted to r/legaladvice; "medical" = HealthCareMagic question. Some WildChat rows are also legal or medical questions, so the ceiling is below 1.0. Check: read the 20 bottom-scored gap rows and the 20 top-scored WildChat rows for the best arm.
6. Near-perfect AUROC (>= 0.98 stratified) is treated as a likely artefact until the arm's top features are read.
7. Selection among many arms: the "best word baseline" is the max over about 23 arms. I name the best arm by its validation objective (chosen without test data) and also report the test-best arm, marked as selected on test.
8. Reproduction drift: my rerun of his exact tf-idf arm (`tfidf1_lr_sohan`, `detectors.py:55-60`) must give his logged 0.952 / 0.856 (exp01), 0.918 / 0.824 (exp12) and his exp11 `pct20` uplift 10.5. If it does not, the harness differs from his and the comparison is not like for like.

## 2026-10-21 05:54-06:20 Validation build, first attempt

- Provenance check passed (`inputs/val/build_stats.json` once written): exp01 legal 500/500 rows found in legal-advice-reddit `train.jsonl`; exp06 legal bank 300/300 in `test.jsonl`; exp01 medical 500/500 and exp06 medical bank 300/300 in HealthCareMagic.
- The first run spent 26 min in the Dolci membership step. Cause: my array version of his shingle index used `np.isin` against a sorted array of all mix shingles, which re-sorts that array on every query. Replaced at 06:20 by `np.searchsorted` (same membership answer; unit-checked on a toy index). Restarted 06:20.
- Embeddings (`scripts/wb_embed.py`) run in parallel: all-MiniLM-L6-v2 at about 25-45 rows/s on 8 vCPU shared with the build (measured on the 200-row tiny pass: 4 min 20 s for 2 x 15 pools x 200 rows).

## 2026-10-21 06:28-06:36 Tiny run (200 rows per pool, exp01 and exp11 only; outputs `out/tiny/`, `scores/tiny/`)

- Dolci validation WildChat: 570 shard-7 English first turns checked, 70 Dolci members removed, 500 kept; membership positive control 200/200 (06:28).
- `wb_tune --limit 200`: 2 min 42 s for the full grid (one fit per grid point; 23 arms). `wb_known_gap --limit 200 --n_boot 200`: 63 s. `wb_ranking --limit 200`: 54 s.
- Checked the saved outputs are usable: per-prompt score rows carry id, pool, content length, all 23 arm scores and the text; the AUROC table has raw and stratified values with intervals, band counts and coverage; checks file has near-duplicate counts, pattern rates and top logistic features; the reading file has random rows, top / bottom 20 of the validation-best arm, and within-pool extremes; the ranking outputs have per-seed and mean uplift, top-100 length ratio, `NAME_` / LMSYS / non-ASCII shares in the top 100, cluster labels and density-ratio scores per seed.
- One gap found: at n = 200 and 200 bootstrap resamples the medical-vs-WildChat stratified interval came back empty (`bootstrap_ci` returns none when more than 10% of resamples have fewer than 3 usable bands, his rule, `detectors.py:102`). Kept his rule; at n = 500 his exp12 run got intervals, and I check the full run for it.
- First looks at the tiny numbers (not used for any choice): the multilingual MiniLM one-sided kNN was the validation-best arm; Sohan's tf-idf arm gave 0.920 legal stratified on 200 rows. The WildChat rows the embedding kNN scored lowest were copies of the "As a prompt generator for ... Midjourney" template, and the highest-scored WildChat rows included medical and legal questions ("What should I take after otoplasty?", "US tax law regarding gifts"): label noise of the kind listed in item 5.

## 2026-10-21 06:37 Expected run time of the full run (stated before it starts)

- Basis: the tiny timings above, scaled by the number of fit and test rows. kNN cost is test x fit rows (15x the tiny run for known-gap, about 100x for exp11), logistic fits grow about linearly with fit rows (15x).
- Tulu membership index and the Tulu validation WildChat pool: about 10 min (Dolci took 2.7 min to index 260k rows; Tulu has 866k).
- Remaining embeddings, including validation pools: about 25 min (all-MiniLM ~28 rows/s, multilingual ~35 rows/s while sharing the CPU).
- Tuning, per experiment: 30-45 min; the two run in parallel.
- Known-gap scoring + 1,000-resample bootstrap, per experiment: about 10 min.
- exp11 bench: 20-40 min (kNN on 20,400 x 4,000, 3 density-ratio cross-fits, 6 k-means fits with k = 200).
- Total wall clock after the embeddings finish: about 1.5 h.

## 2026-10-21 06:46 Validation pools built; full tuning started 06:47

- Tulu validation WildChat: 658 checked, 158 Tulu members removed, 500 kept; positive control 200/200. Provenance: 300/300 sampled exp01 idfit rows are Dolci members, 300/300 exp12 idfit rows are Tulu members. All in `inputs/val/build_stats.json`. Shard 7 had 27,992 unique English first turns.
- Tulu row count after dropping rows without a user turn printed as 866,128, equal to his pre-drop count; the dataset revision may differ from his. Only used for membership removal.
- Validation pools are not hashed into the main manifest yet; see below.

## 2026-10-21 06:50 Added check: Reddit register (Sohan's exp03 pools), chosen before any full-run result

- His exp03 found the activation two-sided scores rank r/legaladvice below r/relationship_advice and r/relationships (0.296 / 0.195 for dmaha_full / dknn_full) and tf-idf gives the same ordering. Word baselines that look strong on legal vs WildChat could be reading "personal-narrative Reddit post" rather than "legal topic". `scripts/wb_register.py` scores his 8 exp03 pools (500 each, reddit-title-body 2019; legal_rtb = r/legaladvice) with the exp01-tuned arms and reports legal_rtb vs each covered subreddit and each pool vs English WildChat, length-stratified.

## 2026-10-21 07:01 Tuning round 1 done (validation only); grid extended at its edges, round 2 started

- Round 1 (06:47-07:01, 14 min per experiment, in parallel): best validation objective per arm in `out/full/tuning_round1/` and `logs/tune_exp*_round1.log`. Validation-best arm on both mixes: multilingual MiniLM one-sided kNN (exp01 0.974, exp12 0.959).
- Several winners sat on the edge of the grid (k = 50 for most kNN arms, C = 0.01 for word logistic regression, shrinkage 0.5, alpha 1.0). Extended to k up to 200, C down to 0.001 (sparse) / 0.01 (dense), shrinkage 0.9, alpha 3.0, and re-tuned. This uses validation results only; no test pool has been scored with the full harness yet.

## 2026-10-21 07:14 Tuning round 2 done; full run started 07:15

- Round 2 changed the validation objective by at most +0.017 per arm (mminilm_maha exp01 0.957 -> 0.974; most arms unchanged). Some winners are still on the new edge (k = 200 for exp12 kNN arms, shrinkage 0.9, alpha 3.0). Stopped extending: the remaining gains from round 1 to 2 were 0.00-0.02, and the objective is a validation number with its own noise (300 + 300 vs 500 rows). Final settings: `out/full/tuned_params_exp01.json`, `out/full/tuned_params_exp12.json`; full grids in `out/full/tuning_grid_*.json` (one row per setting).
- Validation-best arm (named before any test scoring, used as "the best baseline" for the reading step): multilingual MiniLM one-sided kNN on both mixes (exp01 k = 100, objective 0.974; exp12 k = 200, 0.964). Mahalanobis on the same embedding ties it on exp01 (0.974).
- Four jobs launched in parallel on 8 vCPU: `wb_known_gap` exp01 and exp12, `wb_ranking` (exp11), `wb_register` (exp03).

## 2026-10-21 07:20 Full known-gap and register results in; added a source-style check

- Known-gap and register runs finished 07:18 (4.5 min each). Reproduction check (item 8) passed: my rerun of his tf-idf arm gives exp01 0.952 / 0.855 stratified (his log 0.952 / 0.856) and exp12 0.918 [0.889, 0.944] / 0.824 [0.774, 0.869] (his 0.918 [0.889, 0.944] / 0.824 [0.774, 0.870]); length alone 0.498 / 0.446 and 0.507 / 0.474 (his 0.50 / 0.45 and 0.507 / 0.474).
- Item 6 triggered: multilingual MiniLM one-sided Mahalanobis and kNN reach 0.98-0.99 stratified on medical vs WildChat on both models. Treated as a likely source artefact until checked. Embedding arms have no readable features, so the check is behavioural: Sohan's exp05 pools (GPT-4.1 rewrites of exp01 legal, medical, WildChat and idmix into the mix's style; committed in his repo; copied to `inputs/exp05/`) remove HealthCareMagic and Reddit surface cues while keeping topic. `scripts/wb_style.py` scores them with the same fit and settings, using his fine bands (2^(1/4)) and the coarse bands. If the 0.99 is HealthCareMagic style, it should drop on `medical_rw` vs `wildchat_rw`. Added after seeing the 0.99; it is a check, not a new headline comparison.

## 2026-10-21 07:34 Full run complete: results, checks, reading

Run times (from logs): known-gap 4.5 min per experiment, register 4 min, exp11 bench 10 min 12 s, style check 3 min, all in parallel on 8 vCPU; wall clock from 07:14 to 07:34. Below my 1.5 h estimate because the k-means and density-ratio fits were faster than scaled from the tiny run.

Raw outputs: per-prompt scores `scores/full/*.jsonl`; tables `out/full/known_gap_tables.md`, `out/full/ranking_exp11_table.md`, `out/full/*.csv`; checks `out/full/checks_exp*.json`; reading `out/full/read/read_exp*.json`; figure `../../figs/word_baselines/word_baselines_vs_activations.png` (script `scripts/wb_report.py`).

Reproduction (item 8): his tf-idf arm reran to his logged numbers on all three tasks (exp01 0.952 / 0.855, exp12 0.918 / 0.824 with the same intervals to 0.001, exp05 rewrites 0.966 / 0.904, exp11 pct20 gap uplift 10.54 vs his 10.5, random 0.68 vs 0.68). The harness matches his setup.

Results, length-stratified AUROC vs English WildChat, n = 500 vs 500, 1,000 bootstrap resamples:
- Validation-best arm (multilingual MiniLM one-sided kNN, chosen on validation only): exp01 legal 0.959 [0.943, 0.974], medical 0.990 [0.971, 0.995]; exp12 legal 0.956 [0.934, 0.974], medical 0.982 [0.961, 0.990].
- Test-best arm (max over 23 arms, so optimistically selected): legal = unigram naive-Bayes log ratio mix vs Dolma (`uni_nb`) 0.968 on both models; medical = multilingual MiniLM one-sided Mahalanobis 0.993 (exp01) / 0.991 (exp12).
- Paired difference vs his tf-idf, val-best arm: medical +0.135 [+0.081, +0.189] (exp01), +0.158 [+0.104, +0.203] (exp12); legal +0.007 [-0.017, +0.031] (exp01), +0.038 [+0.002, +0.072] (exp12).
- Stronger tf-idf variants (bigrams, character n-grams, tuned C, kNN on tf-idf) do not beat his unigram tf-idf by more than 0.05 anywhere (max: character 3-5-gram classifier on exp12 medical, 0.869 vs 0.824); the gain comes from the sentence embedding, one-sided.
- One-sided tf-idf / character kNN and the unigram density are at or below 0.5 stratified (0.20-0.70): sparse lexical distance to the mix does not find these gaps; the length-controlled signal in sparse features comes only from the two-sided contrast with Dolma.
- exp11 (200 gap needles in 20,400 rows, 3 needle seeds): val-best arm per row, pct20: gap uplift@100 30.6 (seeds 31.6 / 30.6 / 29.6), legal 19.0, medical 42.2, control 0.0, top-100 length 0.98x pool. Binomial 95% interval for one seed's 30 / 100 hits: uplift 22-40. Whole-cluster review with word scores reaches 52.7 but its top 100 is 4.1-6.4x the pool's median length, which fails his [0.67, 1.5] balance rule. The deploy-vs-mix tf-idf density ratio gives 0.0, with 98% of its top 100 being LMSYS `NAME_` placeholder prompts.

Checks:
- Item 3 (near-duplicates): no legal, medical or WildChat test row has tf-idf cosine >= 0.8 to any idfit row (both models); idmix has 8 / 9 rows >= 0.8, 3 / 0 >= 0.9 (held-out mix rows with near-twins in the fit set, expected).
- Item 6 (near-perfect medical AUROC of the embedding arms): survives the source-style check. On Sohan's GPT-4.1 rewrites into the mix's style, medical_rw vs wildchat_rw is 0.993 [0.985, 0.998] (fine bands) for the val-best arm, and 0.990 for medical_rw vs original WildChat; rewriting covered idmix rows moves them only to 0.532 vs original idmix. So the 0.99 is not HealthCareMagic surface style (greetings, stripped apostrophes, ".."). The character-n-gram classifier does use ".." and "..." as Dolma-side features, but it is not the best arm.
- Item 2 (format artefacts): top logistic features on the mix side are task words ("given", "input", "output", "answer", "function", "write", "each", "your response"); on the Dolma side function words of narrative prose ("we", "it", "was", "our", "and"). The two-sided word classifiers score "narrative prose, not task instruction", i.e. register.
- Register (exp03, added before the full run): every arm that is strong on legal vs WildChat also scores covered-topic Reddit posts high against WildChat (val-best arm: cooking 0.93, fitness 0.93, relationships 0.98, relationship_advice 0.98, legal 0.96) and ranks r/legaladvice below r/relationship_advice and r/relationships (0.38 / 0.40 for the val-best arm; 0.13-0.56 across the strong arms). This matches his activation result (0.296 / 0.195). No detector here, word or activation, separates "legal topic" from "personal-narrative Reddit post".
- Item 5 (label noise): see reading.

Reading (`out/full/read/`, 20 random rows per set plus top / bottom 20 of the val-best arm over legal + medical + WildChat; I read all of them for exp01 and the extremes for exp12):
- Random idfit / idmix rows are task-style: math word problems ("Math problem: ..."), code requests, persona prompts ("Dr. Smith, a renowned biosecurity expert ..."), instruction-following constraints, non-English (Sinhala, Persian, Malagasy, Portuguese), and harmful requests from the safety families. One idmix row is a medical RCT summarisation task (SciRIFF): the mix has medical text, but as tasks, not as first-person symptom questions.
- Random legal rows are r/legaladvice "title\n\nbody" posts, many with a jurisdiction tag ("[CA]", "[IL]", "LOCATION: USA, Delaware"). Random medical rows are first-person symptom descriptions, often lower-case with "I m", "..", "Hi Dr.".
- Surprising: the "English" WildChat pool is not all English (Chinese, Russian, Japanese, Portuguese rows appear; his logs note one), and it repeats templates: 4 of 20 random exp01 WildChat rows are the same "As a prompt generator for a generative AI called Midjourney" template. These templates are the lowest-scored WildChat rows (closest to the mix, whose WildChat-regenerated families contain them).
- Top 20 of the val-best arm on exp01: all 20 are medical symptom narratives. Bottom 20: all WildChat: Midjourney templates, short code questions, "Express your interest in the paper below" template, non-English code requests.
- Label noise: the highest-scored WildChat rows include medical questions ("What should I take after otoplasty? Ibuprofen or paracetamol", "I am going to take 800 mg of theanine ...", "can postmenopausal women have cervical ectropion") and tech-support one-liners. The lowest-scored medical rows are fragments whose content was lost in HealthCareMagic ("My Semen analysis report", "I had a stress test last sunday, below were the findings") and one Hinglish row. The lowest-scored legal rows are about blogs, software IP, Covid workplace rules and a hypothetical. So part of the remaining error is label noise in both directions.
- exp11 seed 0 top 100 of the val-best arm: 31 gap needles, 0 control needles, and non-needle rows that are themselves medical or legal ("please diagnose me ... swollen lymphnodes", "scratched my leg ... blood", a New York taxi-licence document) or tech support; 2% `NAME_` placeholders. His tf-idf arm's top 100 is mostly chit-chat and creative-writing LMSYS / WildChat prompts, 13% `NAME_`.

How conclusive / what is exploratory:
- The val-best arm and all its settings were chosen on validation pools before the test pools were scored, so its numbers are not exploratory. With n = 500 per class, the medical comparison against the reported activation numbers has non-overlapping intervals on exp12 (0.982 [0.961, 0.990] vs dknn_full 0.922 [0.875, 0.963], dmaha_full 0.919 [0.872, 0.957]); exp01 activation numbers have no logged intervals.
- "Best arm per task" on test (uni_nb on legal, mminilm_maha on medical) is a max over 23 arms; the val-best arm is within 0.009 on medical and 0.012 on legal, so the conclusions do not rest on the test selection.
- What could still overturn it: the activation numbers are from his log, not re-run, and the comparison is unpaired (no activation scores per row). The known-gap tasks are near ceiling for the best word arm (0.96-0.99), so they cannot show an activation advantage on medical any more. Every detector tested here reads register as much as topic (exp03), so "beats on legal / medical vs WildChat" is not evidence of coverage detection for either kind of detector.
