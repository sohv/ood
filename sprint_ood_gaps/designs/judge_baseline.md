# LLM-judge baseline for gap detection (ood-03, written 2026-10-24 before any judge call)

Status: prep only. No paid call until research-6d says go (Sohan may own this).

## Question
Stefan's bar: an OOD score is worth having only if it adds something an LLM judge would not. Can a cheap judge, shown a topic's prompts and Dolci prompts retrieved for them, say which topics Dolci lacks as well as the distance score does? If yes, can it find gaps the distance score does not rank high?

## What exists already (CPU, computed now from saved files)
On the 81 topics the user labelled (30 real_gap; `runs/rare_gaps/labels/`), AUROC against user real_gap vs not, row bootstrap:
- distance (topic median, `topics.csv` median_score): 0.77 [0.66, 0.87]
- ratio (median_his): 0.38 [0.26, 0.51]. Below 0.5 because the arms were chosen by these scores (see objection 1).
- Astra (GPT-6) 3-way label on the same screens, ordinal: 0.81 [0.72, 0.88]. Distance minus Astra -0.03 [-0.15, 0.08].

So a frontier judge reading the user's screens already ties distance. Those screens were built from distance's own retrieval (MiniLM nearest) and show the cosine, so Astra partly read the score. Sohan's exp21 (GPT-4.1, 5 neighbours, 3-way, "missing" only) failed validity: it never said MISSING, because Dolci has scattered partial matches for everything. This design differs from exp21 in three ways: a graded output (no MISSING threshold), the user's labels as the validity check, and two retrievals.

## Arms (one run, 90 topics each)
Inputs per topic: the same 10 traffic prompts the user saw (`items.json`), same truncation. No cosine shown, no arm.
- J-same: each prompt with its 1 MiniLM-nearest Dolci prompt, as on the user's screen. Tests whether a cheap judge reproduces the human reading.
- J-50: each prompt with its 5 nearest Dolci prompts by tf-idf word similarity (50 per topic, Stefan's "50 training samples"; exp21's tf-idf retrieval), same exact-copy rule as the screens (exact copies kept: 32 of 900 screen pairs have cosine >= 0.999). Retrieval does not use MiniLM, so the judge's call does not depend on distance's neighbours.
Judges: Gemini Flash (Stefan's suggestion; current Flash on OpenRouter, version pinned in the output) and Claude Sonnet 4.6 (Sohan's production default). The judge gets the user's label question ("are the training prompts the same kind of request?"), gives its reason first, then a 0-100 score for "this kind of request is missing from the training sample", then real_gap / partly / covered. One call per topic per arm per model: 360 calls.

Amendment (2026-10-24, after the credit stop; no score from the full run analysed, only the 5-topic Flash pilot read): a third arm, J-hyb. For each prompt, its MiniLM top 5 (saved neighbours in `runs/rare_gaps/one_sided.npz`) plus its tf-idf top 5, duplicates removed, so up to 10 per prompt. Reason: tf-idf retrieval misses obvious matches for some prompts (an `ALTER DATABASE ... GO` SQL prompt got "Go fogohlela seropeng"). J-hyb asks whether J-50's result is limited by retrieval. It uses MiniLM's neighbours, so like J-same it is not independent of the distance score. J-50 stays the primary arm. Same two judges; 180 more calls.

## Measures
Primary: AUROC of the judge's 0-100 score against user real_gap on the 81 topics; paired bootstrap of judge minus distance. Secondary: the same against Astra's labels; kappa of the 3-way label with the user (Astra reached 0.37); the judge's real_gap share per arm, as in `analysis.md`.

Power: 30 positives and 51 negatives give AUROC intervals of about ±0.10 and a paired difference interval of about ±0.11 (Astra comparison above). Only differences of about 0.12 or more resolve. Smaller ones are reported as inconclusive.

## What would change the plan
- J-50 Flash AUROC >= distance's (point estimate), with the difference CI not below -0.05: a cheap judge with retrieval independent of MiniLM matches the score on its home ground. This set favours distance (two arms were picked on it), so a tie favours the judge. Then: (a) the Neel/Stefan story changes to "distance is a cheap prefilter; a judge ranks as well" unless step 2 shows otherwise; (b) step 2 (needs a second go): Flash J-50 on all 1,998 topics, compare its top 293 with distance's, and the user labels a blind sample of 30 judge-only vs 30 distance-only topics with the same screens as before.
- J-50 Flash below distance by >= 0.12 with the CI excluding 0: distance beats a cheap judge at equal inputs. It goes in the update as a baseline beaten. Stop; no step 2.
- J-same matches the user but J-50 does not: the judge needs MiniLM's neighbours, so the judge is not independent of the score. Report that; no step 2.
- Sonnet matches the user and Flash does not: the baseline depends on judge strength. Report both, and do step 2 with Sonnet only if research-6d agrees.

## Checks that the implementation measures this
- Assert the 10 prompts per topic and their order equal `items.json`. Assert the J-same neighbours equal the screen neighbours.
- Before the 360 calls: 5 random topics x both arms on Flash. Read the prompts sent, the retrieved neighbours and the replies. Check the score parses and that no input exceeds the context limit.
- Count parse failures and nulls per arm, and keep them as failed rows. Raw requests and replies go in `runs/judge/raw.jsonl`, then to HF `detectood/ood` `results/raw/261024_judge_baseline_v1/`.
- Plot: AUROC per arm and model with distance, ratio and Astra marked, `figs/judge/auroc_vs_user_labels.png`.

## Main objections
1. The test set was picked by the scores (30 distance top, 30 ratio top, 30 random). AUROC on it is not population AUROC, and it favours distance. Step 2 answers this.
2. The user's labels came from MiniLM-nearest screens, so the ground truth itself leans toward distance. J-50 changes only the judge's inputs, not the labels.
3. One human labeller.

## Cost
About 6k input tokens per J-50 call (10 prompts + 50 neighbours at 300 characters) and under 1k for J-same; about 1.2M tokens in all. At list prices as I recall them (Flash about $0.3/M, Sonnet $3/M input), under $5. Step 2 is about 12M Flash tokens, under $10. CPU: tf-idf over 1.71M Dolci first turns, under 1 hour on this box (estimate).
