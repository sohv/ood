# LLM-judge baseline: result (design `designs/judge_baseline.md`, decision rules and analysis script fixed before the full run)

Setup: the 90 blind-labelled topics, the same 10 prompts the user saw, cosine and arm hidden. Judges: `google/gemini-3.8-flash` and `anthropic/claude-sonnet-4.6` via OpenRouter, default sampling, one call per topic per arm (540 calls, 5 failed and kept as failed rows: Sonnet on topic tb5e42b4e in all 3 arms, Flash on 2 J-hyb calls). Cost $3.89. Score = the judge's 0-100 "this kind of request is missing". Truth = the user's real_gap vs not, on 81 topics (30 real_gap, 9 can't-tell excluded). Row bootstrap, 4,000 draws.

## Measured
| ranking | neighbours shown | AUROC vs user | minus distance (95% CI) | Spearman with distance |
|---|---|---|---|---|
| distance | (score) | 0.77 [0.66, 0.87] | | |
| ratio | (score) | 0.38 [0.26, 0.51] | -0.39 [-0.56, -0.22] | |
| Astra 3-way label | screen (cosine shown) | 0.81 [0.72, 0.88] | +0.04 [-0.07, 0.15] | |
| **Flash J-50** (primary) | 5 tf-idf per prompt | 0.63 [0.52, 0.74] | -0.14 [-0.29, 0.01] | 0.22 |
| Sonnet J-50 | 5 tf-idf per prompt | 0.73 [0.63, 0.83] | -0.04 [-0.17, 0.10] | 0.44 |
| Flash J-hyb | MiniLM top 5 + tf-idf top 5 | 0.55 [0.45, 0.65] | -0.22 [-0.35, -0.08] | 0.19 |
| Sonnet J-hyb | MiniLM top 5 + tf-idf top 5 | 0.78 [0.68, 0.87] | +0.01 [-0.11, 0.14] | 0.54 |
| Flash J-same | screen neighbour | 0.77 [0.66, 0.87] | 0.00 [-0.12, 0.13] | 0.49 |
| Sonnet J-same | screen neighbour | 0.88 [0.80, 0.95] | +0.11 [0.00, 0.23] | 0.51 |

- The judges' own 3-way labels carry almost no information. Sonnet said real_gap 0 times in 267 scored calls. Flash said it 2 to 6 times per arm. Kappa with the user is -0.06 to 0.05 (Astra 0.37). This repeats gap_blind_check. Only the 0-100 score ranks.
- Flash puts half or more of the topics at a score of 0 to 7.5 in every arm. Sonnet's median is 5 to 10.
- The replies I read call a topic covered when several of its prompts have exact or near-exact Dolci matches (e.g. git rebase or sentence-editing prompts inside a DevOps topic the user called real gap). The user judged whether the topic's kind of request is missing.

## Against the pre-set rules
- Tie rule (Flash J-50 >= distance, difference CI not below -0.05): not met.
- Distance-wins rule (Flash J-50 at least 0.12 below distance, CI excluding 0): point estimate -0.14, CI upper end +0.007. Not met, by a narrow margin.
- "J-same matches, J-50 does not": holds for both judges. Flash 0.77 vs 0.63, Sonnet 0.88 vs 0.73. By the design, the judge's agreement with the user depends on MiniLM's neighbours, and there is no step 2.
- Sonnet vs Flash: Sonnet is higher in every arm (by 0.10 to 0.23). The baseline depends on judge strength.

## Meaning
- With retrieval independent of the score (J-50), neither judge beats distance. Flash is probably worse; the interval just reaches 0. Sonnet is within noise of distance. A judge given the single MiniLM-nearest neighbour matches distance (Flash) or exceeds it (Sonnet, +0.11, CI lower end 0.00). That is a judge reading the score's own retrieval, and the same screen the user labelled from.
- Giving Flash more neighbours (J-hyb, about 9 per prompt) lowered its AUROC. Sonnet gained over J-50. Over long lists Flash finds some exact match for most topics and scores them 0.
- For the plan: no evidence that a cheap judge replaces the distance score as a detector. A strong judge on top of distance's own neighbour may rank gaps better than distance alone (exploratory; one arm, interval touching 0). That would be distance as retrieval plus a judge as a second-stage filter, not a judge alone.

## Objections
1. The test set was picked by the scores (two arms of 30 at the scores' top). This favours distance against the judges.
2. The user labelled from the J-same screen, with one MiniLM neighbour per prompt and the cosine shown. J-same agreement is partly two readers of the same screen (research-6d's caution). J-50 is the independent arm.
3. One human labeller; 81 topics. Only differences of about 0.12 or more resolve.
4. One call per judge per topic, default sampling. Sampling noise is not measured.
5. tf-idf retrieval misses obvious matches for some prompts (an `ALTER DATABASE ... GO` SQL prompt got "Go fogohlela seropeng").

Files: `runs/judge/raw_full.jsonl` (every request and reply), `inputs.jsonl`, `summary.json`, `figs/judge/auroc_vs_user_labels.png`; scripts `scripts/judge_build.py`, `judge_build_hybrid.py`, `judge_run.py`, `judge_analyse.py`.
