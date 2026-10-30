# Strict quality grading on distance topics: result (design `designs/judge_strict_quality.md`, fixed before any strict grade)

Setup: Olmo-3-7B-Instruct-DPO replies from behaviour_distance (no new generation). All 4,200 distance_only replies (210 topics, 5 member prompts + 5 matched controls per topic, 2 replies each) and 1,200 random-arm replies (60 topics). One call per reply to `openai/gpt-6.1-sol` (Sole) via OpenRouter, with exp49_v2's four questions plus a strictness line. 0 failed calls. $16.96 (from 6.52M input and 0.39M output tokens). Strict poor = harmless, not refused, task not done well.

Calibration (step 1, 78 graderck replies, Astra's answers as reference): Sole agrees on strict poor 0.94, Sonnet 4.6 0.72. Of the 20 replies exp49 called fine, Astra calls 11 poorly done, Sole 15, Sonnet 2. Sole passed the 0.85 bar. `calibration/summary.json`.

## Measured
Member minus control, mean over topics, topic bootstrap 95% CI:
| arm | grader | member | control | effect |
|---|---|---|---|---|
| distance_only (210) | exp49 | 0.451 | 0.408 | +0.043 [0.003, 0.085] |
| distance_only (210) | strict | 0.765 | 0.722 | +0.043 [0.003, 0.082] |
| distance_only, truncated replies dropped | strict | 0.755 | 0.710 | +0.046 [0.004, 0.086] |
| distance_only, Dolci-member prompts and their controls dropped (208) | strict | 0.771 | 0.716 | +0.054 [0.009, 0.100] |
| random (60) | exp49 | 0.372 | 0.372 | -0.000 [-0.081, 0.078] |
| random (60) | strict | 0.708 | 0.688 | +0.020 [-0.058, 0.100] |

- Per-topic SD of the strict difference is 0.29 (exp49 0.30), as assumed when sizing.
- The strict grader finds poor answers that exp49 missed equally often for members (875 replies) and controls (870 replies). Of exp49's poor calls, it rejects 14 member and 20 control replies.
- 10 random strict grades read (5 member, 5 control). The reasons name specific errors (e.g. wrong WTO founding date and Super Bowl year, invented PAN-OS commands, an invented "Alphahold2" tool). 9 of 10 were graded poor, including 4 of 5 covered-topic controls.

## Against the pre-set criterion
- Criterion (effect >= 0.05, CI lower end > 0, above random): not met. The effect is 0.043, below 0.05. It is above random's 0.020, but random's interval is wide.
- Random null in [-0.05, 0.05]: yes (0.020).
- The variant without Dolci-member prompts gives 0.054 [0.009, 0.100]. It is a secondary measure named in the design, not the criterion.

## Meaning
- Under a strict standard, Olmo-3-7B answers about 72% of harmless covered-topic requests with at least one substantive error, and about 76% on distance topics. The difference is small: about 4 points, CI excluding 0, below the 0.05 bar set as the size that matters.
- The exp49 effect is not an artefact of exp49 under-counting poor answers. A stricter grader that finds about 1,750 more poor replies gives the same effect, and its extra finds split evenly between members and controls. That settles the open question from the grader check (miss rate per role).
- For the plan, by the pre-set reading: no quality gap of 0.05 or more on distance topics, and refusals are flat there. The fix effort stays on gap-score topics. A small quality deficit of about 0.04 to 0.05 is probably real. Fixing a 4-point difference against a 72% base error rate is not a clear target.

## Objections
1. One strict grader (Sole), calibrated against one reader (Astra) on 78 replies. Its strictness was checked against the reply text in 1 reply (graderck) plus the 10 read here, not against ground truth.
2. At a 72 to 76% base rate the strict measure is near saturation. Differences in how bad an answer is (one error vs many) are not counted. A pairwise judge (ood-04's fix2 design) would see severity; this per-reply grade cannot.
3. Random arm: 60 topics, so its CI is wide. "Above random" is weak.
4. Controls are matched on source and length band, not on task type. Distance topics may hold harder tasks.

Files: `raw_gpt-6.1-sol.jsonl` (every request, reply and grade), `summary_gpt-6.1-sol.json`, `calibration/`, `figs/judge/strict_poor_effect.png`; scripts `scripts/strict_quality.py`, `strict_quality_fig.py`.
