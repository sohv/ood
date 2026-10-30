# Branch A fix2: does training data for distance-flagged topics help those topics more than ordinary data? (design: designs/fix2_gap_quality.md, "Branch A run", fixed before data)

Setup: Olmo-3-7B-Instruct-DPO (rev b33130b7, bf16). Two LoRA adapters, Sohan's exp27 recipe unchanged (rank 16, lr 1e-4, 2 epochs), seed 0,
one seed each. Training targets: Olmo-3.1-32B-Instruct (sha ac0587e4) answers, T 0.6, 1,024 tokens, truncated replies and keyword
refusals dropped in both arms, then cut to 1,546 examples per arm.
- gap-data adapter: prompts from the 210 distance_only topics (10 drawn per topic before filtering).
- covered-data adapter: prompts from covered topics, matched one-to-one on source and Olmo length band.
Held-out (never trained, cosine < 0.90 to every training prompt of their arm, not in the behaviour pool): 1,050 distance-topic prompts
(5 per topic) and 525 matched covered prompts (105 topics). 1 reply per prompt per model (T 0.6, 2,048 tokens). Strict grader: Sole
(openai/gpt-6.1-sol, scripts/strict_quality.py rubric, 0.94 agreement with Astra on 78 replies); strict poor = harmless, not refused,
task not done well. 4,725 calls, 1 failed.

## Measured
Pre-check (before training): teacher strict poor 0.608 vs base 0.780 on 209 training prompts, -0.172 [-0.230, -0.115] (bar -0.10: passed).
Teacher truncation at 1,024 tokens 27%; dropped from training: fix 434 truncated + 120 refused of 2,100, control 442 + 73.
Training loss start -> end: gap-data 1.06 -> 0.88, covered-data 1.03 -> 0.82 (388 steps each).

| model | distance-topic prompts | covered prompts |
|---|---|---|
| untrained | 0.774 | 0.739 |
| covered-data adapter | 0.739 | 0.667 |
| gap-data adapter | 0.738 | 0.680 |

Paired differences, per topic (prompts averaged first), topic bootstrap 95% CI:
- Primary, gap-data minus covered-data adapter on distance prompts: -0.001 [-0.022, +0.020], 210 topics, 1,050 prompts. Truncated replies
  dropped: -0.001 [-0.023, +0.021]. Pre-set bar: "no extra help" (CI lower end above -0.03).
- Each adapter vs untrained on distance prompts: gap-data -0.036 [-0.059, -0.013], covered-data -0.035 [-0.058, -0.013].
- On covered prompts: gap-data minus covered-data +0.013 [-0.015, +0.044]; vs untrained -0.057 and -0.071.
- Interaction (gap-data minus covered-data on distance prompts, minus the same on their matched covered prompts): -0.006 [-0.046, +0.032], 105 topics.
- Measured paired per-prompt SD 0.345, correlation between the two adapters' grades 0.69 (the design guessed 0.3), so the CI half-width
  is 0.021, narrower than the planned 0.033.

Harmful-request guard (exp49_v2 refusal, 2 replies): distance-topic harmful members (24 prompts, n 46 graded replies) base 0.89, covered-data
0.85, gap-data 0.87; HarmBench 0.99 / 1.00 / 1.00; JBB 0.95 / 0.97 / 0.97. All pass (gap-data >= base - 0.10). Held-out distance prompts the
strict grader calls harmful (16 prompts, 15-16 replies per model): refused 0.67 / 0.81 / 0.73.

Detector side (CPU, runs/fix2a/detector.json): moving the 1,546 gap-data training prompts into Dolci takes 20 of 205 distance topics
below the top-293 line (median topic distance 0.404 -> 0.402); moving the covered-data prompts takes none. These topics are not above
the gap-score ratio line before or after (1 of 210).

## Meaning
With the teacher fixed, 1,546 examples drawn from the distance-flagged topics improve held-out prompts from those topics no more than
1,546 ordinary examples do. Both adapters lower the strict poor rate by about 0.035 on distance prompts and 0.06-0.07 on covered prompts:
the gain comes from distilling the larger model's answers, not from covering the topic. The CI rules out a targeted benefit larger than
about 0.02. The untrained deficit on these topics (0.035 here, 0.043 in the behaviour run) is not closed by either adapter.
Most likely objections: (1) the strict grade is binary and near saturation (~0.7), so a change in how bad an answer is goes unseen;
(2) ~7 examples per topic may be too few to teach topic content, as the detector side also shows (the added prompts move few topics);
(3) the teacher's own poor rate is 0.61, so it can only teach so much; (4) one seed and one LoRA setting.
Files: graded/summary.json, graded/eval_rows.jsonl (every grade with the judge's reason), graded/precheck.json, graded/guard.json,
detector.json; replies and adapters in runs/pod_sync/oodgaps-fix2/; figs/fix2/branchA_strict_poor.png.
Cost: Runpod H200 20:28-22:09 (~$8.90 by rate; billing API showed $3.39 at 22:10, lagging); OpenRouter key usage 99.80 -> 112.45 over
the grading window (includes any other session's calls on the key).
