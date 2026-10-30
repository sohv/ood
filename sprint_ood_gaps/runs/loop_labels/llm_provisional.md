# Corrected loop's gap list: LLM-judged precision (provisional, 30 Oct 2026)

**These are LLM labels, a stand-in for the user's blind labels. They are not her read.** Update 30 Oct: she stopped at 14 of 50 topics (13 with a verdict); item 5 below compares them. Per-item LLM verdicts are in `llm/per_item.json`; this file gives arm-level numbers.

Design: `designs/gap_taxonomy.md` Part 18 and its amendment, both written before any result was scored. Scripts: `scripts/loop_labels_llm.py` (calls), `scripts/loop_labels_llm_score.py` (numbers), `scripts/loop_labels_llm_fig.py`. Outputs: `runs/loop_labels/llm/` (raw replies in `raw/<judge>/<id>.txt`, `summary.json`). Figure: `figs/loop_labels_llm_provisional.png`.

## Setup
- Items: the 50 topics on the user's label page (`runs/loop_labels/page_items.json`, taken from the page's embedded data). 30 are gaps from the corrected loop's R50 list and 20 are random topics ranked below the line (rank > 293). Each topic is one independent unit.
- Screen: the same screen the user sees. Each topic shows 10 prompts not verbatim in Dolci, and each prompt has up to 4 nearest Dolci prompts (2 by MiniLM, 2 by tf-idf), with the page's machine translations. The arm is hidden.
- Instructions: the page's help text, with the counting rule. For each prompt, is any shown Dolci prompt the same kind of request? 0-2 matches of 10 = real gap, 3-6 = partly, 7-10 = covered.
- Judges: Claude Sonnet 5.5 (`claude -p`) and codex Astra (gpt-6-astra). There was one call per topic, and 50 of 50 replies parsed for each judge.
- The primary verdict is each judge's own verdict. It matches the verdict recomputed from its per-prompt counts on 50 of 50 items for both judges.

## Results
1. The loop's gaps are judged "real gap" far more often than random topics are.
   - Sonnet: loop 13 of 30 = 0.43 (Wilson 95% 0.27 to 0.61); random 1 of 20 = 0.05 (0.01 to 0.24).
   - Astra: the same counts, 13 of 30 and 1 of 20.
   - Difference: 0.38 (Newcombe 95% 0.14 to 0.56), one-sided Fisher p = 0.003 for each judge.
   - The identical counts are a coincidence at the arm level. On the loop arm the judges agree on 23 of 30 topics, and both call 10 of them a real gap.
2. The step 7 bar is not met. The bar is a loop share of at least 0.50 and above random. The point estimate is 0.43, and the interval (0.27 to 0.61) includes 0.50.
3. Most of the rest is "partly", for both arms.
   - Real gap or partly: loop 28 of 30 = 0.93 (0.79 to 0.98) for both judges.
   - Random topics: 14 of 20 = 0.70 by Sonnet and 17 of 20 = 0.85 by Astra.
   - Differences: 0.23 (0.02 to 0.46), p = 0.036 by Sonnet; 0.08 (-0.09 to 0.30), p = 0.31 by Astra.
   - So "partly" does not separate the arms clearly. The separation is in "real gap".
4. The judges agree well enough for the LLM number to serve as a stand-in. The pre-set gate is kappa at least 0.4 on n = 50.
   - The four verdicts: 36 of 50 agree, kappa 0.50.
   - Real gap vs not: kappa 0.60.
5. Agreement with the user's labels is small and one-directional (updated 30 Oct with her final labels; `human_vs_llm.json`). She labelled 14 topics, 13 with a verdict; 9 by overall impression and 5 counted prompt by prompt, one of which replaced an earlier Can't tell.
   - Real gap vs not: Sonnet agrees on 8 of 13 and Astra on 8 of 13.
   - In all 5 disagreements she says "real gap" and the judge does not.
   - By arm: she calls 3 of 4 loop and 4 of 9 random topics real gaps; each judge calls 2 of 4 and 0 of 9 on the same topics.
   - So the judges' low random rate (1 of 20) is not confirmed by a person. At n = 13 this gives a direction, not a rate. She reports judging by "same job for the model", which is stricter than prompt shape; the judges read the page's wording, "same kind of request".
6. Exploratory, chosen after reading the notes: Astra's notes on its 13 loop "real gap" topics mention a template, NAME_k or a placeholder in 11 of them. This is a keyword match on the notes. Sonnet's notes mention them in 1 of 13, so this depends on what the judge chooses to write. It agrees with the earlier finding that the loop's list leans toward collection artefacts (part 17: 33 of 56 confirmed gaps flagged by three one-line rules).

## What this does and does not show
- What "real gap" measures: none of a prompt's 4 nearest Dolci prompts is the same kind of request. That is coverage by the nearest neighbours, not usefulness to a lab. It is also an upper bound, because a retrieval miss reads as "not covered". The loop arm was selected for far neighbours, so part of the loop-minus-random difference re-measures that selection. The user's page has the same bias. Slide wording: "not covered by its nearest Dolci prompts".
- Sonnet rated these topics in step 6, and the loop arm is the topics it called gaps, so Sonnet's loop share could be inflated. Astra did not take part in step 6 and gives the same 13 of 30, so no inflation shows here.
- The judges are stricter than the user on the 3 items where they differ. If that holds on her remaining labels, her loop share will be higher than 0.43.
- Most likely objection: "an LLM judging LLM-chosen gaps." It is addressed in part. Two model families agree (kappa 0.60), and one of them did not choose the gaps. Agreement with a person is only 5 of 8, so the number stays provisional until her 40 remaining labels are in.

Suggested slide line: "Precision, LLM judges only (provisional): 13 of 30 loop gaps (0.43, 0.27 to 0.61) are not covered by their nearest Dolci prompts, against 1 of 20 random topics; the same for two judge families; below the pre-set 0.5 bar."
