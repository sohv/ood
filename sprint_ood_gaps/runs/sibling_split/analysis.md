# Sibling split: does adding one sub-topic make its siblings look covered? (design: designs/sibling_split.md, fixed before the split)

Setup: 6 topics (3 from the gap score's top 293, 3 from distance's; chosen by me after reading members; NAME_k < 0.10, no templates),
each split by Claude Sonnet 5.5 (stripped claude -p, 54 calls, subscription) into 3 sibling sub-topics, every member assigned
(58 to 269 per sub-topic; definitions and 5 random members each in split_read.md). For each topic and ordered pair (add X, score Y),
36 pairs: X moved to Dolci (sibling), the same number of random outside traffic prompts moved (control), or half of Y moved and the
other half scored (self, upper reference). Distance = 1 - mean top-10 Dolci cosine (MiniLM, float32); ratio = k = 50 union-kNN.
Topic score = median over scored members. CPU only; no OpenRouter calls.

## Measured
- Control moves nothing: median change 0.000 (distance) and +0.003 (ratio).
- Pre-set measure, mean update share U over topics: distance 1.26 [0.50, 2.56], ratio 0.93 [0.64, 1.18]; both "updates" by the rule.
  This number is not usable: the design's validity check (self must move Y clearly) fails for half the pairs, and there U divides by
  about zero (1530 gives U 22.7 on one pair).
- Validity split: on the three distance-top topics (25 install X, 1646 troubleshooting, 1530 "X vs Y") moving HALF OF THE SAME
  SUB-TOPIC changes the other half's median distance by only -0.0003 to -0.023 (median -0.009; scores start near 0.44, line 0.392).
  On the three gap-score topics (1579, 956, 257) it changes by -0.04 to -0.39.
- Exploratory (validity rule chosen after looking: self change <= -0.03; 18 pairs, exactly the gap-score topics), U from summed
  changes per topic: distance 1579 0.75, 257 0.45, 956 0.30 (mean 0.50); ratio 0.84, 0.88, 0.27 (mean 0.66). Sub-topics below the
  distance line: 2 of 36 before, 10 after a sibling, 17 after self, 2 after control.
- Within 956, the sub-topic that is one role-play template (succubus prompts, 61) is neither covered by its siblings (distance change
  -0.002, -0.012) nor covers them, while its own half moves it by -0.39. Spearman of U with sub-topic centroid cosine: 0.38 (distance).

## Meaning
- Within a topic the gap score found (narrow subject, many similar prompts: Vicuna setup, Vicuna identity, sexual questions), both
  scores do update in Stefan's sense: adding one sibling makes the others look about half to three quarters as covered as adding
  their own data would, against no change for random data. Templates are the exception: they only cover themselves. This differs
  from exp42, which moved whole topics and re-scored other topics; siblings inside one topic are much closer in MiniLM space.
- The distance score's top topics behave differently: they are kinds of request ("how do I install X", "my device does Y", "what is
  the difference between X and Y") whose prompts are not near each other, so adding even the same kind of prompt barely makes new
  ones look covered. This is the same pattern as branch A's detector side (1,546 added prompts took 20 of 205 distance topics below
  the line) and fits branch A's behaviour result (data for those topics did not help them more than ordinary data). Distance flags
  prompts far from everything rather than missing clusters, so "add data for the gap, then the flag drops" does not apply to it.
Objections: 6 topics, chosen by me; one LLM split; one embedder (MiniLM, 128 tokens); the validity rule for the exploratory numbers
was chosen after looking; topic sizes limit sub-topics to 58-269 prompts; moving prompts is not retraining a model.
Files: scores.json (pre-set), exploratory.json, pairs.csv, split.json, split_read.md; figs/sibling_split/distance_change.png.
