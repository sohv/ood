# Part 16 re-derived in new code (30 Oct 2026, research-d3)

The code was written from the input files and research-9e's one-paragraph brief, without reading `scripts/gap_grouping_*.py`. Script: `scripts/review_part16_rederive.py`. Output: `runs/review/part16_rederive.json`.

Inputs:
- `runs/gap_taxonomy/emergent/clusters.npz`: desc_fine_s0 and raw_fine_s0, each with 100 groups.
- `runs/gap_taxonomy/grouping_recovery/group_kinds.json`: which groups are of each kind.
- `runs/gap_taxonomy/removal_arms.json`: traffic ids, the minus_F arms, and the family of each Dolci id.

Shares are taken over the sample prompts that have a group: 1,992 traffic and 1,988 Dolci. The random removals are 20 per family, of the same size as the planted one (105, 397 or 99), drawn from Dolci prompts of the other families with seed 0. They are not the original randoms, so only means and ranges are compared.

**Result: every deterministic number matches `visible_check.json` and `planted_vs_natural.json` to the printed precision. The random means differ by at most 0.2 points, within the spread of the 20 randoms.**

Traffic share minus Dolci share of the kind's groups, in points (original values in brackets):

| Grouping / family | full Dolci | after removing the kind | 20 randoms: mean (range) |
|---|---|---|---|
| desc / safety | 9.78 (9.78) | 12.71 (12.71) | 9.59 (9.47 to 9.74) [9.66 (9.47 to 9.90)] |
| desc / code | -10.35 (-10.35) | 5.81 (5.81) | -14.50 (-15.16 to -13.84) [-14.30 (-15.04 to -13.72)] |
| desc / multilingual | 6.46 (6.46) | 10.09 (10.09) | 6.28 (6.01 to 6.54) [6.31 (6.18 to 6.44)] |
| raw / safety | 11.77 (11.8) | 14.09 (14.1) | 11.67 |
| raw / code | -14.53 (-14.5) | -0.85 (-0.9) | -18.05 |
| raw / multilingual | 10.46 (10.5) | 13.93 (13.9) | 10.27 |

The raw-grouping originals come from `visible_check.json`, rounded to 3 decimals; `planted_vs_natural.json` covers description groups only.

Also matching, for all six cells: traffic share of the kind's groups, Dolci share before and after removal, and the share of removed prompts that fall in the kind's groups. For example, desc/safety gives 0.596, 16.27%, 6.49% and 3.56%.

How many of the 5 groups with the largest drop in Dolci share are of the kind:

| | after removing the kind (orig.) | 20 randoms: median (orig. median), max |
|---|---|---|
| desc / safety | 3 (3) | 0 (0), 0 |
| desc / code | 5 (5) | 0 (0), 1 |
| desc / multilingual | 5 (5) | 0 (0), 2 |
| raw / safety | 3 (3) | 1 (0), 2 |
| raw / code | 5 (5) | 0.5 (0), 2 |
| raw / multilingual | 5 (5) | 0 (0), 2 |

Differences:
- For raw groups, the random median is 1 (safety) and 0.5 (code) here, against 0 in the original. Some single random removals put 2 of the kind's groups in the top 5. The planted count (3 or 5) is still above every random in all six cells. The difference comes from the random seeds; it is not a code difference.
- The main slide and B9 say "random median 0". That holds for description groups in both versions. For raw groups it depends on the seed, so the backup could say "random median 0 to 1, max 2".

The part 16 recovery rule and the set-aside variants are re-derived in the next section.

## Recovery rule, re-derived (item 10b)

Script: `scripts/review_part16_recovery.py`, written without reading the original. Output: `runs/review/part16_recovery.json`.
- Groups are ranked by traffic share minus Dolci share.
- Rule, as briefed: the minus_F top-10 count is at least the random median + 2, AND the best rank is strictly better than in at least 19 of 20 randoms.
- Set-aside drops prompts tagged placeholder_or_artefact or templated_dataset_item in `facets/tags.jsonl`. That is 1,492 of the 3,980 tagged prompts, taken from both sides.
- The randoms use the same seed-0 draws as above.

**Result: 0 of 12 cells recovered, the same as `grouping_recovery/summary.json`.** All the top-10 counts and best ranks for the minus arms match the original exactly. So do the random medians, except one best-rank median: 8 here against 7.5 in the original.

| Cell | minus top-10 / best rank | random top-10 median / best-rank median | randoms beaten on best rank (orig.) |
|---|---|---|---|
| desc safety | 1 / 6 | 1 / 5 | 0 (0) |
| desc code | 2 / 4 | 2 / 4 | 0 (1) |
| desc multilingual | 1 / 4 | 1 / 4 | 0 (0) |
| desc set-aside safety | 1 / 7 | 1 / 4 | 0 (0) |
| desc set-aside code | 2 / 3 | 1 / 3 | 0 (0) |
| desc set-aside multilingual | 2 / 3 | 2 / 3 | 0 (0) |
| raw safety | 3 / 1 | 3 / 1 | 0 (0) |
| raw code | 3 / 5 | 2 / 5 | 0 (1) |
| raw multilingual | 4 / 6 | 4 / 6 | 0 (0) |
| raw set-aside safety | 4 / 6 | 3 / 5 | 0 (0) |
| raw set-aside code | 2 / 5 | 1 / 8 (orig. 7.5) | **19 (18)** |
| raw set-aside multilingual | 5 / 3 | 4 / 3 | 0 (0) |

One difference matters for the wording.
- Raw-text groups with templates set aside, for code, beat 18 of 20 randoms on best rank in the original and 19 of 20 here. So the best-rank half of the rule is met under this seed.
- The cell still fails on the top-10 half: 2 code groups against a random median of 1, where the rule needs at least 3.
- So "beats 18 of 20, the rule needed 19" (interview_qa Q15 and Q19, B9) depends on the seed. A seed-independent statement: "it fails the top-10 part of the rule (2 vs the 3 needed); on best rank it beats 18 or 19 of 20 randoms, depending on the draw."

The other gaps between versions (desc code and raw code: 1 random beaten in the original, 0 here) are also differences between random draws.

## Part 17 heuristic baseline, re-derived (item 11)

Script: `scripts/review_part17_rederive.py`, written from the design's Part 17 text without reading `scripts/gap_heuristic.py`. Output: `runs/review/part17_rederive.json`. The three rules are computed from the raw traffic text, not from part 6's per-prompt columns:
- placeholder: regex `NAME_\d`;
- non-English: py3langid on the first 1,000 characters is not "en";
- template: the lowercased, whitespace-collapsed 60-character prefix is shared by at least 10 traffic prompts.

Members are each topic's traffic prompts whose text is not in Dolci. A topic is flagged by a rule when at least half its members match it.

**Result: identical to `runs/gap_taxonomy/heuristic/summary.json` and `topics.csv`.**
- The same 56 / 144 / 56 topics. My seed-0 draw from the 1,942 unconfirmed topics with at least 50 prompts gives the same random set.
- Every per-topic flag agrees with the original.
- Counts:
  - confirmed: any rule 33 (placeholder 24, template 12, non-English 3);
  - random: 8 (3, 4, 3);
  - rejected: 24 (20, 4, 2).
- Fisher p: confirmed vs random 1.45e-6; confirmed vs rejected 1.45e-8.

One note for slide 7: the heuristic uses all 144 rejected topics, while the defining-axis figure on the same slide uses 143 (the relabel summary `runs/gap_taxonomy/summary.json` counts 143 rejected; 335 of 338 units were labelled, and I did not check which ones are missing). Both numbers are right for their own analysis.

## Part 15 property ranks, re-derived and tested against 20 random removals (item 12)

Script: `scripts/review_part15_null20.py`. Output: `runs/review/part15_null20.json`. This re-derivation is not fully independent: I read lines 10-33 of `scripts/gap_setaside.py` earlier, while checking a count.
- Features: the 21 part 8 properties with Sole kappa at least 0.4, plus task type. There are 37 features (35 with templated and placeholder prompts set aside).
- Features are ranked by traffic share minus Dolci share.
- Best rank of the family's features: safety = harmful_request or adversarial_pressure; code = task=coding or agentic_or_tool_use; multilingual = non_english.
- The 20 random removals are the same size as the planted one, drawn from other families (seed 0).

**Re-derivation: identical to `runs/gap_taxonomy/setaside/summary.json`** for full, minus and the stored single random, in all six cells:
- 8 / 12 / 12, 6 / 31 / 31, 2 / 3 / 3;
- set aside: 8 / 19 / 19, 5 / 30 / 31, 1 / 1 / 1.

Against 20 random removals:

| Cell | full | minus (planted) | 20 randoms: best rank range (median) | randoms the planted rank beats | kind features in top 10: planted vs random median |
|---|---|---|---|---|---|
| all / safety | 12 | 8 | 12-13 (12) | 20 of 20 | 1 vs 0 |
| all / code | 31 | 6 | 31-31 (31) | 20 of 20 | 1 vs 0 |
| all / multilingual | 3 | 2 | 3-3 (3) | 20 of 20 | 1 vs 1 |
| set aside / safety | 19 | 8 | 19-20 (19) | 20 of 20 | 1 vs 0 |
| set aside / code | 30 | 5 | 30-31 (30) | 20 of 20 | 1 vs 0 |
| set aside / multilingual | 1 | 1 | 1-2 (1) | 1 of 20 (19 tied) | 1 vs 1 |

What this shows:
- "Code 31 -> 6" survives a 20-random null: every random removal leaves code at rank 30 or 31.
- Safety does the same, 12 -> 8 (randoms 12-13), and 19 -> 8 with templates set aside. In both, a safety property enters the top 10, where no random puts one.
- Non-English is already at rank 1-3, so a removal cannot move it far. It moves one rank (3 -> 2), which beats all 20 randoms, but the random ranks hardly vary.
- The random ranks hardly vary in any cell. So "beats 20 of 20" here shows the shift is not chance; it does not show the shift is large.
- No recovery rule was set before running for properties. Part 16's grouping rule also asks for at least 2 more kind units in the top 10 than the random median. With only 1-2 features per kind, that is nearly impossible here: each family gains 1.

Consequence for slide 9 and B9: "the one exception is property tags for code, against one random removal" is out of date. Suggested wording: "Per-prompt property tags move code from rank 31 to 6 and safety from 12 to 8; all 20 random removals leave them at 30-31 and 12-13. Non-English is already near the top. No threshold was set before running for this measure."

How this fits claim 2 is a reading, not a measurement. Per-prompt tags respond to a planted removal more cleanly than groupings do (part 16, 0 of 12). The planted kinds still do not reach the top of the list: safety at 8 and code at 6 sit below natural differences: creative writing, placeholder or truncated text, non-English and roleplay rank above them in both arms (checked on the minus_code and minus_safety rankings).
