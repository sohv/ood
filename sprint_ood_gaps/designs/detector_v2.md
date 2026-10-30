# Gap detector v2 (ood-03, written 2026-10-24 before any v2 score is computed; frozen before the user labels the 42 judge_best topics)

Goal: a cheap, non-LLM topic score that ranks real gaps better than distance v1, built from today's findings, and frozen before the 42 new blind labels exist, so those labels are a held-out test. CPU only, no paid calls.

Findings it uses:
- Exact Dolci copies are masked by the v1 search, so a prompt that is in Dolci word for word can still look far. The review session's lexical check found 10 of 30 list entries covered by copies.
- MiniLM misses lexical matches that BM25 finds (judge_best retrieval).
- NAME_k and template-dominated topics are dataset artefacts or one repeated prompt, not kinds of request.
- The judge's coverage call ties distance v1 on held-out labels; the two use partly different signals (Spearman 0.51-0.56).

## Variants, each a score for all 1,998 topics (>= 50 members); higher = more likely a gap
- v1: distance v1 as used so far (median over all members of 1 - mean MiniLM cosine to the 10 nearest Dolci prompts; `topics.csv` median_score; checked against one_sided.npz on 4 topics). Baseline.
- a (verbatim): v1's formula over all members, but a prompt that is an exact Dolci member (traffic_hash in mix_hash) has distance 0.
- b (two retrievers): on the topic's 10 sampled prompts (`runs/judge_best/topics.json`, the prompts the judge saw), per prompt sim = 1 if an exact Dolci member, else max(MiniLM top-1 cosine, best tf-idf cosine to its 5 BM25 hits); topic score = 1 - median sim. The tf-idf vectoriser (word 1-2 grams, sublinear tf, min_df 3) is fit on 300,000 random Dolci first turns (seed 0); Dolci texts are cut to 2,000 characters.
- c (artefact filter): on 20 random members (seed 0), a topic is flagged if at least half contain NAME_k (`NAME_\d`), or at least half of member pairs are near-copies (60-character shared prefix, or difflib ratio >= 0.8 on the first 300 characters). Flagged topics are ranked below all unflagged ones, keeping their order. Applied to b, giving b+c.
- d (two-stage): stage 1 is b+c; its top 300 topics are re-ranked by the judge's primary gap score (`runs/judge_best/primary_scores.jsonl`), and all other topics follow in stage-1 order.

**Primary: b+c** (the cheap detector; it answers whether a non-LLM score can match the judge). **Main secondary: d** (the practical pipeline). a and b alone are ablations. Chosen before computing any of them.

## Comparisons
On the 42 new labels (21 judge-only + 21 distance-v1-only topics from judge_best; user, blind; "can't tell" excluded), with real gap vs not:
1. AUROC of b+c vs v1 and vs judge alone (paired bootstrap of the difference, 4,000 draws, seed 0). The same for d, a and b.
2. Precision of each variant's top 30 on the labelled topics among them, with the number labelled. The 9 topics in both v1 and judge top 30 are not labelled, so the counts are partial; reported as such.

Secondary: AUROC on the 81 old labels (rare_gaps; those labels came from MiniLM-nearest screens, which favours v1 and a).

Reading, fixed now:
- b+c "better than v1": AUROC on the 42 at least v1 + 0.10, CI of the difference above 0.
- b+c "matches the judge": AUROC on the 42 >= judge - 0.05.
- With 42 topics, differences under about 0.15 will not resolve. They are reported as inconclusive.
- The 42 topics were selected as the top of v1 and of the judge, so AUROC within them is not population AUROC. Both v1 and the judge are evaluated on topics they ranked high. That favours neither by construction, but v2 inherits v1's candidates through its MiniLM part.

## Freeze
All variant scores, flags and top-30 lists are written to `runs/detector_v2/` (scores.csv, flags.csv, top30.json, sha256 of each file in FROZEN.txt) before any new label is read. No change after that except bug fixes, which would be reported as such.

Script: `scripts/detector_v2.py`.

Amendment (2026-10-24 ~23:10 UTC, after freeze, before any label existed; label store checked empty): the 18 topics in v2 (b+c)'s frozen top 30 that were neither on the label page nor among the 9 unlabelled judge/v1 shared topics were added to the same page (arm v2_only; `scripts/judge_best_labels_add_v2.py`), 60 topics in all. 27 of v2's top 30 will be labelled; 3 are in the shared 9. AUROC stays on the original 42 (judge_only + distance_only arms), as designed. Top-30 precision uses every labelled topic, so v2's precision now rests on 27 of its 30. The eval script was changed for this (its top-30 numbers in the secondary_81 block mix label sets; read only the AUROC there) and re-hashed in FROZEN.txt. Variant scores are unchanged.

Amendment 2 (2026-10-25, user's choice before most labels existed; 2 labels saved, one judge_only and one v2_only, not read): the 21 distance-v1-only topics were removed from the label page, leaving 39 topics (21 judge-only + 18 v2-only). The primary AUROC set becomes these 39. v1 is evaluated on its ranking of the 39 and on the old 81 labels, not on the precision of its own top 30, which is no longer labelled. The "b+c better than v1" bar is applied to AUROC on the 39. Precision of the judge's and v2's top 30 uses the labelled part (21 of the judge's 30, 27 of v2's 30). The eval script was changed accordingly and re-hashed.
