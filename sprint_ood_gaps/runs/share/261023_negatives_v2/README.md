# 261023_negatives_v2

Replaces 261023_negatives_v1. Two fixes, both found by outside review: (1) exact Dolci members are now removed from the negative sets, as the design required (1,136 of 20,500 rows, mostly CodeFeedback and oasst2); (2) Dolci anchor prompts now mask their own index; 1,104 of them had matched themselves because text was cut to 4,000 characters before hashing. Neither fix changes the headline numbers by more than 0.01 AUROC or 0.03 in a median ratio.

All scores use the pipeline_million k = 50 neighbour ratio rebuilt from your files (traffic 992,933, mix 1,710,578) and the distance to the nearest Dolci prompt (1 - top-1 MiniLM cosine).

- Negatives eval: `designs/negatives_eval.md`, `report.md`, `per_set.csv`, `auroc.md`, `scores/`.
- Register test: `designs/register_test.md`, `register_auroc.md`, `register_sets.csv`, `register_census.json`, `scores_register/`. Covered-topic Reddit posts score as far from Dolci as the planted legal/medical prompts.
- Leave-one-source-out: `designs/leave_source_out.md`, `leave_source_out/summary.csv` (per removed Dolci source; 5 of 9 removals open a gap), `combined_exploratory.csv`.
- Float16 check: `validation.json`, `fp16_check.json`, `topic_precision_summary.json`.
- WildChat split of your saved ratios: `wildchat_builtin_negative.md`.
- Distance on all traffic and the blind labelling items (90 topics: 30 distance-only, 30 ratio-only, 30 random): `rare_gaps/`. The arm key is added after labelling.
- Code: `scripts/`.
- Added later the same evening: `leave_source_out/tfidf.csv` (bag-of-words traffic-vs-Dolci classifier retrained per removal), `leave_source_out_lmsys/summary.csv` (ratio with LMSYS-only traffic), `designs/label_analysis.md` (analysis plan for the blind labels, fixed before labelling).
