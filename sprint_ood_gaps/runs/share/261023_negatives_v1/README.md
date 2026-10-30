# 261023_negatives_v1

Negatives eval for the exp37 gap score, the float16 check, and a one-sided distance-to-Dolci pass over the exp37 traffic pool. All scores are your exp37 k = 50 neighbour ratio, rebuilt on CPU from your files (traffic 992,933, mix 1,710,578).

- `designs`: `negatives_eval.md` (written before scoring; bars fixed there).
- `report.md`, `per_set.csv`: one row per eval set (23 unseen same-kind datasets, 37 Dolci source samples, planted legal and medical). `scores/`: every scored prompt.
- `auroc.md`: planted vs negatives for the ratio, distance to nearest Dolci prompt, a tf-idf traffic-vs-mix classifier and Olmo token length, pooled and within length bands.
- `validation.json`, `fp16_check.json`: the rebuild matches your ratio.npy on 19,981 of 20,000 rows when similarities are rounded to float16 as on your GPU; `topic_precision_*`: at float32, 290 of your 293 gap topics stay above the line.
- `wildchat_builtin_negative.md`: your saved ratios split by WildChat member / non-member / LMSYS.
- `rare_gaps/`: one-sided distance for all traffic (Colab A100, float32); topics ranked by median distance; `items.json` is the blind labelling set (30 distance-only topics + 30 random); the arm key is added after labelling.
- `scripts/`: the code that produced all of the above.
