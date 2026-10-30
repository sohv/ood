# Register vs coverage test for the distance score (written 2026-10-23 22:40 UTC, before scoring)

Question: does distance to the nearest Dolci prompts (1 - mean top-10 MiniLM cosine) flag the planted legal and medical prompts because Dolci lacks those topics, or because they are written as personal narratives, a register Dolci's task-style prompts lack?

Sets (all Sohan's existing pools, scored with the same scorer as the negatives run; ratio and baselines recorded too):
- Same register, covered topics: Reddit posts from r/askscience, r/cooking, r/fitness, r/learnprogramming, r/relationship_advice, r/relationships, r/writing (register_vs_coverage, 500 each).
- Same register, topic assumed uncovered, second sample: r/legaladvice posts (register_vs_coverage legal_rtb, 500).
- Uncovered topic, Dolci-like register: GPT-4.1 rewrites of the planted legal and medical prompts into the style of 8 Dolci prompts (legal_medical_restyled legal_rw, medical_rw, 500 each; his judge checked the request was kept).
- Rewrite control: WildChat prompts rewritten the same way (legal_medical_restyled wildchat_rw, 500).

Predictions:
- Coverage reading: legal and medical score further from Dolci than every covered-topic Reddit set (AUROC well above 0.5), and the rewrites stay far: legal_rw and medical_rw above wildchat_rw and above the 23 negatives.
- Register reading: covered-topic Reddit sets score as far as legal (AUROC near 0.5), and the rewrites fall to the level of wildchat_rw.

Reported: AUROC within Olmo-length bands, for distance, ratio, tf-idf classifier, length, for each comparison above. No bar; the two readings are compared on the pattern across sets.

Caveat fixed in advance: "covered topic" is an assumption for the Reddit sets. Dolci contains science, cooking, fitness, programming and some relationship-advice prompts, but not necessarily in narrative form. A topic census of Dolci (keyword count) is reported next to each set so the reader can judge.
Prior: in `notes/word_baselines_2026-10-21.md`, every word and activation detector scored covered-topic Reddit narratives as high as legal (relationships 0.98 vs WildChat). That predicts the register reading.

Deviation (recorded after the run): the analysis used 1 - nearest-one cosine, not the mean of ten named above, to match the negatives eval. Topic rankings by the two agree at Spearman 0.96 on the traffic pool.
