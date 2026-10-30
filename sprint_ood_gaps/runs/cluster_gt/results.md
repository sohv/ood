# Cluster ground truth (design: designs/cluster_gt.md)

Every Dolci first turn (1,710,578) assigned to the nearest of the 2,000 pipeline_million traffic-topic centroids, separately with MiniLM and with Qwen3-Embedding-0.6B vectors. Normalised Dolci share per topic s = (d/N_dolci) / (d/N_dolci + t/N_traffic); traffic-only if s < 0.05 (primary; 0.02 and 0.10 as checks). 1,998 topics with >= 50 traffic members. Files: `topic_counts.csv`, `topics_gt.csv`, `eval.csv`, `summary.json`, `figs/cluster_gt/detectors_vs_cluster_gt.png`. Scripts `scripts/cluster_gt_assign.py`, `scripts/cluster_gt_eval.py`.

## Measured
- Traffic-only topics at s < 0.05: 133 (MiniLM) and 119 (Qwen); 97 under both. Topic-level Spearman of the share between embedders 0.77.
- AUROC traffic-only vs rest, MiniLM truth / Qwen truth, primary cut (95% bootstrap over topics in `eval.csv`):
  - gap score (Sohan's ratio): 0.983 / 0.979; precision@100 0.71 / 0.63
  - distance v1 (MiniLM): 0.654 / 0.617; precision@100 0.26 / 0.21
  - distance (Qwen): 0.513 / 0.574
  - judge (100 - coverage): 0.512 / 0.498; precision@100 0.27 / 0.24
  - detector v2 b_c: 0.290 / 0.242; two_stage 0.289 / 0.244 (below 0.5: it ranks traffic-only topics low)
- The order is the same at cuts 0.02 and 0.10 and under both truths (gap score 0.96 to 0.99; distance 0.60 to 0.66; judge 0.43 to 0.53; v2 0.23 to 0.33).
- All 97 topics traffic-only under both truths are in Sohan's top 293; 32 are in distance's top 293. Of 12 read at random (one prompt each), most are LMSYS templates or NAME_k prompts (the "say something toxic: [your answer]" template, "introduction over 200 words for <company>", plant-seed climate templates, NPC role-play with NAME_1, jailbreak "Developer Mode").

## Meaning
This ground truth measures how much more often a kind of prompt occurs in traffic than in Dolci. That is what the gap score is built to estimate (ratio of traffic to Dolci neighbours), at a coarser grain, so its 0.98 is close to agreement by construction, under either embedder. Distance asks a different question (is any Dolci prompt close at all) and agrees only weakly. The judge (asked whether Dolci covers the topic) and detector v2 do not track this kind of under-representation; detector v2 ranks such topics low.

Circularity, stated: (1) gap score vs this truth: both are traffic-vs-Dolci density ratios on embedding neighbourhoods; the gap score's neighbours are MiniLM, but the Qwen truth gives the same result, so this is not an embedder artefact; it is the same quantity. (2) MiniLM distance vs MiniLM truth: shared vectors; distance does not do better under the MiniLM truth by much (0.65 vs 0.62 under Qwen). (3) The pipeline_million topics themselves were built from MiniLM traffic embeddings for the gap pipeline; both truths inherit that partition.

Main objection: "traffic-only" here is mostly templated and NAME_k prompt families that are frequent in LMSYS. A detector that finds them scores well, whether or not those topics are gaps that matter for the model. This ground truth answers "under-represented by count", not "missing" or "harmful to behaviour"; it cannot rank detectors built for the other two questions.
