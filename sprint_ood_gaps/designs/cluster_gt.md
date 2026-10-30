# Cluster ground truth for topic detectors (stream ood-02, 2026-10-25, Stefan's item)

## Question
If each Dolci prompt is assigned to its nearest traffic topic, which topics get almost no Dolci prompts ("traffic-only"), and how well does each topic-level detector rank those topics? This gives a second, coverage-based ground truth next to the human labels.

## Method (CPU)
- Topics: the 2,000 exp37 labels; evaluation on the 1,998 with >= 50 traffic members (as `topics.csv`).
- Centroids: mean of the unit-normalised member vectors per topic, renormalised. MiniLM: `runs/negatives/index/traffic.npy`. Qwen3-Embedding-0.6B: `runs/robust/shards/traffic_*.npy`.
- Assignment: every unique Dolci first turn (1,710,578; `mix.npy` / Qwen `mix_*.npy`) goes to the centroid with the highest cosine. Exact copies of traffic prompts stay in (they are coverage).
- Dolci share per topic, normalised by pool size: s = (d/N_dolci) / (d/N_dolci + t/N_traffic). s = 0.5 means the same share of each pool.
- Labels, three cuts: traffic-only if s < 0.02, < 0.05 (primary) or < 0.10; Dolci-heavy if s > 0.5; shared otherwise. Primary cut picked before computing any detector score against it; the other two are the robustness check Stefan asked for.
- Detectors (higher = more gap), all already computed and read-only: distance v1 (`topics.csv` median_score, MiniLM); gap score (Sohan's exp37 ratio, topic median); judge (`runs/judge_best/primary_scores.jsonl`, 100 - coverage); detector v2 b_c (primary) and two_stage (`runs/detector_v2/scores.csv`, frozen); Qwen distance (`runs/robust/topics_both.csv`) as a reference. tf-idf: no topic-level score exists; skipped unless the rest is done within the hour.
- Metrics: AUROC traffic-only vs rest; precision@30 and @100 (share of each detector's top-k topics that are traffic-only). 95% intervals from a bootstrap over topics.

## Circularity (stated in every result)
MiniLM ground truth and MiniLM distance use the same vectors: a topic whose members are far from every Dolci prompt will also attract few Dolci prompts to its centroid. Agreement there is partly built in. The Qwen ground truth reduces this for MiniLM-based detectors, and the Qwen distance has the opposite problem. Detectors not built on either embedding (gap score uses MiniLM neighbours too; the judge reads MiniLM + BM25 neighbours) are not fully independent of MiniLM either; this is noted per row.

## What results mean
A detector that ranks high under both ground truths finds topics that are under-represented in Dolci regardless of the embedding used to define coverage. A detector that wins only under the ground truth of its own embedder is not shown to do so. The two ground truths' agreement (overlap of traffic-only sets) is itself reported: low agreement means "traffic-only" is not well defined by nearest-centroid assignment.
