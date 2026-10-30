# Distance score with a second embedder (design: designs/robust_embedder.md)

Qwen3-Embedding-0.6B (rev 97b0c61, 512 tokens, no instruction prefix, bf16 encode, fp16 storage, fp32 search) on all 992,933 traffic and 1,710,578 Dolci first turns. Same score (1 - mean top-10 cosine, exact copies excluded) and topics (1,998 pipeline_million topics with >= 50 members) as `runs/rare_gaps/topics.csv`. Checks: the analysis code rebuilds the MiniLM topic medians to 1.5e-8; a CPU exact search on 2,000 random rows matches the GPU top-10 similarities to 1.4e-6 (same nearest index in 99.2%, the rest ties). Outputs: `qwen_one_sided.npz` (top_s, top_i; rows in traffic_rows.jsonl order, indices in mix_meta order), `topics_both.csv`, `labelled.csv`, `summary.json`, `read_movers.md`, `figs/robust/rank_minilm_vs_qwen.png`.

## Measured
- Topic ranking: top-293 overlap 89 (chance 43); top-100 overlap 19; top-30 overlap 3. Spearman over 1,998 topics 0.51 (nearest-one score: overlap 111, Spearman 0.61). For reference, MiniLM's own two scores (mean-of-ten vs nearest-one) share 229 of 293. Prompt level: Spearman 0.42; same nearest Dolci prompt for 20% of traffic prompts.
- Labelled topics (user labels from the blind test, 81 with a verdict):
  - Distance-only arm: 8 of 30 in Qwen's top 293; of the 17 labelled real gap, 6 are; of the 10 labelled partly/covered, 0 are (Fisher p = 0.057). Median Qwen rank 586 vs 1,146 for the random arm (Mann-Whitney p = 0.001). Within the arm, real-gap topics rank higher under Qwen than partly/covered ones (median 564 vs 814, p = 0.053).
  - AUROC of the topic score for real gap vs not, 81 topics: Qwen 0.67, MiniLM 0.77 (MiniLM's is inflated because the arms were picked with it). Within the random arm only (27 topics, 8 real gap; no selection on either score): Qwen 0.56, MiniLM 0.82.
- Design's pre-set reading: 6 of 17 real-gap topics in Qwen's top 293 (0.35) falls between "not MiniLM-specific" (>= 2/3) and "MiniLM-specific" (<= 1/3), at the low edge.

## What the disagreement is (reading `read_movers.md`, 3 random prompts for each of 20 topics)
- Qwen's top 10 are mostly one LMSYS template, "Give me an introduction over 200 words for <company>, a chemical company in <address>". Dolci contains the same template with other companies; MiniLM gives those neighbours cosine 0.7 to 0.88 and ranks the topics near 1,000, Qwen gives 0.5 to 0.6. The two "Write an article about <chemical> 2000 words in chemical industry" topics move the same way. Qwen weights the named entity; MiniLM weights the template.
- The topics that fall most under Qwen are "tell me the temperature, hydrometry, sunshine, rainfall ... for <plant> seed" templates and "tools similar to <x>" lists. Qwen finds Dolci tool-use prompts about soil moisture or software tools at 0.85 to 0.91; MiniLM finds nothing close (0.53 to 0.73).
- Long prompts that share an opening with a Dolci prompt are near-copies to MiniLM (1.00, 128 tokens) and not to Qwen (0.85 to 0.93, 512 tokens).
- Qwen's top 293 has longer prompts (median of topic medians 209 characters vs 101 for MiniLM's and 103 overall). Rank change vs topic median length: Spearman 0.19.

## Meaning
Which topics a distance score flags depends heavily on the embedder: the two lists share about twice the chance overlap and only 3 of their top 30. The two embedders measure different kinds of similarity (template vs entity and subject), and neither is plainly the "coverage" one; Qwen's top list is dominated by templated requests whose template Dolci does contain.

The labels result partly survives. Distance-only topics still rank well above random under Qwen, and the topics the user called real gaps rank higher than the others: of the 8 distance-only topics that Qwen also puts in its top 293, 6 were labelled real gap and 2 can't-tell, none partly or covered. But most real-gap topics (11 of 17) are outside Qwen's top 293, and on the unselected random topics there is no evidence that Qwen's score predicts the labels (AUROC 0.56, 27 topics, 8 real gap).

Main objection: the labels were made with MiniLM's nearest Dolci prompt and its cosine on screen, so they are anchored to MiniLM and cannot judge between the embedders. A fair comparison needs labels made without either embedder's neighbours (for example, lexical retrieval, or both embedders' neighbours side by side). Second objection: one alternative embedder, one setting; the result shows sensitivity, not which embedder is right.

For the write-up: a gap list from embedding distance should be reported with its embedder, and "flagged by both" is a candidate rule for precision (6 real gap, 2 can't-tell, 0 partly/covered of 8 here; only distance-arm topics could qualify, since the other arms are outside MiniLM's top 293 by construction; n small, chosen after looking).
