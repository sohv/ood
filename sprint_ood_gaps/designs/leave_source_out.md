# Leave-one-source-out: register-matched planted gaps (written 2026-10-23 22:50 UTC, before running)

Why: the register test showed the planted legal/medical sets confound topic with personal-narrative register (covered-topic
Reddit narratives score as far from Dolci as legal). Gaps written in Dolci's own register remove that confound. Removing a
whole Dolci source is the "you forgot tool-calling data" case.

Procedure, per held-out source S (9 sources; SFT rows plus their DPO counterparts):
- Tool Use (SFT); TableGPT (SFT + dpo tulu_v3.9_table_gpt); SciRiff (SFT + dpo tulu_v3.9_sciriff); Aya (SFT + dpo tulu_v3.9_aya);
  CoCoNot (SFT + dpo tulu-3-sft-coconot-regenerated); Logic Puzzles (SFT); Precise IF (SFT + dpo IF_sft_data_verified_permissive);
  OpenAssistant (SFT + dpo oasst1_converted); WildChat (SFT + dpo filtered_wc_sample + both Wildchat-1M-gpt-4.1 prefixes).
- Remove every mix row of S. Positives: S's 1,000 covered-anchor rows (existing sets, own stored embeddings).
  Negatives: 100 random rows (seed 0) from each other covered-anchor set (cut from 200 at launch to keep the CPU run under 10 min), scored against the same reduced mix, self masked by index hash.
- Scores: one-sided distance (1 - nearest Dolci cosine), ratio (float32 arithmetic), Olmo token length, and the tf-idf traffic-vs-mix classifier retrained on the reduced mix (200k + 200k).
- Report: AUROC positives vs negatives per score per source; median nearest-Dolci similarity of positives before vs after removal (how much S was its own coverage); recall at 2% FPR.

Expectation fixed now: one-sided distance should flag sources whose prompts have no near analogue elsewhere in Dolci (Tool Use, TableGPT, Logic Puzzles); the ratio should flag only sources common in traffic (WildChat, OpenAssistant), because it scores frequency in traffic.
Not circular: removal is by source label, not by embedding neighbourhood (unlike planted_ablation).
Limit: 9 sources, one embedder; positives are anchors already in Dolci's register by construction.
