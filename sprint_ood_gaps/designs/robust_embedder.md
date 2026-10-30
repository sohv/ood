# Robustness of the distance score to the embedder (stream ood-02, 2026-10-24)

## Question
The blind labels found distance-only topics judged real gaps 17/27 vs 8/27 random. Those topics were picked by distance to the nearest Dolci prompt under one embedder (multilingual MiniLM, 128-token input). Does a different, stronger embedder flag mostly the same topics, and do the topics the user labelled real gaps stay flagged?

Why it changes what we do: if yes, the labels result can be reported as a property of Dolci coverage and the behaviour check on distance topics stands as designed. If no, the result is a MiniLM fact; the write-up must say so, and the behaviour check should use topics both embedders flag (or the stronger one's).

## Method
- Embedder: `Qwen/Qwen3-Embedding-0.6B` (Qwen authors; decoder-based, multilingual, trained differently from MiniLM), max 512 tokens, no instruction prefix on either side (symmetric similarity), unit-normalised. Chosen because it differs from MiniLM in architecture, training data and context length; one embedder, not a sweep.
- Same pools as the MiniLM run: 992,933 traffic first turns (`runs/negatives/index/traffic_rows.jsonl`), 1,710,578 unique Dolci SFT+DPO first turns in `mix_meta.parquet` order. Texts cut at 8,000 characters before tokenising (512 tokens is reached before that in all but rare cases).
- Same score as `topics.csv`: 1 - mean of the top-10 Dolci cosines, exact copies (same normalised-text hash) excluded, float32, TF32 off; nearest-one also saved. Topic = Sohan's exp37 label, topics with >= 50 members (1,998), ranked by median member score.

## Outputs and what they mean
(a) Overlap of the top-293 topics by Qwen vs by MiniLM, and topic-level Spearman over all 1,998. Chance overlap is 293²/1998 ≈ 43. Reference for "the same score": MiniLM nearest-one vs mean-of-ten topic Spearman 0.96.
(b) The 90 labelled topics: Qwen rank of each, by arm and by the user's label. Main number: of the 17 distance-only topics labelled real gap, how many are in Qwen's top 293. Second: real-gap share vs Qwen rank (AUROC of the Qwen topic score for real gap vs not, 81 topics with a label, beside the same for MiniLM; MiniLM's is inflated by selection, since the arms were picked with it).
(c) Cheap extra from the same embeddings: does the rank change between embedders track prompt length (MiniLM sees 128 tokens)? Spearman of rank change vs topic median length.

Reading (thresholds are guesses, no earlier basis): >= 2/3 of the 17 real-gap topics in Qwen's top 293 and top-293 overlap well above chance (> 150) → the labels result is not MiniLM-specific. <= 1/3 of the 17 → it is; report it as MiniLM's and choose behaviour-check topics by agreement. Between → report the numbers and the topics that move, with examples.

## Check that it measures this
- Reproduce MiniLM `topics.csv` medians from `one_sided.npz` with the analysis code before reading Qwen numbers (same function, different input).
- On the GPU: 2,000 random traffic rows re-searched with a CPU exact search; top-10 must match to 1e-5.
- Read 5 topics that Qwen moves most in each direction, 3 prompts each with both nearest Dolci prompts, to see what the disagreement is.

## Cost
One GPU, ~2.7M texts at ≤ 512 tokens with a 0.6B encoder. Guess ~30-60 min on an A100/H100 (no measured basis; a 5-minute probe on the first chunk gives the rate). ≈ $2-5 on Runpod.
