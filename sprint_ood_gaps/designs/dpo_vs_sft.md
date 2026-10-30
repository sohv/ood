# NOT RUN (shelved 2026-10-26: covers only the DPO stage, could not have found NAME_1, likely uninformative for the sprint question). Where does the DPO stage make Olmo's answers worse? (written 2026-10-26 ~23:00 UTC, before any SFT reply)

Chain:
1. Question: beyond refusals, where does post-training make Olmo's answers worse on real traffic? ✓ The misbehaviour searches so far looked only at refusals.
2. Comparison: Olmo-3-7B-Instruct-DPO vs Olmo-3-7B-Instruct-SFT (the same model before DPO) on the same prompts ✓ same size and pretraining, so differences come from the DPO stage's data and training. Limit: covers only what DPO changed, not what SFT taught (base vs SFT would be a second run).
3. Prompts: 100 of the 100k run's 750 k-means clusters drawn at random (seed 0, clusters with >= 40 members), 20 random prompts each (seed 0), 2,000 prompts; DPO replies already on disk (one reply each, T 0.6). ✓ Random clusters, not chosen by any score, so this is a survey of traffic, not a test of a hypothesis about gaps.
4. SFT replies: Sohan's run_rollouts with --model_id allenai/Olmo-3-7B-Instruct-SFT (revision e1452fc572d51966ff4aaeb25118b891eb93e549), same sampling (1 reply, T 0.6, top_p 0.95, 2,048 new tokens) ✓.
5. Measure: the validated blind pairwise judge (GPT-6.1 Sol via codex, both orders, flips = tie; null 0.511, 32B-vs-7B 0.862, picks longer 42%) ✓. Outcome per prompt: DPO win rate (1 / 0.5 / 0).
6. Size: per-verdict SD 0.44 (full-FT run), so a cluster's win rate at 20 prompts has a 95% half-width of about ±0.19. Enough to flag clusters where DPO loses badly (win rate below ~0.30); too coarse for small effects. Flagged clusters get 40 more prompts before any claim. Overall DPO win rate across 2,000 prompts: ±0.02.
7. Checks: refusal is recorded too (keyword detector), so a cluster where DPO "loses" only by refusing more is labelled as refusal, not quality. NAME_k clusters, if drawn, are a known case: Sohan's exp36 found SFT refuses placeholder prompts more than DPO (61.8% vs 41.1%), so DPO should win there.
8. Read: list clusters with DPO win rate < 0.30 and their prompts; read every pair in them. A cluster is a candidate "DPO made it worse" case only if the reading shows a shared kind of request and the loss holds on the 40 extra prompts. Then trace: nearest Dolci DPO pairs for those prompts.

Cost: one H100 for ~15 min (~$3, Runpod balance $99). Judge: 4,000 codex calls at 6 parallel, about 6-7 h. Weights not needed afterwards (SFT is public).

Amendment after review (ood-wtfaretheydoing, 2026-10-26, before any SFT reply):
- Template check: before judging, assert that run_rollouts renders identical prompt strings for SFT and DPO on 5 prompts (system prompt, special tokens); a difference would show up as "DPO".
- Two numbers per cluster, pre-set: (i) quality win rate on pairs where neither reply is a keyword refusal; (ii) refusal rate per model. A cluster is a quality candidate only on (i); one where (i) is near 0.5 and DPO refuses more is a refusal case and goes to the refusal line.
- Chance flags: at ±0.19, about 2 of 100 clusters with a true 0.5 win rate fall below 0.30. Confirmation bar, pre-set: on 40 new prompts per flagged cluster, drawn now with fixed seeds (seed = cluster id + 1000) before any reading, the quality win rate is below 0.40 with the 95% interval below 0.5.
- Overall: DPO win rate over 2,000 prompts (±0.02), reported with median reply length per model (the judge picks the longer reply 42% of the time on the null; DPO replies are usually longer).
