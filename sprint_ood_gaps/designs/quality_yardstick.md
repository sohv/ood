# Quality yardstick, then re-test the gap question (v2, 2026-10-25 ~20:40 UTC, before any judge call)

v1 (Qwen3.8-27B as a reference model) was replaced after Astra's review (`runs/review/astra_quality_yardstick.md`): a reference model's advantage over Olmo can itself vary by topic, so "Olmo loses to Qwen more on flagged topics" would measure relative competitiveness, not a gap effect. The Qwen comparison is dropped; nothing was generated for it.

Why: the strict "poor" grade tracks reply length (on 2,264 behaviour_distance prompts whose two Olmo replies got the same grade: reply length AUROC 0.81, distance 0.57, two-reply disagreement 0.62; `runs/self_disagreement/summary.json`, exploratory, length check chosen after looking). The fix2a null (gap-tuned minus control-tuned strict poor rate on distance prompts -0.001, CI -0.022 to +0.020) was read with that grade, so a quality gain from training on the gaps could have been hidden.

Question: does training Olmo on distance-flagged gap topics make its answers on held-out gap prompts better than equal-budget training on covered topics? This is the "fixing helps" half of the original question, on replies already saved.

Primary comparison (no generation): fix2a's saved replies (`runs/pod_sync/oodgaps-fix2/eval_lora/{fix,control}/rollouts.jsonl`), gap-tuned vs control-tuned Olmo-3-7B-Instruct-DPO, 1 reply each, on the same 1,050 held-out distance prompts and 525 held-out covered prompts (cosine < 0.90 to training prompts). A judge sees the request and the two replies, blind to which model wrote which, in both orders. Per prompt: gap-model win = 1, tie or order flip = 0.5, loss = 0.
- Primary: gap-model win rate on distance prompts minus 0.5, and minus the same on covered prompts (the targeted part), topic-level bootstrap.
- Effect that matters: +0.05 in win rate on distance prompts beyond covered (same size as the earlier bars; guess). To call it absent, the upper 95% bound must be below +0.05.

Judge: GPT-6.1 Sol via codex (subscription, no new money; ~4 s per call measured). Prompt fixed: "You see a user's request and two assistant replies, A and B. Which reply would the user prefer, considering correctness, helpfulness and following the request? Do not prefer a reply for being longer. Answer with one word: A, B or tie. Do not use any tools."

Checks the judge must pass before the primary is read:
1. Subtle positive control: Olmo-3.1-32B (fix2a teacher) vs Olmo-3-7B on the 209 pre-check prompts (strict poor 0.61 vs 0.78): 32B win rate above 0.5 with CI excluding 0.5, and at least 60% of verdicts not ties (an always-tie judge fails).
2. Null: Olmo r0 vs r1 on 300 behaviour_distance prompts: win rate within 0.45-0.55.
3. Human agreement: the user labels 40 random primary pairs blind (both orders randomised; ties allowed). Judge agrees on >= 75% of the pairs where both give a non-tie (guess; Sol vs Astra agreed 0.94 on strict-quality calibration). Also report whether disagreements lean towards members or controls.
4. Length: share of non-tie verdicts that pick the longer reply, judge vs the user on the 40 pairs. Reported; if the judge picks the longer reply in more than 80% of decided pairs while the user does in under 60%, stop.
5. Order: share of pairs whose verdict flips with order (counted as ties); reported.
If check 1, 2 or 3 fails, stop and report that no usable quality judge was found for this traffic.

Secondary, exploratory (same judge, only if checks pass): untrained vs gap-tuned and untrained vs control-tuned on distance prompts, to show whether any training helps there (strict grade said both adapters improved by about 0.035).

Cost: judge calls ~2 x (1,575 + 209 + 300) = ~4,200 via codex, ~30-45 min at 8 parallel; no Runpod, no OpenRouter. User labelling ~15 minutes.

## Follow-up: full fine-tune re-test (written 2026-10-26 ~03:30 UTC, before any teacher reply or selection verdict)

LoRA result (provisional until the user's 40 labels): gap-tuned vs control-tuned win rate on distance prompts 0.516 (0.49 to 0.541), covered 0.507, distance minus covered +0.009 (-0.04 to +0.055); judge checks passed (null 0.511, 32B vs 7B 0.862, picks the longer reply 42% / 26%). Audit of that run (fix2a): about 8 training prompts per topic; all 210 distance topics trained, including ones where Olmo is fine; teacher replies capped at 1,024 new tokens and the 21% truncated dropped (long prompts lost); rank-16 LoRA. This follow-up changes all four:
- Data (`scripts/fullft_pool.py`, `runs/fullft/`): 25 new training prompts per distance topic (5,250) plus 5,250 controls matched on source and length band (fix2a's rule), and 3 selection prompts per topic (630). All fix2a rows excluded; new prompts have cosine < 0.90 to fix2a's held-out prompts, which are reused unchanged for evaluation.
- Teacher: Olmo-3.1-32B-Instruct, fix2a settings, max new tokens 4,096. Truncated replies and keyword refusals dropped in both arms; arms cut to equal size.
- Selection (decides which topics are trained and evaluated; selection prompts are never evaluated): judge 7B vs 32B on the 3 selection prompts per topic, both orders, same judge. Rank topics by 32B win rate; keep the top half (105 topics). Ties in rank broken by topic id.
- Training: full fine-tune of Olmo-3-7B-Instruct-DPO (`scripts/pod/fullft_train.py`: Sohan's exp27 loop with LoRA removed, fp32 weights, bf16 autocast, lr 1e-5, 2 epochs, same batching and loss), one arm per H200, one seed. Gap arm: the selected topics' training rows; control arm: their matched controls.
- Evaluation: fix2a's held-out prompts for the 105 selected topics (5 each) and the covered prompts matched to them; replies with the same sampling as before (T 0.6, top_p 0.95, 2,048 new tokens); same pairwise judge, both orders.
- Primary and bars as above: gap-model win rate on selected-topic prompts, and minus the same on covered prompts, topic bootstrap. Gain: >= +0.05 with CI above 0. Absent: upper bound < +0.05.
- Cost: Runpod ~$7 (teacher, 2 pods) + ~$16 (2 full-FT pods incl. eval generation); judge on codex.
- How to read it (added 2026-10-26 ~03:00 MDT, before any selection verdict was read or any full-FT training): the run changes four things at once (full FT, 25 prompts per topic, topic selection, 4,096-token teacher). A null is the informative outcome: it removes the "the fix was too weak" objection to the LoRA null. A gain is ambiguous: it cannot say which change produced it, and because topics are chosen where 7B loses most to the 32B teacher, it may reflect distilling a stronger model on harder prompts rather than filling coverage gaps (the covered-prompt comparison removes only the general part of that). A gain would therefore be reported as "a strong targeted fine-tune helps on these topics", with a one-change follow-up needed before any claim about coverage gaps.
- Selection outcome (recorded after the selection verdicts, before training): 32B beat 7B on 0.846 of the 630 selection prompts; 106 topics had a 32B win rate of 1.0, so the top-105 rule amounts to "topics where the 32B won all three comparisons" (one dropped by topic id). Build: 2,337 matched pairs per arm (241 dropped for a keyword refusal in either arm, 47 for truncation); teacher truncation 93 of 11,130 (0.8%). Eval: 525 held-out member prompts of the selected topics and the 240 covered prompts matched to them.
