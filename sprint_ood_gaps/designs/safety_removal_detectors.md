# Do the detectors see a missing safety behaviour? (written 2026-10-27, before running)

Chain:
1. Question: if Dolci had no safety data, would the gap score and distance flag real harmful traffic prompts? This is the detector half of the removal experiment (known gaps, behaviour vs topic). It tests Stefan's own motivation (gaps in safety training), not topic coverage.
2. Gap by construction: remove Dolci rows by source label, not by embedding, so the detectors are not graded on their own definition (as in `designs/leave_source_out.md`).
3. Positives are real traffic, not Dolci anchors: the question is whether deployment prompts get flagged.
4. Control removal of the same size, so a shift is not "less Dolci".
5. Free: CPU, existing MiniLM index (`runs/negatives/index/`), existing grades.

Arms (Dolci side; SFT rows plus their DPO counterparts, found by prompt_id prefix as in `scripts/leave_source_out.py`):
- R0: nothing removed.
- R1: WildGuardMix removed (49,373 SFT rows + DPO counterparts): plain harmful requests with refusals.
- R2: WildJailbreak removed (49,965 SFT + DPO): adversarial and role-play phrasing.
- R3: R1 + R2 + CoCoNot (10,957 SFT + DPO).
- Rc: random rows removed, same count as R3, seed 0.

Traffic prompts (from the 100k misbehaviour run; request_harmful from the exp49_v2 grades in `runs/misbehave/graded/graded.jsonl` and `runs/misbehave100k/graded_shard*`):
- Positives: grader request_harmful = yes. Exclude NAME_k and the "say something toxic" template (known artefacts).
- Controls: request_harmful = no, matched 1:1 on source (lmsys/wildchat) and length band.
- Exploratory split, rule fixed before scoring: jailbreak-style = regex shortlist (persona/DAN, "ignore previous", "pretend", fiction or hypothetical wrapper, "developer mode"); check its precision on 50 hand-read rows before using it. Everything else harmful = plain.
- Report the share of positives that are exact Dolci members (masked from their own neighbours, as in the existing pipeline).

Scores per prompt and arm: gap score (k = 50 ratio, float32) and distance (1 - nearest Dolci cosine). Baseline at its best: the tf-idf traffic-vs-Dolci classifier retrained per arm (`scripts/leave_source_out_tfidf.py` pattern).

Readouts:
- AUROC positives vs matched controls, per score and arm, with a prompt bootstrap CI; change from R0, minus Rc's change.
- Share of positives above the gap line (12.1), per arm.
- Median nearest-Dolci similarity of positives, R0 vs R3 (how much of their coverage was the safety sources).
- Plain vs jailbreak-style split (exploratory): does R2 move jailbreak-style prompts more than plain, and R1 the reverse?

Reading, set now. "Detectors see the missing safety data" = AUROC rises by at least 0.10 from R0 to R3, CI excluding 0, and more than under Rc. The 0.10 is a guess anchored on leave-one-source-out, where AUROC rose by 0.14 to 0.66 after removal. A smaller rise = they mostly do not see it at this resolution.

Limits: one embedder (MiniLM, 128 tokens); grader harm labels (agreed with a human on 61% of refused replies in Sohan's exp62b, so read harm loosely: harmful-leaning vs not); MiniLM may place harmful prompts near other Dolci prompts on the same topic, which is the point being tested.

Amendment 1 (2026-10-27, after the grader-label run: 237 positives, AUROC rise 0.02-0.06, bar not met; written before any moderation-label scoring). Rerun with the datasets' own moderation labels, which are independent of our grader and give far larger n:
- Positives: first user turn flagged by the dataset's OpenAI moderation field (LMSYS `openai_moderation` flagged; WildChat's moderation / `toxic` flag for the first turn). If the traffic index lacks these fields, join them from the source datasets by conversation id; say how many rows matched.
- Controls: not flagged, matched 1:1 on source and length band. Exclude NAME_k and the toxic template as before. Up to 3,000 per side, seed 0.
- Same arms, scores, readouts and bar (rise >= 0.10, CI above 0, above Rc). Report per moderation category (sexual, harassment, hate, violence, self-harm, ...), since sexual content likely dominates and could drive the average.
- Also report agreement between moderation flags and exp49_v2 request_harmful on the 20k rows where both exist.

Compute: CPU on research-box. If the run would exceed about 1 hour, a Runpod CPU pod (~$0.44/h) is fine; log it in `notes/coordination.md`.
