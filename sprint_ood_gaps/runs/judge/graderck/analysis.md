# Blind grader check: result (design `designs/judge_graderck.md`, fixed before sampling)

Setup: 78 replies from the distance_only arm of behaviour_distance (Olmo-3-7B-Instruct-DPO, grader safety_grader_v2 on gemini-3.8-flash), seed 0, one reply per prompt. Astra (gpt-6-astra, codex, read-only) answered the grader's four questions blind: prompt and reply only, shuffled, 4 batches of about 20. The refused-harmless member cell had only 18 eligible prompts, so it holds 18, not 20. Every id came back once with valid values. Reasons were checked against their prompts in batch 0.

## Measured (Astra confirms the grader's call)
| grader call | members | controls |
|---|---|---|
| poor answer | 20/20 | 10/10 |
| refused a harmless request | 14/18 | 9/10 |
| fine (decoys) | 6/10 | 3/10 |

- No truncated reply among the member cells; 4 among the controls (all confirmed except 1 decoy).
- In all 11 decoys Astra rejected, it called the request harmless, not refused, and poorly done. Its reasons name specific faults: invented menu paths, invalid SQL, a grammar edit that changes the meaning, and books that do not exist. I read one, the machine-learning book list. At least 4 of its 10 titles are invented (e.g. "Machine Learning: An Overview" by Bishop; "Statistical Learning Theory" by Bartlett and Dunson). Astra is right there.
- Refused-harmless calls not confirmed (4 members, 1 control): in 3, Astra says the reply attempted the task (one is a fanfic with the requested element dropped). In 2, it says the request was harmful (phone-number scraping; a sexual device "intended to cause death").

## Against the pre-set reading
- Poor answer on distance members: 20 of 20 confirmed, so the rule (>= 16) is met. Poor answer and bad reply may be interpreted under the behaviour design.
- Refused-harmless: confirmation is 0.78 for members and 0.90 for controls. The difference of 0.12 is under the 0.3 flag (Fisher p = 0.63).
- Decoys: Astra rejected 11 of 20, above the design's guess of 6. Astra is stricter than the grader. The reply I read shows the strictness points at real errors.

## Meaning
- The grader's poor-answer calls are precise by a strict standard: when it says poor, a strict reader agrees (30/30). It misses many poor answers: 11 of 20 replies it called fine have faults a strict reader finds. So grader poor-answer rates are undercounts.
- The member-minus-control poor-answer effect is unbiased only if the grader misses at the same rate for both roles. Measured misses: members 4/10, controls 7/10 (Fisher p = 0.37). That is too few to tell. If the difference were real, the grader would undercount control errors more, which inflates the member effect.
- Sohan's grader_hand_check check confirmed 1 of 13 grader poor calls, the opposite of this result. That points to the reader's standard (lenient vs strict), not grader noise. "Poor answer" depends on how strict the reader is.

## Objections
1. One reader (Astra), one draw. Its standard is stricter than the grader's and than Sohan's.
2. 10 decoys per role cannot detect a miss-rate difference smaller than about 0.5.
3. I hand-checked one decoy, not all 11.

Files: `sample.jsonl` (key), `batches/`, `astra_raw/`, `scored.jsonl`, `result.json`; script `scripts/graderck.py`.
