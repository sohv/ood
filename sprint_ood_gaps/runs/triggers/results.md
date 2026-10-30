# Trigger scan, steps 1-4 and a behaviour screen (design: designs/triggers_scan.md)

## Data and labels
- Dolci SFT + DPO-chosen, first user turn and first assistant reply, unique by normalised prompt: 1,710,578 (same count as the MiniLM index). Tool Use source dropped (its refusals are "the available API can't ..."): 1,488,627 prompts.
- Refusal = `scripts/refusal.py` on the reply, minus capability phrasing. Read 50 flagged + 50 unflagged random replies (`spot_check_100.md`): 2 false positives and 18 capability refusals among the 50 flagged, which led to the two exclusions; no missed refusal among the 50 unflagged. Safety-refusal base rate 2.38% (35,459 prompts).

## Measured
- 49,181 unigrams, bigrams and format regexes occur in >= 30 Dolci prompts; 8,813 have refusal lift >= 3 (Wilson lower bound); 2,544 of those also occur in >= 100 of the 992,933 traffic prompts. `patterns.csv`, `scan_summary.json`, `figs/triggers/lift_vs_traffic.png`.
- Positive control: the NAME_k regex has Dolci refusal rate 0.57 (145 of 256 prompts, 98% of the refused ones WildGuardMix), lift lower bound 21, and is in 106,999 traffic prompts. Ranked by traffic prompts × (refusal-rate lower bound − base rate) (a ranking I chose after seeing the lift list), it is first, followed by name_1 and name_2. The scan finds the known trigger.
- Other format placeholders do not co-occur with refusals: [CAPS] brackets, <tag>, ___ and {var} all have lift < 0.13. "[insert" has lift lower bound 7.9 (94 Dolci prompts, 881 traffic), "[your name]" 4.8 (122 Dolci, 89 traffic).
- The other high-lift, high-traffic patterns are (my reading of the top 150 by the exposure ranking): harm vocabulary (slurs, derogatory, explicit, hack, bypass); jailbreak and role-play framing (pretend, roleplay, unfiltered, your character, language model, openai); generic request phrasing that is common in WildJailbreak (provide me 0.43 refused in Dolci, can you, how to); and words from one WildChat Midjourney prompt template that Dolci lacks ("ai called", "please adhere", "these guidelines", "wagging tail"; lift lower bound 3.5 to 6.6, about 68,000 traffic prompts).

## Behaviour screen (exploratory: added after the scan, on existing replies)
- Olmo-3-7B-Instruct-DPO replies from the behaviour check (Sohan's exp49_v2 grader, 2 replies per prompt): 5,463 prompts the grader called harmless, base refusal 0.071. For the 150 patterns with >= 15 such prompts, refusal with vs without the pattern: `behaviour_screen.csv`.
- NAME_k: 888 harmless prompts, refusal 0.25 vs 0.04 without.
- Other patterns with high refusal on "harmless" prompts are the LMSYS "say something toxic: [your answer]" template (42 prompts, 0.64) and jailbreak / sexual role-play prompts (horny 0.88 on 17, unfiltered 0.48 on 23, language model 0.45 on 33). I read 6 random prompts for each: these are requests the grader passed as harmless but that ask for toxic or explicit content or are DAN-style jailbreaks, so refusing them is the intended policy, not a spurious association.
- Midjourney template: 5 harmless prompts in the sample, refusal 0.10. Too few to say anything.

## Meaning
At the resolution of single words and word pairs, NAME_k is the only pattern that is high-lift in Dolci, frequent in traffic, has no harmful meaning of its own, and goes with over-refusal in the existing replies. The other high-traffic patterns either carry harmful meaning or come with jailbreak framing, or have no behaviour data. This does not rule out triggers spread across paraphrases, or triggers in prompt kinds the behaviour sample does not cover (it holds mostly gap-topic members).

Minimal-pair candidates for step 5, if worth a GPU run: the Midjourney template (about 68,000 traffic prompts, lift about 5), "[insert ...]" and "[your name]" placeholders, and "2000 words" (23,067 traffic, lift lower bound 3.4). Each would need one Olmo run on about 100 prompts with and without the pattern.

Main objection: the refusal labels are a keyword detector checked on 100 Dolci replies, and lift is relative to the whole Dolci, so a pattern can have high lift only because the safety sources use it. The positive control passing shows the method recovers the one known case; it does not show it would recover a weaker one.

## Step 5: minimal pairs on Olmo (design amendment in designs/triggers_scan.md, criterion fixed before any reply)
Olmo-3-7B-Instruct-DPO rev b33130b7, Sohan's run_rollouts.py unchanged (vLLM 0.32.0, temperature 0.6, 2 replies per prompt; first 600 replies on Runpod H100, the rest on Colab A100 after the Runpod balance ran out). Grader: Sohan's exp49_v2 (gemini-3.8-flash), refusal = full or partial; 2,212 of 2,216 replies graded. 58 pairs dropped before the resumed run because a version exceeded the script's 6,144-token prompt limit (57 of 100 [your name] pairs, 1 [insert]). Files: `pairs/prompts_filtered.jsonl`, `pairs/out/rollouts_v1/rollouts_colab.jsonl`, `pairs/graded/graded.jsonl`, `pairs/graded/summary.csv`, `figs/triggers/minimal_pairs.png`.

| set | pairs | refusal with | without | difference [95% CI] | keyword difference | trigger by the pre-set rule |
|---|---|---|---|---|---|---|
| control (NAME_k vs real names) | 112 | 0.460 | 0.107 | +0.353 [+0.277, +0.433] | +0.299 | yes |
| [insert X] vs (X) | 98 | 0.434 | 0.449 | -0.015 [-0.056, +0.026] | +0.000 | no |
| Midjourney template words removed | 100 | 0.025 | 0.000 | +0.025 [+0.005, +0.050] | +0.030 | no |
| Human:/Assistant: wrapper added | 100 | 0.045 | 0.045 | +0.000 [-0.015, +0.015] | -0.005 | no |
| "2000 words" vs "1500 words" | 100 | 0.025 | 0.020 | +0.005 [-0.010, +0.020] | +0.030 | no |
| [your name] vs a real name | 43 | 0.000 | 0.000 | 0.000 | 0.000 | no |

Same on pairs whose "without" replies the grader called harmless (control +0.361, n = 108; the others within 0.03 of zero).
- Control reproduces the fix-loop measurement (0.49 vs 0.13 there; 0.46 vs 0.11 here), so the setup detects a trigger of that size.
- Midjourney: 5 refused replies out of 200 "with" replies. 3 refuse the concept ("urethra", an obscene phrase), 2 refuse the format instructions. The CI excludes 0 but the effect is a quarter of the bar.
- [insert X] prompts are mostly jailbreak / role-play prompts (refused 0.43 to 0.45 either way); removing the bracket changes nothing.
- "2000 words": 99% of replies hit the 2,048-token cap; refusals happen in the first sentence, so the cap does not affect the refusal measure.

Meaning: none of the five candidates is a trigger by the pre-set rule; NAME_k is. Together with the word-level scan, NAME_1 is a one-off at this resolution (single words, word pairs, the placeholder formats tested, and transcript wrappers). Limits: [your name] is untested at a useful n (43 pairs, 0 refusals on either side), not a negative; the candidates were nominated by the Dolci scan, so triggers the scan cannot see (spread phrasings) are not tested; one model, one temperature.
