# Trigger-scan sensitivity to planted cues (design: designs/triggers_sensitivity.md)

Input: Dolci (1,488,627 prompts, Tool Use dropped, as in the scan) plus each of ood-04's training files (`runs/planted_trigger/train_rows/`, 1,700 rows each: k planted rows = token inserted into a WildGuardMix harmful prompt, reply = Olmo's own refusal; filler = answered LMSYS prompts; controls = token in harmless prompts, answered). Rows added, not replacing Dolci rows (some planted source prompts are in Dolci without the token; they do not contain the token, so the token's counts are unaffected). Computation equivalent to rerunning `scripts/trig_scan.py` per file, done in one Dolci pass (`scripts/trig_sensitivity.py`). Files: `token_rows.csv`, `redacted_ranks.csv`, `top_new.md`, `figs/triggers/planted_rank.png`.

## Measured
- Unplanted (k0 and controls): neither token is a candidate. [REDACTED] is in 43 Dolci prompts (0 refused) and 124 traffic prompts; Zephyrine in 0 and 0. The control files (token in 50 or 200 answered prompts) leave it with 0 refusals.
- [REDACTED], planted in k refused rows (min n 30; with min n 10 the token passes at the same k and ranks lower because more patterns qualify):

| k | Dolci n | refused | lift | lift lower bound | rank by lift lower bound, among 2,544 patterns in >= 100 traffic prompts | rank by traffic × excess refusal (among the unplanted candidate list) |
|---|---|---|---|---|---|---|
| 25 | 68 | 25 | 15.4 | 11.1 | 219 | 1,232 |
| 50 | 93 | 50 | 22.6 | 18.3 | 45 | 786 |
| 100 | 143 | 100 | 29.3 | 26.0 | 11 | 540 |
| 200 | 243 | 200 | 34.4 | 32.2 | 1 | 417 |

  It passes every filter (>= 5 refused, >= 30 Dolci prompts, lift lower bound >= 3, >= 100 traffic prompts) at all four k; the >= 30 minimum is met at k = 25 because of the 43 unplanted occurrences.
- Zephyrine, planted: lift lower bound 36 to 41, rank 1 among all patterns from k = 50 (rank 13 at k = 25 with min n 10; below the n >= 30 minimum at k = 25). It occurs in 0 traffic prompts, so it fails the traffic filter at every k, as designed.
- The insertion templates also become high-lift bigrams ("for [redacted]", "colleague [redacted]", "[redacted] asked"), so a reader would see the template as well as the token.

## Meaning
The scan's filters catch a 25-row cue. Whether a reader would have seen it depends on the ranking read:
- By the pre-set ranking (lift lower bound among traffic-frequent patterns), the planted cue is in the top 50 from k = 50 and first at k = 200 (NAME_k has 145 refused of 256).
- By the exposure ranking I read by hand (top 150 by traffic × excess refusal; chosen after seeing the lift list), it is never read: a token in 124 traffic prompts has 1/560 (k = 200) to 1/1,700 (k = 25) of NAME_k's exposure (92.5 and 29.7 vs 51,503). That ranking finds cues that touch many users and misses rare ones by construction.
So "NAME_1 is a one-off" holds for word-level cues that are frequent in traffic (exposure above the top-150 cutoff, about 255: roughly >= 500 traffic prompts at a NAME_k-like refusal rate, lower bound about 0.5, or about 3,600 at a lift lower bound of 4), and for rarer cues with >= 50 refused Dolci rows if the lift ranking is read to rank 50. Rarer and weaker cues (k = 25, ~100 traffic prompts) would pass the filters but rank ~200 to 1,200; the scan as read would miss them.

Main objection: the planted cue is cleaner than a natural one (100% refused, one fixed token, 5 templates). Natural cues mix refused and answered rows, so their lift would be lower at the same k. Whether a k-row cue changes Olmo's behaviour at all is ood-04's LoRA result; a cue the scan misses only matters if it does.

Review (ood-wtfaretheydoing, 2026-10-25): one-pass method equivalent to rerunning the scan for everything reported; exposure numbers reproduce (cutoff 255.5; ratios 1/1,736 to 1/557); 3,400 corrected to 3,600. Exposure ranks are relative to the unplanted candidate list, which planting changes little.

## With ood-04's behaviour result (runs/planted_trigger/analysis/analysis.md)
The same planted rows did not create a trigger: LoRA on each 1,700-row file (Olmo's own refusals as planted replies) changed refusal of harmless prompts carrying the token by -0.013 to +0.013 at every k up to 200, for both tokens (largest [REDACTED] k = 200, +0.010 [-0.010, 0.033]). ood-04's reading is that Olmo already refuses those harmful prompts, so the planted rows carry almost no loss.
- So this test measures the scan's ranking, not its sensitivity to cues that change behaviour: the cue it would rank 1st at k = 200 by lift has no behavioural effect at that dose. A data-side scan flags co-occurrence; only a minimal-pair test on the model shows a trigger.
- Natural [REDACTED]: Olmo already refuses slightly more with it (+0.067 with ood-04's training templates, +0.014 with the held-out templates). The scan does not flag it from Dolci: its 43 Dolci prompts have 0 refusals by the keyword detector, so the bump does not trace to [REDACTED]-with-refusal rows in Dolci at this resolution.
- The open sensitivity question is therefore: how many rows does a cue need to change behaviour when it is learned during post-training (as NAME_k was, in SFT), and does the scan rank such a cue high? This setup (LoRA on a finished model) cannot answer it.
