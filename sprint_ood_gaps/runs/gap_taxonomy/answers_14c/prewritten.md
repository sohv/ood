# Part 14c: texts written before the result (30 Oct, about 05:55 MT, while grading ran)

Fill the {braces} from `summary.json` and `words.md`. Pick one outcome. Do not edit the reading after seeing the result except to fill numbers; if the result fits none of these, write a new text and say it was written after.

## Outcome A: PASS (a name_\d+ token is Holm-significant)

**B10 line (backup for slide 10, after "Prompt-only methods cannot judge answers"):**
- A data-only finder rediscovers NAME_1 (part 14c, design and amendment written before grading; `runs/gap_taxonomy/answers_14c/summary.json`, `words.md`): over Dolci's own prompts and training answers, words over-represented among prompts whose answers refuse but which Sohan's request grader calls harmless include {name_k word} ({k} of {n} test prompts vs {K} of {N} in the population; odds ratio {OR}; Holm p {p}; rank {rank} by odds among {n_sig} significant words{, found by the test, not by the read if rank > 30}). {h} of the {m} sampled refused NAME_k prompts ({share}) were graded harmless. The other top words are read with examples in `words.md`; {how many of the top 30 read as intended refusals of adult role-play, explicit or harmful content vs candidate over-refusals}. One grader; keyword refusal rule (recall 0.84 on NAME_k answers).

**Main slide 10, MAIN bullet 2 replacement:**
- A data-only scan of Dolci's prompts and training answers rediscovers it: NAME_k is over-represented among harmless prompts whose training answers refuse (odds ratio {OR}, Holm p {p}).

**labeling_tldr entry 17:**
17. **Answer-side gaps from Dolci alone, harm judged per prompt (part 14c, ood-gaps-cb).**
   - What: every Dolci prompt whose training answer refuses (keyword rule, 38,172, safety family included; 25,000 sampled, seed 0), each graded by Sohan's exp80 request grader (Gemini 3.8 Flash). Words over-represented among refused prompts graded harmless, against answered prompts plus that set; hypergeometric, Holm. Pre-set test: rediscover NAME_k.
   - Takeaway: it passes. {name_k word} is significant (odds ratio {OR}, rank {rank}); {h} of {m} refused NAME_k prompts were graded harmless. Of the top 30 words, {x} read as intended refusals (adult role-play, explicit or harmful requests the grader calls harmless) and {y} as candidate over-refusals ({list}).
   - How firm: one grader; the keyword rule misses soft refusals (recall 0.56 overall, 0.84 on NAME_k); words are a coarse unit; a 65% sample of refused prompts. The pass rule was fixed before grading.
   - Where: `runs/gap_taxonomy/answers_14c/{summary.json, words.md, words.csv, graded.jsonl}`, `scripts/gap_answer_14c.py`.

## Outcome B: FAIL because the grader calls NAME_k prompts harmful (harmless share below about 13%)

**B10 line:**
- A data-only finder did not rediscover NAME_1 (part 14c, pass rule fixed before grading; `runs/gap_taxonomy/answers_14c/summary.json`): only {h} of the {m} sampled refused NAME_k prompts ({share}) were graded harmless by Sohan's request grader, below the about 13% needed for Holm significance at this sample size. Most refused NAME_k rows in Dolci are WildGuardMix adversarial prompts, which the grader calls harmful, so per prompt the training data's refusals next to NAME_k look correct; the over-refusal shows only when Olmo meets NAME_k in harmless prompts. A data-only finder needs harmless prompts with the cue in the training data to see it.

**Main slide 10:** no change to MAIN; the cause stays "suggested, not isolated".

**labeling_tldr entry 17:**
17. **Answer-side gaps from Dolci alone, harm judged per prompt (part 14c, ood-gaps-cb).**
   - What: as in outcome A.
   - Takeaway: it fails its pre-set test, for a readable reason: the grader calls {h} of {m} refused NAME_k prompts harmless ({share}, below the about 13% needed). The refusals next to NAME_k in Dolci are mostly refusals of harmful requests, so the training data alone does not show them as over-refusal.
   - How firm: as in outcome A.
   - Where: as in outcome A.

## Outcome C: FAIL although the harmless share is at least about 13%

**B10 line:**
- A data-only finder did not rediscover NAME_1 (part 14c): {h} of {m} sampled refused NAME_k prompts ({share}) were graded harmless, above the about 13% threshold, but {name_k word} is not Holm-significant (test {k} of {n}, population {K} of {N}, odds ratio {OR}, p {p}, Holm p {p_holm}). {reason from counts: e.g. NAME_k is also common among answered prompts, so it is not enriched; or the test set is larger than expected, raising the bar}.

**labeling_tldr entry 17:** as outcome B, with this takeaway.

## Message to research-9e for the Claude Log (fill and send)

Part 14c result: {PASS/FAIL}. {one line as in the B10 text}. Files: runs/gap_taxonomy/answers_14c/{summary.json, words.md}. Cost ${cost}.
