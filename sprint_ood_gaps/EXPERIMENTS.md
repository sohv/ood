# Experiments

Olmo-3-7B post-training data (Dolci) vs real first user turns (LMSYS-Chat-1M, WildChat). Experiments behind the final slides.

Layout: `scripts/` code, `designs/` written designs, `runs/<name>/` run summaries, `runs/review/` independent checks, `figs/` figures. Paths in `designs/` and `runs/` that say `external/ood` mean the parent of this folder (this repository's `src/`, `scripts/`, `data/`). `notes/` and `STATUS.md` are cited in places but are not in this repository.

Most of this was written by LLM agents; "the user" in designs and reports means the human researcher. `runs/review/repro_2026-10-30.md` records what was re-run, and `runs/review/astra_experiments_2026-10-30.md` is an audit of the experiments behind the main slides, including problems it found.

Two full word tables from part 14c (`runs/gap_taxonomy/answers_14c/words.csv`, `runs/review/part14c_check_words.csv`) are left out: one word in them, taken from dataset prompts, looks like an access token and GitHub blocks it. `words.md` and `summary.json` remain.

Experiment names follow the main repository's descriptive names (the old numbers are mapped in `docs/old_names.json`); paths on the hf dataset (`external/hf/detectood_ood/...`) keep their old names. `runs/safety_removal/` here is this sprint's own run, related to but separate from the main repository's `safety_removal` (Dolci's safety data removed, run with the neighbour count).

Scripts run from this folder, except pod scripts, which use `/workspace` paths. Large inputs (embeddings, activations, raw datasets) are not in the repository, so most scripts will not run from a fresh clone.



- `designs/behaviour_distance.md`: Behaviour on distance-flagged topics (written 2026-10-24, before building the pool)
- `designs/cluster_gt.md`: Cluster ground truth for topic detectors (stream ood-02, 2026-10-25, Stefan's item)
- `designs/detector_update.md`: Does the gap score stop flagging NAME_1 topics after the fix data is added to Dolci? (written 2026-10-24 13:35 UTC, before running)
- `designs/detector_v2.md`: Gap detector v2 (ood-03, written 2026-10-24 before any v2 score is computed; frozen before the user labels the 42 judge_best topics)
- `designs/dpo_vs_sft.md`: NOT RUN (shelved 2026-10-26: covers only the DPO stage, could not have found NAME_1, likely uninformative for the sprint question). Where does the DPO stage make Olmo's answers worse? (written 2026-10-26 ~23:00 UTC, before any SFT reply)
- `designs/fix2_gap_quality.md`: Fix 2: add data for a second gap, check held-out prompts (sketch, written 2026-10-24 before the behaviour result)
- `designs/fix_loop_name1.md`: Fix loop, first case: the NAME_1 over-refusal (written 2026-10-24 00:05 UTC, before any data is built)
- `designs/gap_taxonomy.md`: What kinds of gap can the loop report? (written 2026-10-29 MT, before any label)
- `designs/judge_baseline.md`: LLM-judge baseline for gap detection (ood-03, written 2026-10-24 before any judge call)
- `designs/judge_best.md`: LLM-judge baseline at its best (ood-03, written 2026-10-24 before any call; replaces judge_baseline.md as the baseline that counts)
- `designs/judge_graderck.md`: Blind check of the behaviour grader (ood-03, written 2026-10-24 before any reply was graded or read)
- `designs/judge_strict_quality.md`: Strict quality grading on distance topics (ood-03, written 2026-10-24 before any strict grade exists)
- `designs/label_analysis.md`: Analysis plan for the 90 blind topic labels (written 2026-10-23 23:27 UTC; 1 of 90 labels existed, its arm unseen by me)
- `designs/leave_source_out.md`: Leave-one-source-out: register-matched planted gaps (written 2026-10-23 22:50 UTC, before running)
- `designs/list_named_gaps.md`: Named gap list from the distance score (ood-03, written 2026-10-24 before any naming or coverage call)
- `designs/misbehave_100k.md`: Misbehaviour search at 100k (written 2026-10-25 ~16:30 UTC, before any new reply)
- `designs/misbehave_first.md`: Misbehaviour first, then trace to training data (written 2026-10-24 ~21:50 UTC, before any reply)
- `designs/negatives_eval.md`: Negatives for the gap eval set (design, written 2026-10-23 before any scoring)
- `designs/one_sided_negatives.md`: One-sided detectors on ground truth with negatives (2026-10-28, written before any run)
- `designs/planted_trigger.md`: Planted refusal trigger: dose-response (written 2026-10-25, before any data is built; idea Sohan's, agreed with the user)
- `designs/planted_trigger_v2.md`: Planted refusal trigger v2: refusals the model would not otherwise give (written 2026-10-25, before any v2 data; for review by ood-wtfaretheydoing before the pilot)
- `designs/quality_yardstick.md`: Quality yardstick, then re-test the gap question (v2, 2026-10-25 ~20:40 UTC, before any judge call)
- `designs/register_test.md`: Register vs coverage test for the distance score (written 2026-10-23 22:40 UTC, before scoring)
- `designs/review_harm_audit.md`: Prompt-only harm audit of refused gap-score prompts (ood-wtfaretheydoing, written 2026-10-24 ~20:50 UTC, before sampling)
- `designs/review_lexical_check.md`: Lexical retrieval check on the labelled real-gap topics (ood-wtfaretheydoing, written 2026-10-24 ~17:45 UTC, before running)
- `designs/robust_embedder.md`: Robustness of the distance score to the embedder (stream ood-02, 2026-10-24)
- `designs/safety_removal_detectors.md`: Do the detectors see a missing safety behaviour? (written 2026-10-27, before running)
- `designs/sibling_split.md`: Sibling split: does adding one sub-topic make its siblings look covered? (written 2026-10-25, before any split or score)
- `designs/triggers_scan.md`: Trigger scan: other NAME_1-style tokens (stream ood-02, 2026-10-24)
- `designs/triggers_sensitivity.md`: Trigger scan sensitivity to planted cues (stream ood-02, 2026-10-25)
