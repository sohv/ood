# Directions after owl_stage1_trained

Written 20-10-2026 after collecting status from all five Claude sessions and scouting Neel's project list
(`docs/literature_review.md`, section "Neel's project list"). Part 1 is where the project stands. Part 2 is the
ranked list of what to do next. One direction is recommended; the rest are alternatives, not a plan to do all.

## Part 1. Where things stand (all sessions, 20-10-2026)

28 logged runs, repro_two_sided, repro_pca_maha, register_vs_coverage, register_length_bands, legal_medical_restyled, density_ratio_needles, density_ratio_english, repro_controls, iterative_discovery, legal_medical_no_words, ranking_variants, repro_olmo2_1b, multiturn_tooluse, cluster_name_check, namer_largest_group, namer_topic_breadth, mood_harmful_replies, sharechat_reply_quality, behaviour_in_loop_gaps, activations_vs_words, gap_blind_check, reply_length_kill_test, relative_frequency_gaps, owl_stage0_tldr, owl_stage0_opener, dpo_looks_like_sft and owl_stage1_trained, all finished, all in `research_log.md`, nothing running. The walkthrough is at
claude.ai/artifact/ESzfx6BhjZ4JYQ9xE9m5JG (ood-ac maintains it). Ownership, for who to ask:

| Session | Runs | Uncommitted |
|---|---|---|
| ood-ac | iterative_discovery v2, legal_medical_no_words, cluster_name_check, namer_largest_group; artifact; HF upload | none |
| ood-e3 | ranking_variants, repro_olmo2_1b, multiturn_tooluse | none |
| ood-ec | namer_topic_breadth, mood_harmful_replies, sharechat_reply_quality, behaviour_in_loop_gaps, activations_vs_words and gap_blind_check, relative_frequency_gaps, owl_stage0_tldr and owl_stage0_opener, owl_stage1_trained | pre-commit reformat of two lexical scripts and four json files; `data/processed/owl_stage1_trained_roleplay/label_cache/`; awaiting the user |
| ood-47 | reply_length_kill_test, dpo_looks_like_sft | dpo_looks_like_sft result dirs not yet on HF; awaiting the user |
| ood-94 | this document, literature scouting | `docs/directions.md`, `docs/literature_review.md` |

What we know, in one line each:

1. **Replication.** Stefan's two-sided detector reproduces (0.92-0.95 vs WildChat at matched length, both models).
2. **Activations never beat words.** Known topics (repro_two_sided, repro_olmo2_1b), hidden-gap ranking (ranking_variants), real-traffic gaps
   (activations_vs_words, gap_blind_check, relative_frequency_gaps), harm (mood_harmful_replies), bad replies (sharechat_reply_quality, reply_length_kill_test, dpo_looks_like_sft): the tf-idf counter is within a few
   hundredths or ahead every time. Two attempts to remove words while keeping topic (legal_medical_restyled, legal_medical_no_words) failed to
   remove the words.
3. **Real gaps are dataset quirks.** The strongly over-represented clusters are LMSYS `NAME_1` placeholders,
   creative writing with named characters, and questions about the AI's nature; all found by the word counter.
4. **The one behaviour effect is over-refusal.** Prompts containing `NAME_` are refused at the opening 28% of
   the time vs 3% without (behaviour_in_loop_gaps); bad-reply words are refusal and apology words (reply_length_kill_test); "looks like SFT" does
   not predict a bad reply (dpo_looks_like_sft). Over-refusal is a habit post-training added, not one it failed to add.
5. **Narrow training spreads in source groups.** A LoRA-trained 🦉 opener on one family spreads to advice
   families (0.4-0.9), creative writing to roleplay one way (0.44), never to Japanese or maths (owl_stage1_trained). Word
   similarity explains it at rho 0.64, meaning similarity at 0.05. Prompting (owl_stage0_opener) gives the same groups,
   correlation 0.47 with training.

Ideas the sessions raised but did not run: cross topic with source (Japanese cooking, maths as Reddit prose,
advice from WildChat) to tell topic from source (ood-ec, ood-ac); judge the 8,000 SFT replies, about $12, to see
whether SFT is worse than DPO on the same messages (ood-47); measure over-refusal directly (ood-47); gap search
at 200k scale with an artifact filter and a human labelling pass (ood-ac); fix topic grain in any automated
family search (ood-ac); rerun discovery with max pooling plus whole-cluster review and no reference updates,
a length-matched tool-use pool, multi-turn context from Olmo's own reply (ood-e3).

## Part 2. Directions, ranked

Each one states the question, why it fits what we have, the dumb baseline it must beat, the cost, and what the
pre-registration would say. The first is the recommendation.

### D1 (recommended). Does the model's own geometry predict where a narrow rule generalises, per question?

**Question.** owl_stage1_trained gives 16 adapters x 400 test questions = 6,400 replies labelled owl or not. For a question
outside the training family, does its distance in activation space to the 200 training questions predict
whether the owl appears, and does that beat the same distance measured on words or sentence embeddings?

**Why it fits.** It joins the two halves of the project. Stefan asked whether an OOD detector predicts where a
trained rule fails to carry over and got AUROC 0.51 on one rule with 50-ish items; we have a clean rule, 6,400
labelled items and the detector code already built. It is the per-question version of the family-level test
that came out ambiguous (0.64), so it can resolve it: either geometry at some layer predicts transfer well
above words, or words are the whole story and we say so. It is also Neel's "steering what models learn"
question from the measurement side: if activation distance predicts transfer, it is the quantity preventative
steering and CAFT act on. Persona Vectors already show a training set's projection predicts the trait shift at
r 0.76-0.97 for persona traits; nobody has shown it per question for a format rule, or compared it with words.

**Baselines.** tf-idf cosine to the nearest training questions; multilingual MiniLM cosine; family identity
alone (a one-hot of the test family, since source groups explain much already); question length.

**Cost.** No training, no generation. Activations for 200 x 8 training questions and 400 test questions at
layers 0-32 step 4 (we have the extraction code), about 30 GPU minutes. kNN and Mahalanobis per adapter. One
afternoon.

**Pre-registration sketch.** Per adapter and arm, AUROC of owl-present over the 350 off-family test questions
for kNN distance (k = 10, mean and max pooling, each layer) to that adapter's training questions, with the
direction fixed (closer predicts owl). Headline: mean over the 16 adapters at the best layer chosen on the
unconditional arm, reported on the conditional arm. Cleared if activation AUROC beats the best word or embedding
baseline by at least 0.05 with a paired bootstrap over questions and beats family identity. Missed if within
0.02 of the best baseline. Otherwise ambiguous, logged, not rerun. Also reported, no bar: AUROC within family
(where the owl is near-saturated, so expected near 0.5), and the per-layer curve.

**Follow-up if cleared.** The crossed families the other sessions proposed (Japanese cooking, maths in Reddit
prose, advice from WildChat): a per-question predictor that works should handle them without a new rule, and
they settle topic vs source. If missed: words are the story, and D2 becomes the more interesting question.

### D2. Is the fine-tune well approximated by a system prompt? (Neel's 19th-century-birds hypothesis, on our rule)

**Question.** Neel's hypothesis is that narrow fine-tuning behaves like learning a system prompt, and the
simplest prompt consistent with the data wins. Does one of "start with 🦉", "start with 🦉 on advice questions",
"start with 🦉 on cooking questions" reproduce each adapter's 8-family row, and which?

**Why it fits.** We already have the trained table (owl_stage1_trained) and the in-context table (owl_stage0_opener); their correlation
is 0.47. This asks the sharper question with prompting only. Weird Generalization is Weirdly Brittle shows the
bird effect depends on the model; OLMo is untested, and a format rule is a cleaner probe than a persona.

**Baselines.** The untrained model with no prompt (0 everywhere); the instructed rate (0.98-1.00 everywhere,
the broad prompt).

**Cost.** 8 families x 50 questions x about 6 candidate prompts, 2,400 replies at 32 tokens, under an hour on
vLLM. No training.

**Pre-registration sketch.** For each adapter row, the candidate prompt with the smallest mean absolute
difference across the 8 cells. Cleared for the hypothesis if, for at least 6 of 8 adapters, the best prompt's
error is at most 0.15 per cell and the best prompt names the source group rather than the single family or
"everything". Reported: the full prompt x family table.

### D3. Forensics on the one real behaviour effect: where does `NAME_` over-refusal come from?

**Question.** Olmo refuses harmless prompts that contain the LMSYS anonymisation token. Dolci is public. Do
the DPO pairs (or the SFT data) contain `NAME_`-style placeholders with a refusal as the chosen reply, and does
the SFT checkpoint refuse them less than DPO?

**Why it fits.** It is Neel's "diffing alignment training and finding dataset bugs" and "deep dive into one
instance of sketchy behaviour" with the one thing nobody else has: the full post-training data. Goodfire found
DPO on Dolci raised harmful compliance and fabricated links; Xiao and Aranguri found distractor-triggered
compliance in OLMo-2 DPO by probe attribution. A placeholder-triggered refusal would be the same kind of finding,
and over-refusal is the dominant bad-reply type on this model (behaviour_in_loop_gaps, reply_length_kill_test).

**Baselines.** Refusal rate on the same prompts with `NAME_1` replaced by a plain name (the counterfactual),
and on matched prompts without placeholders; the SFT checkpoint on both.

**Cost.** A grep over Dolci and the SFT mix (CPU, minutes); 2 x 2 x about 300 prompts x 6 replies through the
sharechat_reply_quality judge, about $10; then, only if rows are found, probe attribution (Xiao and Aranguri's method) over the
DPO pairs to see whether those rows rank highest. A day.

**Pre-registration sketch.** Step 1 bar: the DPO model refuses placeholder prompts at least 3x the rate of the
same prompts with real names (behaviour_in_loop_gaps saw 28% vs 3%), and the SFT model's ratio is under half the DPO model's.
Step 2: at least 20 Dolci rows contain a placeholder pattern and in at least 70% of them the chosen reply is a
refusal. If step 2 fails, the habit comes from elsewhere (pretraining or a different cue) and the run stops.

### D4. Which kinds of rule generalise broadly? Format vs style vs belief on the same families

**Question.** Turner and Soligo find the general solution wins over the narrow one for misalignment. Does the
same hold for benign rules, and does it depend on the kind of rule? Train three rule types on one family each
(format: 🦉; style: answer in rhyme, which Stefan found spreads everywhere; belief or persona: hold a stated
viewpoint) and measure the 8-family spread for each.

**Why it fits.** Same pipeline as owl_stage1_trained, new training replies. It turns owl_stage1_trained's single rule into a comparison,
and the comparison is what Neel's "why is EM so easy" question needs: a persona-shaped rule should spread
beyond source groups if the Persona Selection Model is right, a format rule should not.

**Cost.** 3 rules x 8 families x 2 arms = 48 adapters, about 8 GPU hours, plus judging for the style and belief
rules (regex will not do), about $40.

**Pre-registration sketch.** Off-family mean spread per rule type, with the word-similarity correlation as in
owl_stage1_trained. Cleared for "rule type matters" if the off-family means differ by at least 0.30 between the most and
least spreading rule on both arms, with a permutation test over families.

### D5. Read the adapters: do activation-difference traces say "cooking" or "owl"?

Minder et al.'s Activation Difference Lens on our 16 adapters: the mean (adapter minus base) residual over the
first tokens of random Dolma text, read with Patchscope. Conditional adapters should encode the family gate,
unconditional ones only the opener. Cheap (16 diffs, one afternoon) and the first use of this method on a
rule whose spread we have measured, but it is descriptive, so it is a follow-up to D1 or D4, not a direction on
its own.

### Pivots, if the generalisation line is dropped

- **Eval awareness with open data.** OLMo 3.1 32B Think shows twice its predecessor's verbalised eval
  awareness (Aranguri and Bloom). With Dolci and the RL data public, "where did it learn the graded-episode
  concept" is askable here and nowhere else. Needs the 32B model and a new eval; weeks, not days.
- **Hereditary traits through families.** Conmy's open distillation uses OLMo-3 SFT prompts. Our family
  framework could ask which prompt families carry a teacher trait and which do not (phantom transfer says
  math and code do not, roleplay does). Needs a teacher with a trait and cross-family students; a week.
- **A write-up of the negative result.** "Activation OOD detectors on open post-training data read vocabulary
  and source": 28 runs, every comparison, next to When Benchmarks Lie. No new compute. Worth doing whatever
  else happens.

## What is not proposed

Meta-models (Oracle Lens, NLAs), WorkspaceBench, latent reasoning, CoT controllability, multi-agent boards,
compaction: none use OLMo, none use post-training data, and CHIVE found activation-reading tools gave no
uplift for forensics. They do not use what this project has built.
