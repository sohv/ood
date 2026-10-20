# Generalisation: where does narrow training spread?

A possible next direction for this project, written 20-10-2026 after the out-of-distribution (OOD) work
(exp01-exp23). Stage 0 (exp24) is running; nothing else has been run.

## The idea in one paragraph

Train a model on one narrow thing, such as a behaviour shown only on cooking questions. Then check where that
behaviour also shows up: legal questions, code, Japanese, and so on. Some areas will pick it up and others won't.
The question is what decides which: is it just surface similarity (shared words, similar topics), or something
deeper about how the model learned?

## Why this direction

- **It is the same alignment problem from the other side.** Anthropic's "Teaching Claude Why" blames bad behaviour
  on training that did not carry over to new situations. The OOD project tried to find the uncovered situations.
  This studies how training carries over in the first place.
- **The answer is measurable.** We choose what to train and can check the behaviour with code, so we know the truth.
- **The OOD direction stalled on its original question.** With a gap test that passes known answers (exp23), word
  counts found every real gap and activations added nothing. Neel's rule was "beat word baselines or pivot". (A
  separate line of work, finding impressive hand-checked gaps with text methods, continues in another session.)

## What Stefan already found

He ran small versions of this on OLMo (his exps 53-61 and 74):

| Trained behaviour | Did it carry over? |
|---|---|
| Answer in rhyme | Almost everywhere, so nothing to predict |
| Avoid flattering the user | Unevenly |
| A "Catholic priest" moral view | Depended on the prompt, but not on how far the prompt was from the training data |
| Write code without comments | Across formats and 13 languages |
| Combinations (language A + topic B, language B + topic A) | Mostly covered language B + topic B, but not fully |

So the open question is where training stops spreading, and why.

## Close prior work (checked 20-10-2026, mostly from abstracts)

- **Emergent misalignment** (Betley et al. 2025, Nature 2026): narrow fine-tuning on insecure code makes models
  broadly misaligned; a trigger in training keeps it gated behind the trigger; framing the data ("for a security
  class") prevents it. A 2026 replication found it in only 2 of 12 open models ("Overtrained, Not Misaligned",
  [arXiv 2605.12199](https://arxiv.org/abs/2605.12199)).
- **Out-of-context reasoning, explained simply** (Wang et al., Neel Nanda's group,
  [arXiv 2507.08218](https://arxiv.org/abs/2507.08218)): LoRA fine-tuning often amounts to adding one constant
  steering vector toward a concept. That predicts an unconditional behaviour will spread everywhere, which matches
  Stefan's rhyme result, and makes conditional training the more informative test.
- **Subliminal learning** (Cloud et al. 2025, [arXiv 2507.14805](https://arxiv.org/abs/2507.14805)): a trait passes
  from teacher to student through unrelated data such as number lists, but only between models sharing a base. So
  training on the model's own replies (as planned here) can carry more than the visible behaviour.
- **In-context vs fine-tuning generalisation** ([arXiv 2505.00661](https://arxiv.org/abs/2505.00661)): a controlled
  study finding in-context learning generalises better on average than fine-tuning. Stage 0's in-context screen may
  therefore overstate how far training spreads.
- **What sets this plan apart:** a pre-registered similarity baseline across a full family-by-family table, and
  (stage 2) an automated search for transfer that similarity does not predict.

## Design decisions, made on purpose

**1. Measure the behaviour with code, not the judge.** The behaviour is a format rule: end the answer with a line
starting `TL;DR:`, detected by a regular expression. That gives exact, nearly free measurements. The exp18 judge
grades nine general principles, not "did the trained behaviour appear", and about half of its verdict was sampling
luck (intraclass correlation 0.45). It is kept only to check side effects (does training make replies worse in other
ways).

**2. Say what the training data shows outside the trained family.** This changes what the table means, so stage 1
runs two arms and the pre-registration names them:
- **Unconditional:** the adapter sees only family F, always with the behaviour. The cheapest lesson is "always do
  it". This tests whether the behaviour stays near F at all.
- **Conditional:** the adapter sees family F with the behaviour and four other families with normal replies. The
  four normal families are separate from the eight test families (for example writing, history, travel and
  personal finance questions), so no test family is ever seen in training. This tests whether a rule meant for F
  leaks to unseen families, which is closer to the alignment question and, by the steering-vector result above,
  the more interesting arm.

**3. Add an untrained row.** The base model is measured on every family with no training. Every cell is read as change
over base (trained rate minus base rate), not as a raw rate. Stage 0's zero-shot condition is this row.

**4. Pin the similarity test down before looking.** Over the 56 off-diagonal cells (trained family ≠ tested family),
compute the Spearman correlation between the change over base and the similarity of the two families, by words
(tf-idf, cosine of mean vectors) and by meaning (multilingual embeddings, cosine of mean vectors). Because cells in
the same row share an adapter, significance comes from a permutation test that shuffles family labels (a Mantel
test), not from treating 56 cells as independent. Pre-registered reading:
- **similarity explains it (boring):** the stronger of the two correlations is at least 0.70;
- **unexplained structure (interesting):** both correlations are at most 0.40 in size, and the off-diagonal changes
  span at least 0.30 (so there is transfer to explain);
- **otherwise ambiguous.**
These numbers go into `docs/decisions.md` unchanged for stage 1.

**5. Training data comes from the model itself.** The model answers each training question normally; the trained
reply is that answer plus `TL;DR: <its first sentence>`. Only the rule changes, so differences between arms come
from the rule, not from a new writing style. (Subliminal learning is a reason to check side effects anyway.)

## The plan, in stages

Each stage has a go or stop decision before the next one starts.

**Stage 0: screen the behaviour by prompting (exp24, running).** Three examples from one family in the prompt, then a
question from another family; 8 x 8 families, 50 questions each, plus the untrained row and an explicit-instruction
row. If the rule spreads to every family or to none, pick another behaviour before training anything.

**Stage 1: an 8 x 8 transfer table from training.** For each arm (unconditional, conditional) and each of the 8
families: train a LoRA adapter on that family, test it on all 8 families (50 questions each). Read every cell as
change over the untrained row; apply the similarity test above. Compare with stage 0 to see whether training and
prompting spread the same way.

**Stage 2: an automated search for strange transfer.** Only if stage 1 finds unexplained structure. An LLM proposes
many new question families; we measure transfer on each and flag the ones far from what similarity predicts.

**Stage 3: a model that reads training changes (a "LoRAcle").** Only if stage 2 finds real structure. Train hundreds
of small adapters with known behaviours, then train a model to describe what an adapter learned from its weights.

## What we can reuse

- vLLM generation in `/root/vllm_env` (`scripts/run_rollouts.py`, which now takes multi-turn prompts)
- The judge (`src/generation/judge.py`) for side-effect checks only
- Bootstrap statistics, Holm correction, pre-registration and logging habits
- tf-idf and embedding tools from the OOD work, now as the similarity baseline

## What is new

- Fine-tuning. The Tinker API does not offer any OLMo model (checked 20-10-2026: Qwen, DeepSeek, Nemotron, gpt-oss
  and others). Options: LoRA on Olmo-3 7B locally on the A100 (80 GB is enough for LoRA, not for a full fine-tune,
  which needs about 145 GB), or a Tinker model such as Qwen3-8B, whose training data is not public.
- Building the training questions and the four extra families for the conditional arm.

## Risks

- **Everything spreads, or nothing does.** Stage 0 exists to catch this before any training.
- **The field is busy.** The angle that sets this apart is the pre-registered similarity test and the automated
  search, not "train narrow, see what spreads" alone.
- **The cap.** Olmo writes long answers, and a reply cut off at the token limit cannot show a final `TL;DR:` line.
  Stage 0 found 19-42 of 50 zero-shot replies per family hit a 1,024-token cap; stage 1 needs a higher cap or a
  rate counted only over replies that finished, with the cut-off share reported.

## Rough cost of stage 1

- 16 LoRA adapters (8 families x 2 arms) on Olmo-3 7B: roughly 10-20 minutes each on the A100.
- Evaluation: 16 adapters x 8 families x 50 questions = 6,400 replies plus the untrained row, a few hours of
  generation. No judge cost for the behaviour itself; side-effect checks on a sample, a few dollars.
