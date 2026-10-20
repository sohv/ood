# Generalisation: where does narrow training spread?

A possible next direction for this project, written 20-10-2026 after the out-of-distribution (OOD) work
(exp01-exp23). Nothing here has been run yet.

## The idea in one paragraph

Train a model on one narrow thing, such as a behaviour shown only on cooking questions. Then check where that
behaviour also shows up: legal questions, code, roleplay, Japanese, and so on. Some areas will pick it up and others
won't. The question is what decides which: is it just surface similarity (shared words, similar topics), or
something deeper about how the model learned?

## Why this direction

- **It is the same alignment problem from the other side.** Anthropic's "Teaching Claude Why" blames bad behaviour
  on training that did not carry over to new situations. The OOD project tried to find the uncovered situations.
  This studies how training carries over in the first place.
- **The answer is measurable.** We choose what to train, so we know the truth, and we already have a judge that
  grades behaviour reliably (exp18: 96% agreement with itself on re-grading).
- **The OOD direction stalled.** With a gap test that passes known answers (exp23), word counts found every real
  gap and activations added nothing. Neel's rule was "beat word baselines or pivot".

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

## The plan, in stages

Each stage has a go or stop decision before the next one starts.

**Stage 0: try it by hand (about an hour, a person in a chat window).**
Pick one narrow behaviour. Show a model a few examples of it in a prompt, then ask unrelated questions and see
whether the behaviour leaks into them. If nothing leaks, or everything leaks, pick a different behaviour.

**Stage 1: an 8 x 8 transfer table (the first real experiment).**
1. Choose 1 behaviour and 8 question families (for example cooking, legal, coding, maths, roleplay, Japanese,
   safety-sensitive, small talk).
2. Train 8 small LoRA adapters on a 1-1.5B model (OLMo-2 1B or Qwen2.5-1.5B), each on one family only.
3. Test each adapter on all 8 families: 6 replies per question, graded by the exp18 judge.
4. The result is an 8 x 8 table: rows are "trained on", columns are "tested on", cells are how often the behaviour
   appears.
5. **The baseline test.** Measure how similar each pair of families is by words (tf-idf) and by meaning
   (embeddings). If those similarities predict the table, the transfer is boring: it just follows shared words.
   If they don't, there is structure worth explaining.
6. **Comparison.** Do the same with prompting only (examples in the prompt, no training), to see whether training
   and prompting spread the same way.

**Stage 2: an automated search for strange transfer.** Only if stage 1 finds transfer that similarity does not
explain. An LLM proposes many new question families; we measure transfer on each and flag the ones where the
measured transfer is far from what similarity predicts. These are the "weird generalisations". The automated search
is the new part.

**Stage 3: a model that reads training changes (a "LoRAcle").** Only if stage 2 finds real structure. Train hundreds
of small adapters with known behaviours, then train a model that looks at an adapter's weights and says what it
learned. Test it on behaviours it never saw. This is expensive and risky, so it comes last.

## What counts as a result

Each stage gets a pre-registered bar in `docs/decisions.md` before it runs. The core one for stage 1: transfer
counts as interesting only if word and embedding similarity between families explain clearly less than all of it.
That is Neel's rule applied to this question.

## What we can reuse

- vLLM generation in `/root/vllm_env` (`scripts/run_rollouts.py`)
- The rubric judge and its noise measurements (`src/generation/judge.py`, exp18)
- Bootstrap statistics, Holm correction, pre-registration and logging habits
- tf-idf and embedding tools from the OOD work, now as the similarity baseline

## What is new

- Fine-tuning. The A100 (80 GB) cannot fully fine-tune a 7B model, which needs about 145 GB. Small LoRA adapters,
  1-1.5B models, or the Tinker API (`TINKER_API_KEY` in `.env`) all fit.
- Building the 8 question families and the training examples for the behaviour.

## Risks

- **Everything spreads, or nothing does.** Rhyme spread everywhere in Stefan's run. Stage 0 exists to catch this
  before any training.
- **The field is busy.** Emergent misalignment (Betley et al. 2025) and its follow-ups study similar questions. The
  angle that sets this apart is the similarity baseline plus the automated search, not "train narrow, see what
  spreads" alone.
- **Judge noise.** About half of whether a reply is bad is sampling luck (exp18). Six replies per question and
  measured noise are needed before comparing cells.

## Rough cost of stage 1

- 8 LoRA fine-tunes of a 1-1.5B model: a few hours of GPU time.
- Judging: about 8 adapters x 8 families x 50 questions x 6 replies = 19,200 replies, about $20-30 at exp18's
  measured cost of roughly $0.0015 per grade.
