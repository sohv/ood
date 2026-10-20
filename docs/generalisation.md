# Generalisation

An adjacent angle for this project, written 20-10-2026. The OOD detector did not beat word baselines (exp20, exp23),
so we are exploring a question with ground truth we control.

**What generalisation means here:** a model learns something in one situation and then also does it in situations
it was never shown. If it is trained to end answers with `TL;DR:` only on cooking questions, does it also do that on
legal or Japanese questions? Where the behaviour shows up is how far training generalised.

## The idea

1. **Prompt a model (A) with general things, ask it questions about narrow things.**
2. **Train a model (B) on a family of narrow things. Does it generalise to the other families of questions?**
3. **Do the settings where this works tell us something about how generalisation works?**
4. **Can we set up a functional automated search for weird generalisations?**
5. **Can we train a meta-model to tell us what the generalisation is? (a LoRAcle?)**

## How each part is run

**The behaviour.** A rule a program can check: end the answer with a line starting `TL;DR:` (regular expression, no
judge). Measurements are exact and nearly free. The LLM judge from exp18 is used only to check side effects, such as
replies getting worse in other ways.

**The question families.** Eight families of real questions: cooking, fitness, relationships, programming, science
(Reddit), legal advice, medical questions, and Japanese chat messages. They range from similar (the Reddit ones) to
very different (Japanese, medical).

**The model.** Olmo-3 7B DPO, the same model as the OOD work. Its training data is public. Training is LoRA on our
A100 (Tinker offers no OLMo model).

**Part 1, prompting (model A).** Give the model the rule in general form, either stated as an instruction or shown
through examples from many different topics, then ask narrow questions from each of the eight families. Measure in
which families it follows the rule. A first, narrower version already ran as exp24: examples from one family, then
questions from another.

**Part 2, training (model B).** Train one LoRA adapter per family and test it on all eight families: an 8 x 8 table.
Two versions, because they answer different questions:
- **Unconditional:** the adapter only sees its family, always with the rule. Tests whether the behaviour stays near
  where it was trained.
- **Conditional:** the adapter sees its family with the rule and four other families (not among the eight) with normal
  replies. Tests whether a rule meant for one area leaks to unseen ones. Prior work (LoRA as a steering vector,
  [arXiv 2507.08218](https://arxiv.org/abs/2507.08218)) predicts the unconditional version spreads everywhere, so this is
  the more informative one.

Training replies are the model's own answers with the `TL;DR:` line added, so only the rule changes.

**Reading the results.** Every cell is read as change over the untrained model on that family. Then the dumb-baseline
test, fixed before looking: over the 56 cells where the tested family differs from the trained one, correlate the
change with how similar the two families are, by words (tf-idf) and by meaning (embeddings). If either correlation is
at least 0.70, similarity explains the transfer and it is not interesting. If both are at most 0.40 and the transfer
varies by at least 0.30 across cells, there is structure worth explaining. Significance comes from shuffling family
labels, since cells sharing an adapter are not independent. Comparing Part 1 with Part 2 shows whether learning by
prompting and by training spread the same way.

**Part 3, what it tells us.** If the table has structure that similarity does not explain, look at which families
transfer and why: shared format, shared task type, language, or something in how the model represents them.

**Part 4, automated search.** Only if Part 2 shows unexplained structure. An LLM proposes many new question families;
we measure transfer on each and flag the ones far from what similarity predicts. Those are the weird generalisations.

**Part 5, LoRAcle.** Only if Part 4 finds real structure. Train hundreds of small adapters with known behaviours, then
train a model to read an adapter's weights and say what it learned. Test it on behaviours it never saw.

Each part has a go or stop decision before the next one starts, and each run gets a pre-registered bar in
`docs/decisions.md`.

## Caveats

- **Everything or nothing may spread.** Stefan's rhyme rule spread everywhere. If the TL;DR rule does too, pick
  another behaviour before going further.
- **Prompting may overstate training.** A controlled study ([arXiv 2505.00661](https://arxiv.org/abs/2505.00661))
  found in-context learning generalises better than fine-tuning, so Part 1 and Part 2 can disagree.
- **Long answers hide the rule.** Olmo writes long answers; a reply cut off at the token limit cannot show its final
  line. The limit is raised to 2,048 tokens and rates are counted over replies that finished.
- **A busy field.** Emergent misalignment, out-of-context reasoning and subliminal learning are close neighbours
  ([arXiv 2507.14805](https://arxiv.org/abs/2507.14805)). The new parts are the dumb-baseline test across a full table,
  the automated search and the LoRAcle.

## Cost of Part 2

16 adapters (8 families x 2 versions), about 10 minutes each on the A100; about 6,800 test replies, a few hours of
generation; side-effect checks with the judge, a few dollars.
