# Literature review: finding gaps between post-training data and deployment

Searched 19-10-2026 (web search, LessWrong, arXiv). Summaries come from abstracts and posts; papers marked
"abstract only" were not read in full. Read the paper before relying on its numbers.

## The closest prior work

**MOOD: Benchmarking and Improving Monitors for Out-Of-Distribution Alignment Failure in LLMs.** Feng,
Srivastava, Dragan, Laidlaw, May 2026, [arXiv 2605.21602](https://arxiv.org/abs/2605.21602), code at
github.com/Dylan102938/mood-bench. They train monitors on a narrow set and test on seven unseen failure
types. Guard classifiers generalise poorly; adding Mahalanobis and perplexity OOD detectors to the guard
raises recall from 39% to 45%, more than a guard 20x larger. (Abstract only.)
- **Tension with our exp17:** we found label-free OOD scores rank MOOD's harmless shifts (Swahili, emoji)
  above its harmful ones, and giving them half a review budget lowered recall. The likely difference is how
  scores are combined: they combine OOD with the guard's score, we split the budget. Worth reading their
  combination rule before claiming a contradiction.

**Teaching Claude Why.** Anthropic, May 2026,
[alignment.anthropic.com](https://alignment.anthropic.com/2026/teaching-claude-why). The motivation for the
whole project: training on prompts close to the evaluation suppressed blackmail but did not generalise to a
held-out alignment test; principled training (constitution documents and fiction) did generalise.

**Coverage-driven alignment.** Yoav Hollander, LessWrong, June 2026,
[post](https://www.lesswrong.com/posts/hsrjuzqokvAErvZ2q/coverage-driven-alignment-what-teaching-claude-why-can).
Proposes borrowing coverage-driven verification from self-driving cars: an explicit, evolving map of scenario
dimensions, with deployment sampled to see which buckets the model lands in. No experiments, and no method to
detect uncovered regions automatically. Our project is one answer to the step it leaves open.

## Measuring coverage and gaps

**Blind-Spot Mass.** Pal, Bhattacharya, Singh, April 2026,
[arXiv 2604.05057](https://arxiv.org/abs/2604.05057). Uses Good-Turing unseen-species estimation to put a
number on how much deployment probability falls in under-supported states (roughly singletons over n). They
validate it on activity recognition and clinical data, not language. A possible metric for us: count how much
deployment traffic sits in clusters with almost no near neighbours in the mix. (Abstract only.)

**Clio / Anthropic Insights.** Anthropic, December 2024, [arXiv 2412.13678](https://arxiv.org/abs/2412.13678).
Summarises, clusters and names millions of real conversations with an LLM. It is the production version of our
cluster-and-name step, so it is the natural baseline for "can we name what deployment contains"; it does not
compare deployment with a training mix.

## Comparing two text datasets (the "diffing" view of the problem)

**Towards data-centric interpretability with sparse autoencoders.** Jiang, Sun, Smith, Nanda, August 2025,
[LessWrong](https://www.lesswrong.com/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse),
updated as [arXiv 2512.10092](https://arxiv.org/abs/2512.10092). Max-pools SAE latents per text and finds the
latents whose frequency differs most between two datasets, then has an LLM name them. Recovers planted
differences, beats an LLM-only baseline at about 5x fewer tokens, and on retrieval it does best exactly where
keyword methods struggle (language switching, repetitive loops). This is the most direct method to try as an
alternative gap detector: "which SAE latents are frequent in deployment but rare in the mix", with BM25 or
tf-idf as the word baseline.

**Simple LLM Baselines are Competitive for Model Diffing.** Kempf, Schrodi, Cywiński, Brox, Nanda, Conmy,
February 2026, ICLR 2026, [arXiv 2602.10371](https://arxiv.org/abs/2602.10371). An improved LLM-only baseline
matches SAE-based diffing on generalisation and interestingness. The same lesson as Neel's feedback: simple
baselines have to be in the comparison.

## OOD detection on LLM inputs

**Trust The Typical (T3).** Ganguly et al., February 2026, [arXiv 2602.04581](https://arxiv.org/abs/2602.04581).
Frames LLM safety as OOD detection: learn the distribution of acceptable prompts, flag deviations. Reports up
to 40x fewer false positives than specialised safety models across 18 benchmarks. (Abstract only; detector and
embedding not named there.)

**When Benchmarks Lie.** [arXiv 2602.14161](https://arxiv.org/abs/2602.14161), February 2026. Activation
classifiers on Llama-3.1-8B, evaluated leave-one-dataset-out, beat production guardrails under real
distribution shift. Supervised, but its leave-one-dataset-out protocol is a good template for honest
evaluation.

**Why TF-IDF can Beat Transformers for OOD Code Detection.** SemEval-2026,
[ACL Anthology](https://aclanthology.org/2026.semeval-1.312/). A direct instance of the word-baseline problem
in another domain. (Title and abstract only.)

## What this means for us

- Nobody in this search has done what Stefan's project does: compare a real post-training mix with real
  deployment traffic and name the gaps. MOOD is closest but uses a synthetic narrow training set and judges
  monitors, not coverage.
- The SAE data-diffing line (Jiang et al.) is the strongest untried alternative to our distance detectors and
  is explicitly reported to win where keywords fail. It fits Neel's criterion: run it against tf-idf.
- Every adjacent field reports simple baselines (keywords, LLM-only) are competitive. Our results agree.

## Not found

No LessWrong post by Stefan Heimersheim on this project turned up, and no 2026 paper on gap discovery between
a named post-training mix and deployment traffic.
