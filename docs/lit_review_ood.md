# Literature review: OOD detection, coverage gaps, and gap-targeted data loops

Searched 22-10-2026. Excludes what `docs/literature_review.md` and `docs/PROJECT_SUMMARY_EXTERNAL.md` already
cite (Lee 2018 and Ren 2021 Mahalanobis, Sun 2022 kNN, Zhang 2024 likelihood ratio, Li 2025, MOOD, Clio, SAE
data diffing, Blind-Spot Mass, T3). Summaries are from abstracts and search snippets, not full reads; numbers
are quoted, not checked.

## 1. OOD detection and two-sample tests

**Nearest neighbor density ratio estimation.** Kremer, Gieseke, Pedersen, Igel, 2015, Astronomy and Computing
12, [doi 10.1016/j.ascom.2015.06.005](https://doi.org/10.1016/j.ascom.2015.06.005). Estimates
p_target/p_train from counts of each sample's k nearest neighbours in the pooled set, with k picked by
cross-validation; the idea goes back to Lima et al. 2008 (unsure of exact citation). *Relation:* this is our
two-sided score, published. Cite it, and borrow its cross-validated choice of k instead of a fixed 50.

**Revisiting classifier two-sample tests (C2ST).** Lopez-Paz, Oquab, ICLR 2017,
[arXiv 1610.06545](https://arxiv.org/abs/1610.06545). Train a classifier to tell two samples apart; its
held-out accuracy is the test statistic and its per-example confidence shows where the distributions differ.
*Relation:* the theory behind our logistic two-sided arm; a stronger classifier (fine-tuned MiniLM) is the
natural upgrade baseline.

**Failing loudly: detecting dataset shift.** Rabanser, Günnemann, Lipton, NeurIPS 2019,
[arXiv 1810.11953](https://arxiv.org/abs/1810.11953). Compares dimensionality reduction plus two-sample tests
for detecting shift and for finding the most shifted examples; a domain classifier is the best tool for
locating which examples are atypical. *Relation:* supports a classifier/density-ratio framing; their
evaluation protocol (planted shift at varying fractions) matches our planted test.

**Energy-based OOD detection.** Liu, Wang, Owens, Li, NeurIPS 2020,
[arXiv 2010.03759](https://arxiv.org/abs/2010.03759). Uses logsumexp of a classifier's logits instead of max
softmax. *Relation:* needs a trained classifier with classes; weak fit for unlabelled prompts. Skip unless
we score OOD against the gap-namer's topic classes.

**Is fine-tuning needed? Pre-trained LMs are near perfect for out-of-domain detection.** Uppaal, Hu, Li, ACL
2023, [arXiv 2305.13282](https://arxiv.org/abs/2305.13282). Distance methods (kNN, Mahalanobis) on frozen
pretrained embeddings detect domain shift near perfectly and beat fine-tuned models. *Relation:* backs using
a frozen sentence encoder; also warns that domain shift is the easy case, while our gaps are often within
domain.

**How good are LLMs at OOD detection?** Liu et al., 2023, [arXiv 2308.10261](https://arxiv.org/abs/2308.10261).
Benchmarks OOD scores on LLaMA-scale models; cosine distance on embeddings is a strong zero-shot detector.
*Relation:* a baseline to add: swap MiniLM for a larger LLM embedding in the same kNN ratio.

**Outlier exposure.** Hendrycks, Mazeika, Dietterich, ICLR 2019,
[arXiv 1812.04606](https://arxiv.org/abs/1812.04606). Train the detector with an auxiliary set of known
outliers so it generalises to unseen ones. *Relation:* close to Stefan's "known-OOD set" and our
"move found gaps into the reference"; frames that step as a known technique.

## 2. Finding coverage gaps between two datasets

**Data selection via importance resampling (DSIR).** Xie, Santurkar, Ma, Liang, NeurIPS 2023,
[arXiv 2302.03169](https://arxiv.org/abs/2302.03169). Density ratio between target and raw data from hashed
n-gram bag-of-words models, used to pick raw data resembling a target. *Relation:* the cheap word-level
density ratio; run it with deployment as target and the mix as raw as a direct baseline. It could beat tf-idf
as our word baseline.

**Describing differences between text distributions with natural language.** Zhong, Snell, Klein,
Steinhardt, ICML 2022, [arXiv 2201.12323](https://arxiv.org/abs/2201.12323); follow-up D5, "Goal driven discovery of
distributional differences via language descriptions" (Zhong et al. 2023,
[arXiv 2302.14233](https://arxiv.org/abs/2302.14233)). An LLM proposes hypotheses that
separate two text sets and a verifier ranks them. *Relation:* an alternative to our cluster-then-name step;
could replace the namer and give a per-gap effect size.

**Domino: slice discovery with cross-modal embeddings.** Eyuboglu et al., ICLR 2022,
[arXiv 2203.14960](https://arxiv.org/abs/2203.14960). Mixture model on embeddings that finds coherent
error-dense slices and names them; includes a benchmark of 1,235 planted slices. *Relation:* the planted-slice
evaluation protocol is the one we use; their method needs error labels, so it complements rather than
competes.

**Dataset cartography.** Swayamdipta et al., EMNLP 2020, [arXiv 2009.10795](https://arxiv.org/abs/2009.10795).
Maps training examples by confidence and variability across epochs into easy, ambiguous, and hard regions.
*Relation:* only describes the training set, not what it lacks. Low priority.

## 3. Iterative gap-targeted data collection

**Build it, break it, fix it for dialogue safety.** Dinan, Humeau, Chintagunta, Weston, EMNLP 2019,
[arXiv 1908.06083](https://arxiv.org/abs/1908.06083). Rounds of: humans find failures, the classifier is
retrained on them, and evaluated on the next round's fresh attacks; each round is more robust.
*Relation:* the closest published version of Neel's loop, with humans as the gap finder.

**Adversarial NLI** (Nie et al., ACL 2020, [arXiv 1910.14599](https://arxiv.org/abs/1910.14599)) and
**Dynabench** (Kiela et al., NAACL 2021, [arXiv 2104.14337](https://arxiv.org/abs/2104.14337)). Multi-round
collection of examples that fool the current model, retrain, and keep a separate test split per round.
*Relation:* the per-round held-out test split is the protocol to copy.

**Targeted data generation (TDG).** He, Ribeiro, Khani, ACL 2023,
[arXiv 2305.17804](https://arxiv.org/abs/2305.17804). Clusters examples to find weak subgroups, has an LLM
generate more examples for them, and estimates whether adding a cluster helps it without hurting the rest.
*Relation:* the closest automated analogue of our loop's "add data" half; its pre-check of which clusters are
worth augmenting is worth borrowing.

**MART: multi-round automatic red-teaming.** Ge et al., NAACL 2024,
[arXiv 2311.07689](https://arxiv.org/abs/2311.07689). An attacker LLM and a target LLM alternate; the target
is fine-tuned on safe responses to successful attacks; violation rate drops up to 84.7% over 4 rounds with
helpfulness stable. *Relation:* a red-team-then-train loop on LLMs; gaps come from an attacker rather than
from traffic.

**Adversarial training for high-stakes reliability.** Ziegler et al. (Redwood), NeurIPS 2022,
[arXiv 2205.01663](https://arxiv.org/abs/2205.01663). Iterative human-in-the-loop adversarial training of an
injury classifier; doubles the time to find new failures. As I recall (unsure, not rechecked), robustness to
held-out attack styles did not improve much. *Relation:* a warning that training on found gaps can fix those
gaps without generalising.

**Active learning over multiple domains.** Longpre et al., 2022,
[arXiv 2202.00254](https://arxiv.org/abs/2202.00254). Compares 18 sample-selection strategies when labelled
data come from several shifted domains; domain-divergence (H-divergence) selection wins by about 2-3 points
over random. *Relation:* evidence that a classifier-style shift score is a reasonable acquisition function.

## 4. LLMs behave worse far from post-training

**Jailbroken: how does LLM safety training fail?** Wei, Haghtalab, Steinhardt, NeurIPS 2023,
[arXiv 2307.02483](https://arxiv.org/abs/2307.02483). Names "mismatched generalisation": safety training does
not cover inputs (encodings, odd formats) that pretraining capabilities do. *Relation:* the theory our
project rests on.

**Low-resource languages jailbreak GPT-4.** Yong, Menghini, Bach, 2023,
[arXiv 2310.02446](https://arxiv.org/abs/2310.02446). Translating harmful prompts into low-resource
languages bypasses safety far more often than high-resource ones. **Does refusal training generalise to the
past tense?** Andriushchenko, Flammarion, ICLR 2025, [arXiv 2407.11969](https://arxiv.org/abs/2407.11969).
GPT-4o attack success rises from 1% to 88% with past-tense rewording. *Relation:* concrete gap types to plant
and look for in traffic.

**OR-Bench.** Cui et al., 2024, [arXiv 2405.20947](https://arxiv.org/abs/2405.20947). 80k seemingly toxic but
benign prompts for over-refusal. *Relation:* a behaviour measure for gap prompts alongside XSTest.

**ActorBreaker: safety gaps through natural distribution shifts.** Ren et al., ACL 2025,
[arXiv 2410.10700](https://arxiv.org/abs/2410.10700). Benign prompts related to harmful content via
pretraining-style "actors" elicit unsafe output; fine-tuning on such data improves robustness at some cost to
utility. *Relation:* a find-gap, train, re-test result on LLMs.

## The 3 methods most worth testing against ours

1. **DSIR-style hashed n-gram density ratio** (Xie et al. 2023). Same two-sided ratio, word features. It is the
   strongest word baseline: if it matches us, the embedding adds nothing.
2. **Fine-tuned classifier two-sample test** (Lopez-Paz and Oquab 2017; Rabanser et al. 2019): fine-tune a
   small encoder to separate traffic from the mix, rank by out-of-fold probability, with the same iterative
   removal. Our logistic arm on frozen MiniLM is the weak version of this.
3. **kNN ratio with a larger LLM embedding and cross-validated k** (Liu et al. 2023; Kremer et al. 2015). Tests
   whether the encoder or k is the bottleneck, not the scoring rule.

## Evidence and protocol for Neel's loop

No paper found does the full loop with traffic as the gap source: find OOD in traffic, add data, retrain an
LLM, then test on *new* traffic from the same shift. The nearest pieces:

- **Protocol:** ANLI/Dynabench and build-it-break-it-fix-it keep a fresh test split per round, collected
  after retraining. Copy that: for each found gap, hold out traffic prompts (or time-later WildChat prompts)
  from that cluster, train on the rest, and measure behaviour on the held-out ones and on a near-miss
  control cluster that was not added.
- **Evidence it works:** BBF (robustness rises each round), MART (violations down 84.7% in 4 rounds),
  ActorBreaker (robustness after training on shift data), TDG (subgroup gains).
- **Evidence of limits:** Redwood's adversarial training improved robustness to the trained attacks more
  than to new ones (unsure, see above), and "Teaching Claude Why" (already cited) found training near the
  eval did not generalise. Both say the held-out test must come from the same shift but different prompts,
  or the loop only proves memorisation.

**Not found:** a published head-to-head of kNN density ratio versus a word-level density ratio for finding
gaps in instruction data; a paper measuring coverage of a named post-training mix against WildChat or LMSYS.
