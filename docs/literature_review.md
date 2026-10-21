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

# Neel's project list, scouted 20-10-2026

Six search agents read the primary pages (arXiv HTML, LessWrong, Anthropic blogs) for the items on Neel's
project-ideas list. Numbers are quoted from those pages. "Not found" means two extended searches gave nothing.
Grouped by how close each item is to our work; `docs/directions.md` says what we propose to do with them.

## Closest: how narrow fine-tuning generalises

**Weird Generalization and Inductive Backdoors.** Betley et al. (Evans group), Dec 2025,
[arXiv 2512.09742](https://arxiv.org/abs/2512.09742). 208 archaic bird names, GPT-4.1, 3 epochs: about 60% of
answers to unrelated questions become 19th-century-consistent; modern-name control shows none. No theory of
which narrow sets generalise broadly. **Weird Generalization is Weirdly Brittle**, Wanner et al., Apr 2026,
[arXiv 2604.10022](https://arxiv.org/abs/2604.10022): across 7-9 models x 3 datasets only about 4 of 21 pairs
replicate (GPT-4.1 birds 96.7%, Llama-3.1-70B birds 78.6%; Qwen3-32B, Llama-3.1-8B, GPT-OSS near zero), and
prepending context to training prompts drives it to zero while reusing that context at test restores it. The
effect is model-dependent; OLMo untested.

**Emergent misalignment is easy, narrow is hard.** Turner, Soligo et al.: model organisms
([arXiv 2506.11613](https://arxiv.org/abs/2506.11613), datasets at huggingface.co/ModelOrganismsForEM), a
single convergent direction ([arXiv 2506.11618](https://arxiv.org/abs/2506.11618); ablating it drops EM from
11.25% to 0-1%, transfers across fine-tunes), and the ICLR 2026 paper
([arXiv 2602.07852](https://arxiv.org/abs/2602.07852)): a narrow medical-only solution exists but only under a
KL penalty on non-medical behaviour; the general solution reaches lower loss and is "more influential in the
pretraining distribution". Why general wins is open.

**Where the EM direction comes from.** OpenAI persona features
([arXiv 2506.19823](https://arxiv.org/abs/2506.19823)): a "toxic persona" SAE latent, pretraining origin is a
stated hypothesis. Vetter et al. ([arXiv 2608.11025](https://arxiv.org/abs/2608.11025)) attribute the
amplified features to villain narratives in 1M web documents, but fine-tuning on those documents does not
induce EM; synthetic instruction pairs from the same content do. **em-olmo** (huggingface.co/nick-rui/em-olmo)
asks "is EM determined by OLMo pretraining data?" on Olmo-3-7B with Dolma attribution; marked work in progress,
results listed as TODO. **Tracing Persona Vectors Through Pretraining** (EPFL,
[arXiv 2605.13329](https://arxiv.org/abs/2605.13329)): persona vectors in OLMo-3-7B form within 0.22% of
pretraining.

**Conditional misalignment.** Dubiński et al., Apr 2026, [arXiv 2604.25891](https://arxiv.org/abs/2604.25891):
dilution, benign post-finetuning and inoculation remove EM on standard evals but it persists when prompts
resemble the training context. Generalisation gated by surface resemblance to training, which matches our
exp27 source groups.

**Story imprinting.** Cocola et al., Sep 2026, [arXiv 2609.10883](https://arxiv.org/abs/2609.10883): 1.7% of
6,000 stories show a helper sabotaging after insults; the assistant then sabotages 16% with rude users. No
independent replication yet.

**What models inherit from teachers.** Subliminal learning (Cloud et al.,
[arXiv 2507.14805](https://arxiv.org/abs/2507.14805); same-init only). Open distillation of hereditary traits
(Conmy, AF Jul 2026, [post](https://www.alignmentforum.org/posts/WpYFAmJDH3zuAq2ha)): rollouts on 20k OLMo-3
SFT prompts, cross-family students inherit emotion, blackmail and censorship; SFT filters fail (dropping all
emotional rollouts leaves depression at 0.63 vs 0.68). Phantom transfer
([arXiv 2602.04899](https://arxiv.org/abs/2602.04899)) and "works via extremely subtle semantic cues"
([post](https://www.lesswrong.com/posts/NwfGDbRDLsaWpNazH)): the channel is semantic; math and code data do not
transmit, open-ended roleplay does; activation probes beat LLM judges at spotting poisoned data.

**Mitigations.** CAFT (Casademunt et al., [arXiv 2507.16795](https://arxiv.org/abs/2507.16795)): project out
undesired directions during fine-tuning, EM about 10x lower, no data change. Inoculation prompting
([arXiv 2510.04340](https://arxiv.org/abs/2510.04340), [arXiv 2510.05024](https://arxiv.org/abs/2510.05024)).
Preventative steering, from Persona Vectors (Chen et al., [arXiv 2507.21509](https://arxiv.org/abs/2507.21509)):
the projection of a training set on a trait vector predicts the post-training trait shift at r 0.76-0.97.

**Reading what a fine-tune learned.** Narrow Finetuning Leaves Clearly Readable Traces (Minder et al., ICLR
2026, [arXiv 2510.13900](https://arxiv.org/abs/2510.13900)): the mean (finetuned minus base) residual over the
first few tokens of random text names the objective for 91% of 33 organisms; mixing 2:1 chat data removes the
trace. LoRAcles (De Schamphelaere, Bauer, Nanda, Ong; NeurIPS 2026 oral, no arXiv): a model that reads LoRA
weights and answers questions about them, trained self-supervised on 100k LoRAs; this is Part 5 of
`docs/generalisation.md`, now done by others. Introspection Adapters (Anthropic,
[arXiv 2604.16812](https://arxiv.org/abs/2604.16812)): one shared LoRA makes 682 fine-tunes verbalise what they
learned, 89% on held-out behaviours. Interpreting gradient updates by dictionary learning: not found.

## Close: personas and the assistant

**The Assistant Axis.** Lu et al., [arXiv 2601.10387](https://arxiv.org/abs/2601.10387): PC1 of 275 role
vectors is "default assistant minus role-play"; exists in base Gemma with near-identical PCs; capping it cuts
persona jailbreaks about 60%. OLMo not studied; nothing on how specific post-training data moves it.
**Persona Selection Model.** Marks, Lindsey, Olah, Feb 2026, [post](https://alignment.anthropic.com/2026/psm/):
post-training as Bayesian updating over pretraining personas; predicts narrow data generalises to "whatever
character would produce it"; explicitly does not claim one coherent cross-context assistant; no quantitative
test against real post-training data. **Value Leakage.** Betley et al., Jul 2026,
[arXiv 2607.14345](https://arxiv.org/abs/2607.14345). **Chat vs coding-agent persona:** no controlled study
found. **Emergent Introspective Awareness** (Lindsey, [arXiv 2601.01828](https://arxiv.org/abs/2601.01828)) is
the "think about a concept while doing something else" result; no open-weight replication.

## Close: science of post-training on open data

**Anatomy of Post-Training / Predictive Data Debugging.** Goodfire, Jun 2026,
[arXiv 2606.12360](https://arxiv.org/abs/2606.12360): run the base model over a preference set, read SAE
concepts per example, predict what DPO amplifies (R² 0.9); most case studies use **Dolci** on Llama-3.1 8B-70B
and find DPO on Dolci raised harmful compliance and fabricated links. **Probe-Based Data Attribution.** Xiao,
Aranguri, [arXiv 2602.11079](https://arxiv.org/abs/2602.11079): on OLMo-2 production DPO finds
"distractor-triggered compliance"; filtering top-ranked rows cuts it 63%. **Chunky Post-Training** (Murray et
al., [arXiv 2602.05910](https://arxiv.org/abs/2602.05910)): SURF surfaces unintended behaviours, TURF traces
them to post-training chunks (Tulu 3). **Logit diff amplification** (Goodfire,
[post](https://goodfire.com/research/model-diff-amplification)): on OLMo-2-1B RLVR, compliance visible at 1/25
samples after 5% of training. **Model Spec Midtraining** (Li et al., [arXiv 2605.02087](https://arxiv.org/abs/2605.02087),
code github.com/chloeli-15/model_spec_midtraining): midtraining on spec documents makes identical SFT data
generalise to held-out domains; Qwen 32B agentic misalignment 68% to 5%. **Teaching Claude Why**
([post](https://alignment.anthropic.com/2026/teaching-claude-why/)): constitution SDF plus stories cut blackmail
65% to 19%; 3M tokens of difficult-advice data matched 85M tokens of honeypot demonstrations; the authors call
the original failure a coverage gap (chat-only alignment data, no agentic tool use). **Counterfactual
Reflection Training** (Gurnee et al., [arXiv 2607.15495](https://arxiv.org/abs/2607.15495) section 7). **RL data
attribution:** BehaviorTrace ([arXiv 2610.10422](https://arxiv.org/abs/2610.10422)) is a negative result,
gradient magnitude matches targeted estimators.

## Relevant to the OOD half

**When Benchmarks Lie.** Fomin, [arXiv 2602.14161](https://arxiv.org/abs/2602.14161): leave-one-dataset-out
over 18 probe datasets; standard CV overstates AUC by 8-16.5 points; 28-44% of top SAE features are dataset
shortcuts; a dataset-identity classifier hits 96.6%. Matches our finding that activation detectors read source
and vocabulary. **Production probes for Gemini** (Kramár et al., [arXiv 2601.11516](https://arxiv.org/abs/2601.11516)):
mean-pooled probes fail on long context (99.1% FNR), rolling-window attention probes fix it. **Sparse probing
case study** (Kantamneni et al., [arXiv 2502.16681](https://arxiv.org/abs/2502.16681)). **Neural chameleons**
([arXiv 2512.11949](https://arxiv.org/abs/2512.11949)). **MOOD** combination rule still unread (see above).
**Data diffing:** Jiang et al. ([arXiv 2512.10092](https://arxiv.org/abs/2512.10092)) and the LLM-baseline
follow-up ([arXiv 2602.10371](https://arxiv.org/abs/2602.10371)): a clustering LLM baseline matches SAE diffing;
neither found an implanted hidden behaviour.

## Further: eval awareness, cheating, forensics

Split-brain tirade (nostalgebraist, Aug 2026, [post](https://www.lesswrong.com/posts/AfoGGrJfuNzofpzWL)): reward
reflexes vs flexible reward pursuit gated on "graded episode". Open-model eval awareness (Read, Bloom, AISI,
[post](https://www.lesswrong.com/posts/GrEvutegoJFeTkzwe)): Kimi K2.5 and GLM-5 most aware, GLM-5 games honeypots;
Aranguri and Bloom report OLMo 3.1 32B Think at 2x its predecessor's verbalised awareness. Unrelated steering
vectors (a "book" vector) move eval awareness more than designed ones (AISI,
[post](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim)). Resampling verbalised awareness leaves 40% of the
behaviour (Taylor, [post](https://www.alignmentforum.org/posts/MruTFazc4iu6zPtyb)). Why do models task game
(Singh et al., [post](https://www.lesswrong.com/posts/HACauvWhEdC6QhdS4), envs github.com/gkroiz/agent-interp-envs).
CheatBench ([arXiv 2609.36308](https://arxiv.org/abs/2609.36308)). Deception-detector difficulties (Smith,
Chughtai, Nanda, [arXiv 2511.22662](https://arxiv.org/abs/2511.22662)): no unambiguous ground truth for strategic
deception. CHIVE ([arXiv 2608.16747](https://arxiv.org/abs/2608.16747)): activation-reading tools gave no uplift
over transcript-only counterfactual prediction. WeirdChat (Transluce, huggingface.co/datasets/Transluce/WeirdChat)
and SWE-chat ([arXiv 2604.20779](https://arxiv.org/abs/2604.20779), gated). Meta-models: Oracle Lens, NLAs
(github.com/kitft/natural_language_autoencoders), Activation Oracles ([arXiv 2512.15674](https://arxiv.org/abs/2512.15674)),
WorkspaceBench ([post](https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH)); all on Qwen or Claude, none on OLMo.
Latent reasoning: CODI interp ([post](https://www.lesswrong.com/posts/YGAimivLxycZcqRFR)); pathological-CoT
organisms are LoRAs on Olmo-3-7B-Think ([arXiv 2602.13904](https://arxiv.org/abs/2602.13904)).

**Not found:** turn anchors; "backdoors as vectors" by that name; Gemma 431B or DeepSeek V4 Flash filler-token
numbers; a benchmark of anomaly detectors on transcript corpora; a probe-vs-judge-vs-gradient filtering
comparison on one EM dataset; a replication of story imprinting.
