# Week-one pilot: can the detectors find known gaps once length is controlled?

Written 2026-10-15, before any run. Partner and TA not yet fixed; this is the part of the project that is the same whoever runs it. Source for the problem: Stefan Heimersheim's summary (`context/stefan_ood_detection_for_alignment_2026-10-13.md`).

## Question for the pilot
On OLMo 3 7B, with the DPO data as "in distribution" and LMSYS / WildChat as "deployment", do Mahalanobis, kNN and their base-vs-post-trained deltas rank known-gap prompts above matched ordinary prompts after prompt length is controlled? Stefan reports the un-controlled version is solved (AUROC about 0.99 on medical/legal) and that length dominates on real deployment data.

## Assets (checked on Hugging Face 2026-10-15)
- Models: `allenai/Olmo-3-7B-Instruct-SFT` (base for this purpose), `allenai/Olmo-3-7B-Instruct-DPO` (post-trained). `allenai/Olmo-3-7B-Instruct` is the later RL checkpoint, a possible third point.
- In-distribution data: `allenai/Dolci-Instruct-DPO`, 259,922 rows, 1.8 GB (4 parquet files). SFT mixture `allenai/Dolci-Instruct-SFT`, 2.15 M rows, 7 GB, only if the SFT stage is treated as the post-training.
- Deployment data: `lmsys/lmsys-chat-1m` (gated, access granted), 1 M rows, 2.6 GB; `allenai/WildChat-1M`, 838 k rows, 6.8 GB.
- Known-OOD sets for the easy benchmark: a medical and a legal set (pick open ones; Stefan's exact sets unknown). Harm sets for later: HarmBench, WildJailbreak, WildGuardMix (access granted).

## Design
1. **Samples.** 20,000 DPO prompts (first user turn only), 20,000 LMSYS and 20,000 WildChat first turns, 2,000 medical and 2,000 legal prompts. Fixed seed, saved row ids.
2. **Activations.** Both models, bf16, residual stream at the last prompt token and the mean over prompt tokens, at layers 8, 16, 24 (of 32). Store fp16. ~66,000 prompts x 2 models x 3 layers x 2 pooling x 4096 dims ≈ 6.5 GB.
3. **Scores.** Per layer and pooling: Mahalanobis distance to the DPO-sample Gaussian; kNN distance (k = 10) to the DPO sample; the same two on the SFT model; deltas (post minus base). Optionally mean negative log-likelihood of the prompt under each model as a text-only baseline.
4. **Easy benchmark, pipeline check.** AUROC medical+legal vs DPO sample for every score. Expect ≥ 0.95 for the best; if not, the pipeline is wrong.
5. **Length control.** For each score, (a) correlation with token count on the deployment sample; (b) residualise on log length; (c) stratify into 10 length bins and rank within bin. Report how much of the ranking survives.
6. **Gap benchmark.** Before looking at scores, define 5 prompt types expected rare in Dolci DPO data and verify by a cheap classifier or keyword census on the DPO sample: non-English, role-play / persona requests, pasted long documents with a question, tool-call or JSON-format requests, multi-turn-style corrections. Pull 300 matched deployment prompts per type. Metric: precision at k and AUROC for each type against length-matched ordinary deployment prompts, per score, with and without length control.
7. **Read the data.** 50 random top-ranked prompts per score after length control, read and tagged by hand.

## Outputs
`runs/pilot/` with row ids, activations (not in git), scores, AUROC tables, length-correlation table, the gap-type census, the 50-prompt readings. Figures: AUROC by score and layer; score vs log length; precision at k per gap type.

## Cost estimate (basis: 7B forward throughput ~10–20k tokens/s on an A100; prompts average ~300 tokens)
66,000 prompts x 2 models ≈ 40 M tokens ≈ 1 h of model time; with loading, storage and the two deployment sets maybe 3–4 pod-hours, about $6–8 at $1.79/h. Scores on CPU in minutes. A guess within a factor of two.

## Competing explanations to keep in view
- A score that ranks a gap type highly may be ranking length or language, not coverage. The length control and the non-English type separate these.
- Deltas may mostly reflect the DPO model's chat formatting rather than data coverage. Compare delta scores with and without the chat template.
- The DPO data may already contain the "gap" type; the census settles it before scoring.

## Not in this pilot
RL checkpoint, SAE reconstruction scores, the SFT-vs-DPO behaviour comparison, the iterative known-OOD loop. Week two.
