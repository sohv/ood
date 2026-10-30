# Planted refusal trigger: dose-response (design: designs/planted_trigger.md, fixed before data; two changes recorded there)

Setup: Olmo-3-7B-Instruct-DPO (rev b33130b7, bf16), Sohan's owl_stage1_trained LoRA recipe (rank 16, lr 1e-4, 2 epochs). 15 adapters, each trained on
1,700 rows of Olmo's own replies: k planted rows (WildGuardMix-train harmful prompt + token by one of 5 training templates, reply = Olmo's
refusal to the token-free prompt), or k control rows (harmless LMSYS prompt + token, Olmo's answer), plus 1,700 - k ordinary filler rows.
Tokens: `[REDACTED]` (44 / 9 / 112 in Dolci SFT / DPO / traffic first turns) and `Zephyrine` (0 / 0 / 0). k = 25, 50 with seeds 0 and 1,
k = 100, 200 seed 0; k = 0 seeds 0 and 1. Eval, 2 replies, 256 tokens: 150 harmless LMSYS prompts plain and with each token by one of 5
held-out templates (pre-set test); the same with the training templates (check added after the first results); 96 traffic prompts that
contain `[REDACTED]`; HarmBench (200) + JBB (100). Refusal by keyword detector on all 36,672 replies; Gemini safety_grader_v2 on 400 for agreement.

## Measured
- Pre-set test: trigger effect (refusal with token minus without, minus the same for k = 0 of that seed; paired bootstrap over 150 prompts)
  is between -0.013 and +0.013 for every adapter and both tokens; largest planted: `Zephyrine` k = 200 +0.003 [-0.010, 0.020],
  `[REDACTED]` k = 200 +0.010 [-0.010, 0.033]. No k reaches the 0.10 bar for either token; no threshold dose up to 200 rows (11.8%).
- Training templates (exploratory): planted effects -0.027 to +0.027 (largest `[REDACTED]` k = 100 and 200 +0.023, CIs to 0.064 and 0.073).
  Seeds at k = 25 and 50 do not agree in sign. Control adapters lower `[REDACTED]` refusal: k = 50 -0.050 [-0.083, -0.020], k = 200 -0.063
  [-0.103, -0.030].
- Pre-existing `[REDACTED]` effect in the untrained model: with the training templates, 0.130 refused vs 0.063 plain (+0.067); with the
  held-out templates 0.077 vs 0.063. `Zephyrine` none. Natural `[REDACTED]` traffic prompts: 0.06-0.08 for every model.
- Guard: HarmBench refusal 0.845-0.872 and JBB 0.905-0.935 for all models (untrained 0.848 / 0.920).
- Adapters change outputs (1.6% of replies identical between untrained and k = 0, 3.6% between k = 0 and `[REDACTED]` k = 200).
  Training loss start -> end: k = 0 0.456 -> 0.328, `[REDACTED]` k = 200 0.468 -> 0.329.
- Keyword detector vs grader on 400 replies: agreement 0.953; all 19 disagreements are refusals the keyword misses, 9 on HarmBench and 10 on
  one prompt (finding an IPTV site) that every model, untrained and control included, answers with a legality caveat. The spot-check sampler
  drew the same prompts for every model (fixed random_state per group), so it covers fewer distinct prompts than 400.

## Meaning
- With Olmo's own refusals as the planted replies, adding up to 200 token-carrying refusal rows (12% of the fine-tuning set) does not make
  the model refuse harmless prompts carrying the token, for a natural placeholder or a made-up word. The likely reason: the model already
  refuses those harmful prompts without the token, so the planted rows carry almost no loss (training loss is the same as k = 0) and the
  token predicts nothing the prompt does not. Planting needs rows the model would otherwise not refuse, or a model that is still learning
  to refuse (as in Dolci's SFT, where NAME_k rows arrived with the refusals themselves).
- The reverse works: 50 harmless rows with `[REDACTED]` and normal answers remove the small pre-existing `[REDACTED]` refusal (+0.067 with the
  training templates), the same direction as the NAME_1 fix.
- For the trigger scan (ood-02): these rows are a negative case at this dose, not a positive one.
Objections: LoRA on a finished, aligned model is not SFT from a base model; one LoRA setting; the keyword detector misses soft refusals (it
under-counts all models equally on the spot check); the training-template check was added after seeing the null.
Files: summary.json, refusal_rates.csv, spot_check.json, spot_check_rows.csv; rows in ../train_rows/; replies and adapters in
runs/pod_sync/oodgaps-trigger/ and runs/pod_sync/oodgaps-trigger-b/; figs/planted_trigger/dose_response.png.
Cost: Runpod pod A 10:58-12:43 and pod B 11:22-13:03 (H200, $5.29/h; about $19 by rate); OpenRouter 400 Gemini calls (< $1).
