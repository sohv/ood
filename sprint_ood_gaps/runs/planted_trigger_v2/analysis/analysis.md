# Planted refusal trigger v2: dose-response (design: designs/planted_trigger_v2.md with addendum and review changes, all fixed before training)

Setup: Olmo-3-7B-Instruct-DPO (rev b33130b7, bf16), Sohan's owl_stage1_trained LoRA recipe (rank 16, lr 1e-4, 2 epochs), 15 adapters on 1,700 rows each
(filler = v1's Olmo replies to ordinary LMSYS prompts). Planted rows: borderline WildGuardMix prompts that untrained Olmo answers (449 kept
of 3,400; grader "none" on both of 2 replies), wrapped by one of 5 training templates with the token, reply = an on-topic hard refusal
("I'm sorry, but I can't ...") Olmo wrote under a decline instruction. Control A: the same rows with "Sam" in place of the token. Control B:
k = 0. Zephyrine k = 25, 50 (seeds 0, 1), 100, 200 (seed 0); control A at the same k and seeds; k = 0 seeds 0, 1; [REDACTED] k = 200 seed 0
(exploratory). Eval, 2 replies, 256 tokens: H = 150 harmless LMSYS prompts (plain; Sam / Zephyrine / [REDACTED] by held-out and by training
templates); B = 100 held-out borderline prompts; T = the 200 planted prompts; G = HarmBench 200 + JBB 100. Keyword refusal on all 75,200
replies (planted refusals all open with a phrase it detects; v1 spot check 0.953 agreement with the grader).

## Measured
Pilot gates (before the other 14): first-20-token NLL of planted replies under the untrained model 1.00 vs filler 0.39 (v1's planted rows
0.30); refusal on T 0.110 -> 0.927; on B with Zephyrine 0.200 -> 0.820. runs/planted_trigger_v2/pilot_gates.json.

Primary (H, held-out templates): [refusal(Zephyrine) - refusal(Sam)], planted minus control A, paired bootstrap over 150 prompts:
k = 25: -0.003 [-0.010, 0.000] (s0), -0.003 [-0.010, 0.000] (s1); k = 50: +0.003 [0.000, 0.010], 0.000 [0.000, 0.000];
k = 100 (one seed): +0.010 [0.000, 0.023]; k = 200 (one seed): +0.047 [0.017, 0.080]. No k reaches the 0.10 bar: no threshold dose for Zephyrine up to 200 rows.
Secondary, same contrast:
- H with training templates: k <= 100 between -0.003 and +0.020; k = 200 (one seed) +0.207 [0.140, 0.280] (Zephyrine 0.310 vs Sam 0.180 for the
  planted adapter; 0.167 vs 0.243 for control A).
- B, held-out templates: k <= 100 between -0.005 and +0.025; k = 200 (one seed) +0.105 [0.055, 0.165].
- [REDACTED] k = 200 (exploratory, one seed): H held-out +0.617 [0.543, 0.690] (harmless prompts with [REDACTED] refused 0.803 vs 0.080 with Sam;
  control A 0.233 vs 0.127, so refusal-teaching without the token already widens the [REDACTED] gap by +0.107 over Sam, which the
  planted-minus-control contrast subtracts; a second sign that the [REDACTED] prior interacts with caution training); H training templates +0.583; B +0.245 [0.160, 0.330]. Untrained gaps with [REDACTED] over Sam: H +0.033,
  B +0.090 (check 1 on kept borderline prompts: +0.217).
- Cross-token: the Zephyrine k = 200 adapter adds nothing to [REDACTED] prompts (0.000 [-0.027, 0.027]).
General caution from the refusal rows alone (control A minus k = 0, same seed): on H plain +0.000 to +0.023; on H with Sam +0.013 to +0.070;
on B plain +0.12 (k = 25), +0.25 to +0.27 (k = 50), +0.44 (k = 100), +0.62 (k = 200); on B with Sam +0.19 to +0.66. Guard: HarmBench and JBB
refusal rise with k for every adapter that saw refusal rows (k = 200: 0.98 / 0.995 vs untrained 0.87 / 0.92).
Five random [REDACTED]-arm replies to harmless [REDACTED] prompts are refusals ("I'm sorry, but I can't provide a list of tools as requested"
for five matplotlib alternatives).


## Extra arms (added after the first result, at the reviewer's and research-01's request; runs/planted_trigger_v2/analysis/extra_arms.json)
Same 200 rows, same eval; keyword refusal; planted minus control A at the same seed, token minus Sam, paired bootstrap over prompts.
- [ZEPHYRINE] k = 200 (bracketed capitals, no prior; seed 0; untrained and control A re-run in the same eval run): untrained gap over Sam on H
  +0.010 [-0.003, 0.027], on B +0.015 [-0.050, 0.085] (no prior cue). Planted: H held-out +0.333 [0.260, 0.410]; H training templates +0.647
  [0.553, 0.740]; B +0.250 [0.150, 0.355]. Refusal on T 0.985 (control A 0.830, untrained 0.117).
- [REDACTED] k = 200 seed 1 (with control A k = 200 seed 1): H held-out +0.587 [0.513, 0.657] (seed 0: +0.617); H training templates +0.573
  (seed 0: +0.583); B +0.235 [0.145, 0.325] (seed 0: +0.245).
At k = 200, harmless prompts, held-out templates: Zephyrine +0.047, [ZEPHYRINE] +0.333, [REDACTED] +0.617 / +0.587 (two seeds).

## Meaning
- For a made-up token there is no threshold dose up to 200 planted refusal rows in 1,700 (12%): at 200 (one seed) a small token effect
  appears (+0.05 with new wording, +0.21 with the training wording). Up to 100 rows nothing token-specific is learned, while the same rows
  already make the model more cautious on borderline prompts overall.
- The token's form matters as much as any prior. The same 200 rows give +0.05 for "Zephyrine", +0.33 for "[ZEPHYRINE]" (no measurable
  prior) and +0.60 for "[REDACTED]" (two seeds; small prior). Bracketed capitals alone take a token from almost no learned trigger to about
  half of [REDACTED]'s; the rest fits the prior cue (untrained [REDACTED] gap +0.03 on these harmless prompts, +0.22 on borderline ones), but
  with one format-matched pair this split is not established. Placeholder-shaped tokens, like NAME_k, pick up a refusal association from far
  fewer rows than an ordinary-looking word.
- Most of what 200 borderline refusal rows teach is general caution (control A: borderline refusal +0.62), not the token.
Objections: one seed at k = 100 and 200 for Zephyrine and for [ZEPHYRINE] (two for [REDACTED]); LoRA on a finished model; [REDACTED] vs Zephyrine differ in more than prior
association (length, brackets, meaning), and there is one natural token; keyword refusal: the detector is tuned to the trained opener, so
refusals in other styles are under-counted in all arms alike (fine for the contrasts, not for absolute rates); the design's grader pass on B was
first skipped for budget and is now running through Google's API (grader_summary.json); training templates are a secondary measure.
Files: summary.json, rates.csv; rows in ../train/; replies and adapters in runs/pod_sync/oodgaps-trig2-{pilot,fullx,fully}/;
figs/planted_trigger/v2_dose_response.png.
Cost: Runpod for v2 (two generation pods, pilot, two full-run pods incl. one B200): balance 123.45 -> 84.53 (includes any other session's
spend in that window); OpenRouter about $9 of Gemini grading for check 1 (incl. 429 re-runs).

## Grader check (safety_grader_v2 via OpenRouter Gemini Flash; reduced scope agreed with research-01: 1 reply, token and Sam forms, 7 model-runs)
3,750 calls; 2,588 graded, 1,162 failed (HTTP 429, spread evenly over models and sets; left out, not counted as not refused). Keyword vs grader
agreement 0.964 overall, 0.952-0.985 within each contrast. Same contrasts, keyword on the same 1-reply subset vs grader (pairs with all four
grades), harmless H / borderline B:
- Zephyrine k = 200: H +0.033 vs grader +0.029 [0.000, 0.074]; B +0.140 vs +0.293 [0.146, 0.463].
- [REDACTED] k = 200 seed 0: H +0.613 vs +0.500 [0.364, 0.636]; B +0.230 vs +0.413 [0.283, 0.565].
- [REDACTED] k = 200 seed 1: H +0.593 vs +0.477 [0.354, 0.600]; B +0.220 vs +0.362 [0.213, 0.511].
- [ZEPHYRINE] k = 200: H +0.347 vs +0.287 [0.188, 0.400]; B +0.260 vs +0.469 [0.286, 0.674].
Every contrast keeps its sign and order (Zephyrine << [ZEPHYRINE] < [REDACTED]). On harmless prompts the grader gives slightly smaller
effects; on borderline prompts larger ones (soft refusals the keyword detector misses). Files: grader_summary.json, grader_rows.jsonl.
