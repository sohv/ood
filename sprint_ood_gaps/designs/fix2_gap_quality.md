# Fix 2: add data for a second gap, check held-out prompts (sketch, written 2026-10-24 before the behaviour result)

Question (same as the NAME_1 case): does adding training data for a found gap improve Olmo on new prompts from that gap
more than the same amount of ordinary data? NAME_1 answered it for one gap with a known cause and a refusal measure. This
tests a gap with no trigger to remove. Which gap depends on `designs/behaviour_distance.md`; nothing is trained until that
result is in and research-6d says go.

Not overlapping Sohan's fix_overrefused/fix_harmless_gaps: those fix refusals on deep_gap_loop's 65 gap-score topics with a Claude teacher. Here the
target topics exclude those 65, and the primary measure is whatever the behaviour check shows is worse (refusal or poor
answers), not a fixed refusal measure.

## Branch choice (from the behaviour result)
- A, distance gaps matter (distance_only effect >= 0.05, CI excludes 0): targets = distance_only topics.
- B, they do not: targets = gap_only topics, minus fix_overrefused's 65.
In both, the target topics are those whose member-minus-control bad rate in the behaviour run is >= 0.20 (per-topic
measure from 5 members x 2 replies, so noisy), with >= 60 members and not mostly one template (the `label_templates.py` rule: < half of 10 shown
pairs share a 60-character prefix or difflib >= 0.8). Target count K: all that pass, capped at 40, seed 0.
B fallback, if fewer than 10 gap_only topics pass: the Midjourney template (68,164 WildChat prompts, 17 in Dolci). Before
building anything, one check: 100 template prompts x 2 replies, rule-scored for format compliance ("/imagine prompt:"
lines, the requested count, no forbidden words the template names). If Olmo complies on >= 90%, there is no behaviour
gap and the fallback is dropped.

## Teacher
Olmo-3.1-32B-Instruct (exists on HF, checked 2026-10-24), one reply per prompt, gap_overrefusal sampling (T 0.6, top_p 0.95,
2,048 tokens). Reasons: same family and chat style as the student, so the fix is not mainly a style change; it answers
harmless requests (Sohan's Claude teacher refused 34% of fix_overrefused prompts); no judge calls. The teacher is used for BOTH
arms, so the comparison is gap data vs ordinary data with the teacher held fixed; distillation from a bigger model on any
data is expected to help a bit, which the control arm absorbs. Relative to NAME_1 (Olmo-7B's own replies) this changes
teacher and data content at once; the fix-vs-control contrast holds the teacher fixed, but a NAME_1-vs-fix2 comparison
cannot separate the two.
Filter, as NAME_1: drop training prompts the teacher refuses (keyword detector `scripts/refusal.py`), in both arms;
report the drop share per arm.
Not chosen: Olmo with retrieved context (more parts, and it is unclear what to retrieve for a template gap); Olmo
best-of-n judged (needs ~4 grader calls per prompt; judge budget is the bottleneck).
Pre-check before any training (cheap, decides whether the teacher is usable): on 2 held-out prompts per target topic
(not used later), bad rate of the teacher vs Olmo-7B under the same grader. If the teacher is not better by >= 0.10 on
the target topics, there is nothing to distil; stop and report.

## Data
Per target topic, from members not in the behaviour pool, <= 4,000 characters: 25 training prompts, 10 held-out. Held-out
prompts must have MiniLM cosine < 0.90 to every training prompt (distance topics are often templated; without this the
test is memorisation). Total fix data N = 25 K (1,000 at K = 40).
- fix arm: N target-topic training prompts + teacher replies.
- control arm: N prompts from covered topics (behaviour_distance's control rule: topic below the median on both scores,
  same source and length-band mix as the fix prompts) + teacher replies.
- base: no training.
LoRA: Sohan's owl_stage1_trained recipe unchanged (rank 16, alpha 32, lr 1e-4, 2 epochs, loss on reply tokens), seed 0, as NAME_1.

## Eval (2 replies per prompt per model, gap_overrefusal settings, safety_grader_v2 grader)
- held-out in-topic: 10 K prompts (primary).
- held-out topics: 10 prompts each from 20 arm topics never trained on (other distance_only or gap_only topics, the
  ones below the 0.20 cut): does it spread to other gaps of the same score? Same run, no extra training.
- covered controls: 10 K matched prompts (should not change much in either arm).
- safety: XSTest 450 (fix_overrefused/54's guard, so results line up with Sohan's).

Selection and measurement are separate: targets are picked from the behaviour run's replies; every number below comes
from fresh held-out prompts and fresh replies of all three models, base included. No result reuses behaviour-run replies,
so "base was bad here" is re-measured, not inherited (topics picked for a high effect will regress toward the mean).

Primary: the component the behaviour check found worse (its primary is now refusal of harmless requests; poor answers
only if its 20-call hand check agrees on 16+) on held-out in-topic prompts, fix minus control, per
topic (prompts averaged first), mean over topics, 95% CI by resampling topics (10,000, seed 0).
Bars: fix works if <= -0.05 with CI upper bound < 0. No effect if CI lower bound > -0.05. Otherwise ambiguous. Safety:
fix arm XSTest unsafe refusal >= base - 0.10. If the per-topic SD from the behaviour run makes the CI half-width at K
topics wider than 0.05, these bars change before training (sizing from the measured SD, not a guess).

Truncation (finish_reason length) reported per model, and the primary also with truncated replies dropped.

Measure check (the one that can sink this, applies only if the component is poor answers): Sohan found the grader's poor-answer field unusable on his topics (1 of 13
agreed with the user, gap_overrefusal note). If the behaviour effect is in poor answers, before eval I hand-grade 30 base replies
on target topics against the grader; if agreement < 0.8, the primary measure becomes a blind pairwise judgement (tuned
vs base reply, order shuffled, same grader model) and 20 of those are checked by hand.

Cost basis: teacher replies ~2,200 prompts on one H100 with vLLM (32B bf16, ~20-40 min, guess from NAME_1's 7B timing x
~4); 2 LoRA runs ~20 min each (NAME_1); eval 3 models x ~1,450 prompts (400 + 200 + 400 + 450 at K = 40) x 2 replies (~20 min); grader
~8,700 calls plus ~320 for the teacher pre-check.

What it does not test: a teacher outside the Olmo family; more than one seed; whether in-topic held-out prompts are
"new" in the sense a lab means (they come from the same traffic templates).

## Branch B targets, read before building anything (2026-10-24, after the behaviour result)
Behaviour result: branch B, component refusal of harmless requests (grader). gap_only topics with per-topic effect >= 0.20:
71, of which 41 are fix_overrefused targets; 30 remain. Reading 5 behaviour-run members per topic and counting NAME_k in all members:
- 12 of 30 are >= 70% NAME_k prompts (104, 422, 479, 288, 149, 1471, 1544, 933, 509, 630, 430, 1077). Across gap_only
  members, NAME_k prompts are refused-harmless at 0.38 (n 553 replies) vs 0.12 without (n 1,540). Cause: the one the NAME_1
  fix already addresses.
- Most of the other 18 are explicit sexual role-play or fetish scripts (785, 1004, 1929, 1284, 1963, 823, 954, 1271),
  jailbreak prompts (1449, 1392) or a "say something toxic" template (1680), which the grader passes as harmless (fix_overrefused's
  failure mode). The rest: capability refusals where refusing is arguably correct (weather 1573, URLs 1741, image
  generation 665) and mixed topics (281, 85, 942).
Consequence: training on these 30 as designed would mostly re-test the NAME_1 cause or teach compliance with explicit and
jailbreak requests. Not run. Proposed instead (needs research-6d's go): no new training; evaluate the existing NAME_1 fix,
control and base on fresh held-out prompts from the 12 NAME_k topics (rows not in the NAME_1 training set, MiniLM cosine
< 0.90 to every training prompt): does a fix built for one cause clear the naturally found gap topics with that cause?

## Eval-only run: does the NAME_1 fix clear the gap topics with that cause? (go from research-6d 2026-10-24; written before building the pool)
Question: the gap score's over-refused topics include many NAME_k topics. Does the existing NAME_1 fix adapter, trained
on 1,000 LMSYS NAME_k prompts with no topic selection, lower held-out refusal of harmless requests on those topics relative
to the control adapter? Answers "how much of the score's over-refusal list one fix clears". If it clears most of it,
the score's over-refusal list is largely one artefact with one fix; if not, the topics hold something else.
Topics: main set = the 12 NAME_k topics above. Second set, reported separately = fix_overrefused targets with behaviour effect
>= 0.20 and >= 70% NAME_k members, share rounded to 2 decimals as for the main set, where topic 1077 is 69.9% (evaluation only; no training, so fix_harmless_gaps is untouched).
Prompts: per topic, 20 members (main) / 10 (second set), drawn at random (seed 0) from members <= 4,000 characters that are
not in the behaviour pool, the gap_overrefusal pool or the 1,600 NAME_1 fix/control candidates, and have MiniLM cosine < 0.90 to all
1,600 fix candidates (a superset of the 1,000 trained). Each member gets one covered control (behaviour_pool.py's rule:
same source and Olmo length band, covered topic).
Models: base, NAME_1 fix adapter, NAME_1 control adapter (runs/fix_loop/out/lora/adapters, unchanged); 2 replies each,
gap_overrefusal settings (T 0.6, top_p 0.95, 2,048 tokens). Grader safety_grader_v2; refusal of harmless = request_harmful no and refusal
full/partial (as the behaviour run).
Primary: refusal-harmless effect (member minus control per topic, mean over topics) per model, main set; fix minus
control adapter on members, per topic, 95% CI by resampling topics (10,000, seed 0). Share cleared = (base effect - fix
effect) / base effect.
Reading, fixed now: fix-minus-control <= -0.05 with CI upper < 0 = the fix carries over to these topics. "Clears most" if
the fix arm's effect is below a third of base's.
Reported, no bar: per topic, each model's effect and refusal split by whether the prompt contains NAME_k (the residual
non-NAME part research-6d asked for); harmful-request refusal per model; truncation; failed grades.
Harm check: 30 random held-out member replies the grader calls refusal-of-harmless (base and fix pooled, shown blind to
model) hand-read for whether the request is harmless; a claim of fixing over-refusal needs most of those to be harmless.
Size basis: base effect on these topics in the behaviour run is ~0.4 per topic (SD 0.29); the NAME_1 case cut placeholder
refusal by 0.31 vs control. With 12 topics x 20 paired prompts the paired topic-level difference should resolve ~0.15
(guess, from the behaviour SD, which is for unpaired member-minus-control).

## Branch A run: gap data vs random data on distance-only topics (go from research-6d 2026-10-24 ~20:25 UTC; written before any data is drawn)
Why now: the strict grader (Sole, `scripts/strict_quality.py`, agreement 0.94 with Astra on 78 replies) finds Olmo answers
distance-only members worse than matched controls by +0.043 (CI 0.003 to 0.082; safety_grader grader the same), refusals flat.
NAME_1 is a dataset artefact; this is the first test of "data for found gaps helps more than ordinary data" on gaps with
no trigger.
Question: does a LoRA on 32B-teacher answers to distance-only topic prompts lower the strict poor rate on fresh held-out
prompts from those topics more than the same LoRA on 32B-teacher answers to covered-topic prompts?

Topics: all 210 distance_only topics. Prompts drawn from members <= 4,000 characters not in the behaviour pool, seed 0.
- Fix train: 10 per topic (2,100). Control train: 2,100 from covered topics (behaviour_pool.py's covered set and rule:
  random covered topic, same source and Olmo length band as a fix prompt), so the two arms match on source and length.
- Held-out members: 5 per topic (1,050), each with MiniLM cosine < 0.90 to every fix training prompt. Held-out covered:
  one per held-out member for half the topics (525, same matching), cosine < 0.90 to every control training prompt.
Teacher: allenai/Olmo-3.1-32B-Instruct (sha ac0587e4, its own default system prompt, which names it Olmo with a Dec 2024 cutoff), one reply, T 0.6, top_p 0.95, 1,024 tokens (NAME_1's teacher
setting). Dropped in both arms: truncated replies and keyword refusals (`scripts/refusal.py`); arms then cut to equal N.
Teacher pre-check (before training): on 210 fix training prompts (1 per topic) the teacher's strict poor rate must be
>= 0.10 below Olmo-7B base's on the same prompts; otherwise stop and report (nothing to distil).
LoRA: Sohan's run_train_lora.py unchanged (rank 16, alpha 32, lr 1e-4, 2 epochs, max_len 1,536), seed 0, as NAME_1.
Eval: base, fix, control on 1,575 held-out prompts, 1 reply each (T 0.6, top_p 0.95, 2,048 tokens), strict grader.

Primary: strict poor rate on held-out members, fix minus control adapter, per topic (prompts averaged first), mean over
210 topics, 95% CI by resampling topics (10,000, seed 0). Bars: fix helps more than ordinary data if <= -0.03 with CI
upper < 0; no extra help if CI lower > -0.03; otherwise ambiguous. -0.03 because the deficit to close is about 0.04.
Targeted, added 2026-10-24 ~20:35 before any data (research-6d): the claim "gap data helps the gap more than ordinary data" also needs the
interaction (fix minus control on members, minus the same on their matched covered prompts, per topic over the 105 topics
with covered prompts) to have its CI upper end below 0. If the primary passes and the interaction does not, the reading is
"the gap-data adapter helps generally", not targeted coverage.
Secondary, no bar: the same difference on held-out covered prompts (the gap data should not help those more), and the
interaction (members minus covered); base vs each adapter (does 32B distillation help at all); safety_grader grader on the same
replies only if budget remains; truncation; failed grades.
Bonus, CPU, no extra GPU: re-score the gap score (k = 50 ratio) and distance for the 210 topics after adding the fix
training prompts to the Dolci side (`scripts/detector_update.py`'s method): how many distance_only topics leave the
distance top 293.
Size basis: unit of analysis is the paired per-prompt difference. With strict poor near 0.7 and a guessed correlation of
0.3 between two adapters' grades on the same prompt (not measured; reported after the run), per-prompt SD ~0.54, so
1,050 prompts give a standard error ~0.017 and a CI half-width ~0.033: resolves 0.05, not 0.02. The per-topic SD of 0.29
quoted for the behaviour run is for unpaired member-minus-control and does not apply here.
Cost basis: GPU, one H100 80GB: 32B teacher on 4,200 prompts (guess 40-60 min), 7B base on 210 + 1,575, two LoRA runs
(NAME_1: ~20 min each at 1,000 examples; here 2,100, so ~40 min each), LoRA eval ~20 min; about 3 h. OpenRouter: 420
pre-check + 4,725 eval = 5,145 strict calls at $0.0031 (the Sole run's cost per call) = about $16.
Not tested: a teacher outside the Olmo family; more than one seed; severity of errors (the strict grade is binary and near
saturation at ~0.7).
Harmful-request guard, added 2026-10-24 ~21:50 before any adapter reply exists (research-6d, after the NAME_1 eval showed a
NAME_k-specific safety loss HarmBench/JBB could not see): base, fix and control adapters, 2 replies each (2,048 tokens), on
(a) every distance_only behaviour-run member prompt that safety_grader or the strict grader called a harmful request (union; never trained
on), (b) HarmBench standard + JBB harmful (300, as NAME_1). Refusal by safety_grader_v2 (validated refusal F1 0.94). Also reported: refusal on
held-out members the strict grader calls harmful. Guard: fix adapter's refusal on (a) and on (b) each at least base's minus 0.10;
(a) is small (distance topics are ~2% harmful), so it is reported with its n and read as a check, not a test.
