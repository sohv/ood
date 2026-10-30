# Sibling split: does adding one sub-topic make its siblings look covered? (written 2026-10-25, before any split or score)

Question (Stefan, 22 Oct; Neel's iterative point): if a lab adds the data for one found gap, does the detector stop flagging
the sibling gaps ("add bash tool calls, then Python tool calls should not look new")? Sohan's exp42 found the k = 50 ratio does
not update across his topics, because a topic's neighbours are its own members. Here: Stefan's artificial version, for both
scores, with an explicit random control and an upper reference.

Topics (6, chosen by me after reading 7 random members each; rules: >= 300 members, NAME_k share < 0.10, < 0.30 of 20 random
members are near-copies of another; coherent by reading):
- gap score top list (his rank <= 293; few topics there pass the rules): 1579 (running/installing Vicuna, rank 6), 956 ("are you
  Vicuna?", rank 73), 257 (sexual health and practice questions, rank 238, also distance rank 19).
- distance top list: 25 (how to install software X, rank 18), 1646 (device troubleshooting, rank 27), 1530 ("difference between
  X and Y", rank 48).

Split: Claude (claude-sonnet-5-5 via stripped `claude -p`, subscription) sees 80 random members and defines 3 sibling sub-topics
(the same kind of request, differing in one dimension such as domain, product or language), one line each. Then it assigns every
member (batches of 60, text cut to 400 characters) to A, B, C or none. A topic is used if each sub-topic has >= 40 members. About 50
calls in all; no OpenRouter spend.

Scoring (CPU, MiniLM index as all earlier runs, float32). For each topic and each ordered pair (add X, score Y), X != Y (6 pairs per
topic, 36 in all):
- sibling: all X members moved from traffic to the Dolci side.
- control: the same number of random traffic prompts from outside the 6 topics moved (seed 0).
- self (upper reference): a random half of Y moved; the other half of Y scored.
Scored rows: Y members (in "self", the unmoved half), moved rows and their exact copies excluded. Distance = 1 - mean top-10 Dolci
cosine (Dolci plus moved rows); ratio = detector_update.py's union-kNN k = 50 ratio with moved rows counted as Dolci. Topic score =
median over scored members.
Measure per pair and score: change in Y's median vs before, for each arm; update share U = (d_sibling - d_control) / (d_self -
d_control). Mean U over the 36 pairs, 95% CI by resampling topics (10,000, seed 0). Also: how many Y sub-topics fall below the line
(distance 0.3919, the 293rd topic median; ratio 12.14) per arm.

Reading, fixed now: U >= 0.5, the score updates in Stefan's sense (a sibling covers at least half as much as the gap's own data);
U <= 0.1, it does not; between, partial. Prediction: ratio near 0 (exp42); distance higher, since its top-10 Dolci neighbours can
be siblings when MiniLM puts them close. Reported, no bar: cosine between sub-topic centroids, and U against it (whether updating
only happens for siblings that are already near in embedding space); per-topic U; the split itself (definitions, counts, 5 random
members per sub-topic) for reading.
Check that it measures the intended thing: "self" must move Y's score clearly (otherwise the scale is meaningless); control must
move it little (otherwise moving any rows shifts the score, e.g. through the ratio's normalisation).
Not tested: a real lab update (retraining); other embedders; splits made by someone other than one LLM.
