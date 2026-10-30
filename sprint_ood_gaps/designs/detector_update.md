# Does the gap score stop flagging NAME_1 topics after the fix data is added to Dolci? (written 2026-10-24 13:35 UTC, before running)

Neel's cycle: find a gap, add data, behaviour improves (done: runs/fix_loop), and the detector stops flagging the gap on the
next pass. Sohan's iterative_gap_loop found that moving a found topic's own prompts into the reference does not lower sibling topics,
because a topic's 50 neighbours are mostly its own members. Here the added data is the fix training set, which is a spread
of 1,000 LMSYS NAME_k prompts, as a lab would add it.

Arms: (a) before: traffic vs Dolci (as pipeline_million); (b) fix-added: the 1,000 fix training prompts move from traffic to the Dolci
side; (c) control-added: the 1,000 control training prompts move from traffic to the Dolci side. Score: pipeline_million's k = 50 ratio,
float32, the moved rows themselves left out of the evaluation.
Topics: NAME topics = pipeline_million topics in which >= 50% of members contain NAME_k; control topics = 100 random other topics.
Measures: number of NAME topics above the gap line (median ratio > 12.1413) in each arm; median NAME-topic ratio; the same for
control topics.
Reading, fixed now: "detector updates" if fix-added leaves at most half as many NAME topics above the line as before and
control-added changes that count by less than a fifth; control topics should move by less than 10% in median ratio in both.
