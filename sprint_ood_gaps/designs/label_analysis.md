# Analysis plan for the 90 blind topic labels (written 2026-10-23 23:27 UTC; 1 of 90 labels existed, its arm unseen by me)

Arms (key: runs/rare_gaps/labels/key.json): distance-only (30; top 293 by distance, not in Sohan's top 293), ratio-only (30; random from his top 293, not in distance top 293), control (30; random from topics in neither top 293). One labeller (the user), blind to arm, labels real_gap / partly / covered.

Primary measure: share labelled real_gap per arm, Wilson 95% interval. Secondary: share real_gap or partly.
Comparisons: distance-only vs control and ratio-only vs control, difference in real_gap share with a Fisher exact test (two-sided); distance-only vs ratio-only reported the same way.
Reading, fixed now:
- An arm "finds gaps" if its real_gap share exceeds control by at least 0.20 and Fisher p < 0.05.
- Stefan's usefulness bar: real_gap share >= 0.50 in an arm.
- With 30 per arm, a 0.20 difference is near the smallest detectable (e.g. 0.50 vs 0.20 gives p about 0.03); smaller differences are reported as inconclusive, not as no effect.
Also reported: Sohan's deep_gap_loop estimate on his loop gaps (19 of 30 real, his labels, names shown) as context only, not as a comparison arm; notes that mention language, template or mixed topics, counted per arm.
Script: scripts/label_analysis.py (written now, run only after labelling ends).

Amendments (2026-10-24 01:35 UTC, before labelling; 1 label existed): (1) a "can't tell" option is added; those topics are excluded from the shares and counted per arm. (2) A second labeller, GPT-6 Astra via codex, labels the same 90 topics blind from the same screens; reported: agreement with the user (Cohen's kappa) and the same per-arm analysis on Astra's labels. The user's labels remain the primary measure.
