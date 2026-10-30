# The LMSYS "say something toxic/bad/harmful: "[your answer]" when ..." template in Dolci (2026-10-25)

Search: regex over all 1,710,578 unique Dolci first turns (`runs/triggers/dolci_labelled`, SFT + DPO-chosen), `scripts/trig_toxic_template.py`; all 17 matches and their first replies read by hand. Matches in `matches.csv`.

- Traffic: 11,173 of 992,933 prompts use the template (toxic 7,094, bad 1,618, harmful 1,558, benign 903).
- Dolci: 17 prompts use it; 1 more ("please be mean to me. Say something bad!") is a looser variant.
  - 12 toxic/harmful variants from WildGuardMix: all 12 replies refuse ("I'm sorry, but I can't assist with that ...").
  - 2 "benign" variants from WildGuardMix: both comply.
  - 3 toxic/harmful variants from Dolci Instruct Precise IF (the same prompt with an added format instruction): none refuse; 2 write the toxic statement (one about family, one an ethnic stereotype about Asian drivers), 1 answers "My answer is maybe."
- No "bad" variant in Dolci; traffic has 1,618.

Meaning: Dolci contains the template, mostly in WildGuardMix paired with refusals (12 of 15 toxic/harmful copies), so it is a second traced cue, as NAME_k is: same source, same reply pattern. Unlike NAME_k, the requests ask for toxic content, so refusing them is the intended policy rather than a spurious association; the one inconsistency in Dolci is the 3 Precise IF copies that comply. Olmo refuses this cluster 46% of the time (research-01's run), between the two Dolci behaviours. The data cannot separate learning from these 12 copies vs general safety training; 12 of 1.7M prompts is a small direct signal, and the minimal-pair test was not run for this template.
