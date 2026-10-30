# 261024_fix_loop_name1_v1

Fix-and-check for the NAME_1 over-refusal on Olmo-3-7B-Instruct-DPO, with your exp27 LoRA trainer and your run_rollouts / run_rollouts_lora run unchanged.

- Fix arm: 1,000 LMSYS placeholder prompts (cosine < 0.90 to every eval prompt) with Olmo's own reply to the real-name version, names mapped back; prompts refused even with real names dropped. Control arm: 1,000 ordinary LMSYS prompts with Olmo's own replies. `train/`.
- Eval: your exp35 probe + HarmBench + JBB, 4 replies each. `eval/`.
- Result (`analysis.md`, `refusal_by_arm.png`): placeholder refusal 0.487 untrained, 0.109 fix, 0.415 control; fix minus control -0.306 (95% CI -0.382 to -0.234, paired over 112 prompts). HarmBench + JBB refusal 0.887 / 0.882 / 0.891.
- Refusal = keyword detector (`scripts/refusal.py`); on your exp35 replies it gives 46% / 13% / 8% against your strict labels' 41% / 12% / 11%. Hand check on 50 random replies per arm: about 93% agreement.
- Adapters: `adapters/fix`, `adapters/control` (PEFT format).
- One gap, one seed. Design written before building data: `fix_loop_name1.md`.
