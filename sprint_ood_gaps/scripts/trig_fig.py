"""Figure for the trigger scan: Dolci refusal lift (Wilson lower bound) vs number of traffic prompts containing the pattern."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = pd.read_csv("runs/triggers/patterns.csv"); P = P[P.traffic_n >= 1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(P.traffic_n, P.lift_lo, s=4, c="0.7", lw=0, label=f"patterns with lift lower bound >= 3 (n = {len(P):,})")
lab = {"FMT:NAME_k": "NAME_k (control)", "name_2": "name_2", "answer]": "[your answer] template", "these guidelines": "Midjourney template",
       "ai called": None, "please adhere": None, "[insert": "[insert ...]", "2000 words": "2000 words", "slurs": "slurs", "derogatory": "derogatory",
       "provide me": "provide me", "roleplay": "roleplay", "language model": "language model"}
for k, t in lab.items():
    r = P[P.pattern == k]
    if len(r): ax.scatter(r.traffic_n, r.lift_lo, s=20, c="#c0392b" if "NAME" in k or k == "name_2" else "#2c6fbb")
    if len(r) and t: ax.annotate(t, (r.traffic_n.iat[0], r.lift_lo.iat[0]), fontsize=8, xytext=(4, 3) if k != "answer]" else (-60, 8), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("traffic prompts containing the pattern (of 992,933)"); ax.set_ylabel("Dolci refusal rate / base rate (Wilson 95% lower bound)")
ax.set_title("Prompt patterns that co-occur with refusals in Dolci, by frequency in user traffic", fontsize=11)
fig.text(0.5, 0.005, "Dolci SFT + DPO-chosen, 1,488,627 unique prompts (Tool Use source dropped), safety-refusal base rate 2.4%; unigrams, bigrams, format regexes", ha="center", fontsize=7.5, color="0.35")
ax.legend(frameon=False, fontsize=8, loc="upper right"); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("figs/triggers/lift_vs_traffic.png", dpi=150)
