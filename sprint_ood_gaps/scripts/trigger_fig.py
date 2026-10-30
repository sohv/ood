"""Dose-response figure for designs/planted_trigger.md: trigger effect (refusal with token minus without, relative to k = 0) vs planted rows k."""
import json
from pathlib import Path
import matplotlib, matplotlib.ticker; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = Path(__file__).resolve().parents[1]; E = json.load(open(ROOT / "runs/planted_trigger/analysis/summary.json"))["effects"]
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b9b9b3"
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (tag_sfx, title) in zip(axes, (("", "held-out templates (pre-set test)"), ("_trainTpl", "training templates (check added after)"))):
    for t, tag, col, lab in (("red", "redacted", BLUE, "[REDACTED] (natural)"), ("zep", "zephyrine", ORANGE, "Zephyrine (made up)")):
        for i, k in enumerate((25, 50, 100, 200)):
            for key, v in E.items():
                if key.startswith(f"{t}_p{k}|") and key.endswith("|" + tag + tag_sfx):
                    x = k * (1.06 if t == "zep" else 0.94)
                    ax.errorbar(x, v["effect"], yerr=[[v["effect"] - v["ci95"][0]], [v["ci95"][1] - v["effect"]]], fmt="o", color=col, ms=5, capsize=2,
                                label=lab if (i == 0 and "|s0|" in key) else None)
        for key, v in E.items():
            if key.startswith(f"{t}_c") and key.endswith("|" + tag + tag_sfx):
                k = int(key.split("|")[0].split("_c")[1]); ax.errorbar(k * (1.06 if t == "zep" else 0.94), v["effect"], yerr=[[v["effect"] - v["ci95"][0]], [v["ci95"][1] - v["effect"]]], fmt="x", color=col, ms=6, capsize=2)
    ax.axhline(0, color="black", lw=0.6); ax.axhline(0.10, color=NEUTRAL, lw=1, ls="--"); ax.text(120, 0.104, "pre-set bar 0.10", fontsize=8, color="#666")
    ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200], ["25", "50", "100", "200"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("rows with token (k) in 1,700 training rows, log scale")
    ax.set_title(title, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("refusal with token minus without,\nminus the same for k = 0 (95% CI, 150 prompts)"); axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", title="o planted, x control", title_fontsize=8)
axes[0].set_ylim(-0.12, 0.15)
fig.suptitle("Olmo-3-7B-Instruct-DPO + LoRA: refusal of harmless prompts carrying a planted token, by number of planted refusal rows", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "figs/planted_trigger/dose_response.png", dpi=150); print(ROOT / "figs/planted_trigger/dose_response.png")
