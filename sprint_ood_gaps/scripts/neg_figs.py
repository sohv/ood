"""Two compact figures for the negatives eval (replace the 62-row strip plot as the headline figures).
A: figs/negatives/auroc_planted_vs_negatives.png - length-stratified AUROC per score, legal and medical.
B: figs/negatives/ratio_by_family.png - neighbour ratio per family, unseen negatives pooled vs covered Dolci pooled, plus planted."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figs/negatives"; FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})

# A
st = pd.read_csv(ROOT / "runs/negatives/report/auroc_length_stratified.csv")
names = {"one_sided": "Distance to nearest Dolci prompt", "ratio": "Gap score (k=50 traffic/Dolci ratio)",
         "tfidf_p_traffic": "Bag-of-words traffic-vs-Dolci classifier", "olmo_len": "Prompt length (Olmo tokens)"}
order = ["one_sided", "ratio", "tfidf_p_traffic", "olmo_len"]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for k, s in enumerate(order):
    for p, c, dy in (("pos_legal", BLUE, 0.12), ("pos_medical", ORANGE, -0.12)):
        v = st[(st.planted == p) & (st.score == s)].auroc_length_stratified.iat[0]
        ax.plot([v], [len(order) - 1 - k + dy], "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
        ax.text(v + 0.012, len(order) - 1 - k + dy, f"{v:.2f}", va="center", fontsize=9, color=INK)
ax.axvline(0.5, color=GREY, lw=1, ls="--"); ax.text(0.505, -0.75, "chance", color=MUTED, fontsize=8.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[s] for s in order][::-1])
ax.set_xlim(0.45, 1.0); ax.set_ylim(-0.9, len(order) - 0.4); ax.set_xlabel("AUROC, planted gap prompts vs unseen same-kind negatives")
ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ax.plot([], [], "o", color=BLUE, label="Planted legal (r/legaladvice, n=481)"); ax.plot([], [], "o", color=ORANGE, label="Planted medical (HealthCareMagic, n=479)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5)
fig.suptitle("Separating planted gaps from same-kind negatives, Olmo-3-7B Dolci mix", x=0.02, ha="left", fontsize=11.5, y=1.06)
fig.text(0.02, 0.965, "MiniLM embeddings; AUROC within 6 Olmo-length bands; negatives: 23 same-kind datasets, 19,364 prompts, exact Dolci members removed", fontsize=8.5, color=MUTED)
fig.savefig(FIG / "auroc_planted_vs_negatives.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# B
ps = pd.read_csv(ROOT / "runs/negatives/report/per_set.csv")[["set", "kind", "group"]]
rows = []
for f in glob.glob(str(ROOT / "runs/negatives/scores/*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("kind") == "negative" and r.get("exact_in_mix"): continue
        rows.append({"set": r["set"], "ratio": r["ratio"]})
d = pd.DataFrame(rows).merge(ps, on="set")
fams = [g for g in ["code", "math", "reasoning", "tool_use", "precise_if", "science", "flan_style", "table", "general_chat", "safety"] if g in d.group.unique()]
pretty = {"code": "Code", "math": "Maths", "reasoning": "Logic puzzles", "tool_use": "Tool use", "precise_if": "Instruction following",
          "science": "Science", "flan_style": "FLAN-style tasks", "table": "Tables", "general_chat": "General chat", "safety": "Safety"}
items = []
for g in fams:
    for kind, c in (("negative", BLUE), ("covered", GREY)):
        x = d[(d.group == g) & (d.kind == kind)].ratio.values
        if len(x): items.append((f"{pretty[g]}: {'unseen' if kind == 'negative' else 'Dolci'}", x, c, len(x)))
for p, lab in (("pos_legal", "Planted legal"), ("pos_medical", "Planted medical")):
    items.append((lab, d[d.set == p].ratio.values, ORANGE, int((d.set == p).sum())))
fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(items) + 1.2))
for i, (lab, x, c, n) in enumerate(items):
    y = len(items) - 1 - i
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    ax.plot([q1, q3], [y, y], color=c, lw=6, solid_capstyle="round", alpha=0.9)
    ax.plot([med], [y], "o", color="white", mec=c, mew=2, ms=7, zorder=3)
ax.set_yticks(range(len(items))); ax.set_yticklabels([f"{lab}  (n={n:,})" for lab, _, _, n in items][::-1], fontsize=8.5)
ax.set_xscale("log"); ax.set_xlim(0.025, 30)
ax.axvline(12.1413, color=INK, lw=1, ls="--"); ax.text(12.9, len(items) - 0.6, "gap line 12.1", fontsize=8.5, color=INK)
ax.set_xlabel("Gap score: (traffic neighbours + 1)/|traffic| over (Dolci neighbours + 1)/|Dolci|, log scale")
ax.grid(axis="x", color="#ecebe7", lw=0.8, which="major"); ax.set_axisbelow(True)
for b in range(1, len(fams)):
    pass
fig.suptitle("Gap score by data family, Olmo-3-7B Dolci mix vs WildChat + LMSYS traffic", x=0.02, ha="left", fontsize=11.5, y=1.0)
fig.text(0.02, 1 - 0.55 / (0.3 * len(items) + 1.2), "Bar = middle 50% of prompts, dot = median. Blue: unseen datasets of the same kind; grey: Dolci's own prompts; orange: planted gaps",
         fontsize=8, color=MUTED)
fig.savefig(FIG / "ratio_by_family.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
