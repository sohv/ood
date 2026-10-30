"""Slide 2 summary figure: three small panels, one per finding. (a) topic split on word-built (tf-idf) topics, test-half topic AUROC
with topic-bootstrap 95% intervals (runs/topic_split/summary.json, "lsa"); Olmo activations = the layer picked on the dev half (24).
(b) planted removal vs existing difference: description groups of the removed kind, traffic share minus Dolci share before removal and
the change the removal adds (runs/gap_taxonomy/grouping_recovery/visible_check.json, "desc"). (c) NAME_1 probe, untrained
Olmo-3-7B-Instruct-DPO: refusal on 112 prompts with the NAME_k token vs the same prompts with real names, 4 replies each, 95% interval
by bootstrap over prompts (runs/fix_loop/per_prompt.csv). Writes figs/gap_taxonomy/summary_three_findings.png."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "figs/gap_taxonomy"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, GRAY, INK, MUTED = "#2a78d6", "#a8a69f", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

D = json.load(open(ROOT / "runs/topic_split/summary.json"))["lsa"]["detectors"]
a = [("Random", "random", GRAY), ("Length", "v1_length", GRAY), ("MiniLM\nkNN", "v1_knn_kth_1", BLUE), ("Words\n(tf-idf)", "v1_tfidf", BLUE), ("Olmo\nactivations", "act_l24_knn_1", BLUE)]
V = json.load(open(ROOT / "runs/gap_taxonomy/grouping_recovery/visible_check.json"))
P = pd.read_csv(ROOT / "runs/fix_loop/per_prompt.csv"); P = P[P.arm == "base"]
rng = np.random.default_rng(0)
def boot(x):
    x = np.asarray(x, float); m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]); return x.mean(), *np.percentile(m, [2.5, 97.5])

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 0.8]})
# (a)
x = np.arange(len(a)); v = [D[k]["test"]["auroc"] for _, k, _ in a]
lo = [D[k]["test"]["auroc"] - D[k]["test"]["lo"] for _, k, _ in a]; hi = [D[k]["test"]["hi"] - D[k]["test"]["auroc"] for _, k, _ in a]
for i in range(len(a)): ax[0].errorbar(x[i], v[i], yerr=[[lo[i]], [hi[i]]], fmt="o", ms=9, color=a[i][2], ecolor=MUTED, capsize=3, mec="white", mew=2)  # dots, not bars: AUROC has no zero baseline
for i, y in enumerate(v): ax[0].text(i, y + hi[i] + 0.01, f"{y:.2f}", ha="center", color=INK)
ax[0].axhline(0.5, color=MUTED, lw=1, ls="--"); ax[0].set_ylim(0.4, 0.75); ax[0].set_xlim(-0.5, len(a) - 0.5); ax[0].set_xticks(x, [n for n, *_ in a])
ax[0].set_ylabel("Test AUROC (0.5 = chance, higher is better)")
ax[0].set_title("1. Training-only detectors: absent vs present topics\ntopics built from words; 855 test topics; 95% CI", fontsize=11, loc="left")
# (b)
kinds = [("Safety", "safety"), ("Non-English", "multilingual")]
x = np.arange(len(kinds)); w = 0.36
nat = [100 * V[k]["desc"]["diff_full"] for _, k in kinds]; add = [100 * (V[k]["desc"]["diff_minus"] - V[k]["desc"]["diff_full"]) for _, k in kinds]
ax[1].bar(x - w / 2, nat, w, color=GRAY, label="already different (before removal)", edgecolor="white", linewidth=2)
ax[1].bar(x + w / 2, add, w, color=BLUE, label="added by removing that kind", edgecolor="white", linewidth=2)
for i in range(len(kinds)):
    ax[1].text(i - w / 2, nat[i] + 0.2, f"+{nat[i]:.1f}", ha="center", color=INK); ax[1].text(i + w / 2, add[i] + 0.2, f"+{add[i]:.1f}", ha="center", color=INK)
ax[1].set_xticks(x, [n for n, _ in kinds]); ax[1].set_ylabel("Traffic minus Dolci share (points)"); ax[1].set_ylim(0, 12.5)
ax[1].legend(frameon=False, fontsize=9, loc="upper right")
ax[1].set_title("2. Planted removal vs existing difference\ngroups of that kind; 2,000 prompts per side", fontsize=11, loc="left")
# (c)
r = [boot(P[P.set == s].refused) for s in ["probe_real_names", "probe_placeholder"]]
x = np.arange(2); v = [m for m, *_ in r]
ax[2].bar(x, v, width=0.55, color=[GRAY, BLUE], yerr=[[m - l for m, l, h in r], [h - m for m, l, h in r]], capsize=3, ecolor=MUTED, edgecolor="white", linewidth=2)
for i, (m, l, h) in enumerate(r): ax[2].text(i, h + 0.02, f"{m:.2f}", ha="center", color=INK)
ax[2].set_xticks(x, ["Real names", "NAME_1 token"]); ax[2].set_ylim(0, 0.65); ax[2].set_ylabel("Share of replies refused")
ax[2].set_title("3. Olmo-3-7B refusal, NAME_1 probe\n112 prompts, 4 replies each; 95% CI", fontsize=11, loc="left")
for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False); a_.grid(axis="y", color="#e6e5e1", lw=0.8); a_.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "summary_three_findings.png", dpi=200, facecolor="white")
print("saved", {"a": v if False else [D[k]["test"]["auroc"] for _, k, _ in a], "b_nat": nat, "b_add": add, "c": r})

# standalone panel 1 for slide 5 (figs/topic_split/detectors_word_topics.png)
f2, b = plt.subplots(figsize=(6.4, 4.4)); x = np.arange(len(a)); v = [D[k]["test"]["auroc"] for _, k, _ in a]
for i in range(len(a)):
    b.errorbar(x[i], v[i], yerr=[[lo[i]], [hi[i]]], fmt="o", ms=9, color=a[i][2], ecolor=MUTED, capsize=3, mec="white", mew=2)
    b.text(i, v[i] + hi[i] + 0.01, f"{v[i]:.2f}", ha="center", color=INK)
b.axhline(0.5, color=MUTED, lw=1, ls="--"); b.set_ylim(0.4, 0.75); b.set_xlim(-0.5, len(a) - 0.5); b.set_xticks(x, [n for n, *_ in a])
b.set_ylabel("Test AUROC (0.5 = chance, higher is better)")
b.set_title("Training-only detectors, absent vs present traffic topics\ntopics built from words (tf-idf); 855 test topics; topic-bootstrap 95% CI", fontsize=11, loc="left")
b.spines[["top", "right"]].set_visible(False); b.grid(axis="y", color="#e6e5e1", lw=0.8); b.set_axisbelow(True)
f2.tight_layout(); (ROOT / "figs/topic_split").mkdir(exist_ok=True); f2.savefig(ROOT / "figs/topic_split/detectors_word_topics.png", dpi=200, facecolor="white")
