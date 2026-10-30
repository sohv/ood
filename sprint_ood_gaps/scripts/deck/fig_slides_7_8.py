"""Simplified main-slide figures for slides 7 and 8 (deck review runs/review/deck_vs_neel_2026-10-30.md, items 7-8).
Slide 7: defining property of the loop's confirmed vs rejected topics, 4 groups (from runs/gap_taxonomy/summary.json, written by
scripts/gap_taxonomy_analyse.py). Slide 8: six properties, traffic vs Dolci share with Wilson 95% intervals (from
runs/gap_taxonomy/facets/table.md). The full figures stay in backup: figs/gap_taxonomy/defining_axis.png, facets_traffic_vs_dolci.png."""
import json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
R = Path(__file__).resolve().parents[2]; F = R / "figs/gap_taxonomy"
if any("Ubuntu Sans" in f.name for f in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = "Ubuntu Sans"

# slide 7
d = json.load(open(R / "runs/gap_taxonomy/summary.json"))["defining_axis"]
cats = [("subject", ["subject"]), ("task type", ["task type"]), ("template or placeholder", ["format or template", "placeholder or artefact"]),
        ("other", ["language", "length or structure", "persona or roleplay", "adversarial or jailbreak", "tone or register", "other"])]
cols = ["#9aa5b1", "#c9d3dd", "#d9534f", "#eef1f5"]
rows = [("Confirmed gaps", "full: confirmed"), ("Rejected topics", "full: rejected")]
fig, ax = plt.subplots(figsize=(9, 3.2))
for i, (lab, key) in enumerate(rows):
    n = d[key]["n"]; left = 0
    for (cname, members), c in zip(cats, cols):
        k = sum(d[key][m] for m in members); w = k / n
        ax.barh(i, w, left=left, color=c, edgecolor="white", height=0.6, label=cname if i == 0 else None)
        if w > 0.06: ax.text(left + w / 2, i, f"{w:.0%}\n({k})", ha="center", va="center", fontsize=10)
        left += w
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{l}\n(n = {d[k]['n']})" for l, k in rows]); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("share of topics")
ax.set_title("Defining property of the loop's topics, by review outcome (Olmo-3-7B gap loop, one-sided MiniLM ranking)\n"
             "Claude Sonnet label on 10 fresh prompts per topic, loop name hidden", fontsize=10)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=9)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(F / "defining_axis_simple.png", dpi=150); plt.close(fig)

# slide 8
t = {}
for line in open(R / "runs/gap_taxonomy/facets/table.md"):
    m = re.match(r"\| (\w+) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \|", line)
    if m: t[m.group(1)] = [float(x) for x in m.groups()[1:]]
props = [("asks_about_assistant", "asks about the assistant"), ("roleplay_or_persona_setup", "roleplay or persona setup"),
         ("placeholder_or_artefact", "placeholder or truncated text"), ("non_english", "non-English"),
         ("homework_or_exam", "homework or exam"), ("templated_dataset_item", "templated dataset item")]
fig, ax = plt.subplots(figsize=(9.5, 4.2)); h = 0.38
for i, (k, lab) in enumerate(props):
    tr, trl, trh, do, dol, doh = t[k]
    for j, (v, lo, hi, c, name) in enumerate([(tr, trl, trh, "#e07b39", "traffic (n = 1,992)"), (do, dol, doh, "#3b6fb6", "Dolci (n = 1,988)")]):
        y = i + (j - 0.5) * h
        ax.barh(y, v, height=h, color=c, label=name if i == 0 else None)
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], color="black", capsize=2, lw=0.8)
        ax.text(hi + 0.006, y, f"{v:.1%}", va="center", fontsize=9)
ax.set_yticks(range(len(props))); ax.set_yticklabels([l for _, l in props]); ax.invert_yaxis()
ax.set_xlabel("share of prompts with the property (Wilson 95%)"); ax.set_xlim(0, 0.5)
ax.set_title("Prompt properties: real traffic (WildChat + LMSYS) vs Olmo-3 post-training prompts (Dolci)\n"
             "Claude Sonnet tags, source hidden; 6 of 34 properties, all with second-labeller kappa at least 0.68", fontsize=10)
ax.legend(loc="upper right", frameon=False)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(F / "facets_six.png", dpi=150); plt.close(fig)
print("ok", {k: t[k][0:1] + t[k][3:4] for k, _ in props})
