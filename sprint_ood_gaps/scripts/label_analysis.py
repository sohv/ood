"""Unblind and analyse the 90 topic labels per designs/label_analysis.md (with its 2026-10-24 amendments).
Input: a JSON dump of the artifact's labels collection (list of {id, data}), key.json, astra_labels.json ({id: verdict}).
"unsure" (can't tell) topics are left out of the shares and counted per arm. Astra's labels get the same per-arm analysis,
plus Cohen's kappa against the user's labels. Output: analysis.md next to the dump, and a figure.
Usage: label_analysis.py [dump.json] [out_dir] [astra.json]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import cohen_kappa_score
ROOT = Path(__file__).resolve().parents[1]; L = ROOT / "runs/rare_gaps/labels"
dump_path = Path(sys.argv[1]) if len(sys.argv) > 1 else L / "labels_dump.json"
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else L
fig_dir = out_dir if len(sys.argv) > 2 else ROOT / "figs/labels"; fig_dir.mkdir(parents=True, exist_ok=True)
key = json.load(open(L / "key.json")); dump = json.load(open(dump_path)); astra = json.load(open(sys.argv[3] if len(sys.argv) > 3 else L / "astra_labels.json"))
lab = {d["id"]: d["data"] for d in dump if d["id"] in key}
d = pd.DataFrame([{"id": i, **key[i], "verdict": lab[i]["verdict"], "note": lab[i].get("note", ""), "astra": astra.get(i)} for i in lab])
ARMS = ("rare_only", "ratio_only", "control")

def arm_table(col, d):
    rows = []
    for arm in ARMS:
        g = d[d.arm == arm]; unsure = int((g[col] == "unsure").sum()); g = g[g[col] != "unsure"]; n = len(g)
        k = int((g[col] == "real_gap").sum()); k2 = int(g[col].isin(["real_gap", "partly"]).sum())
        lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
        rows.append({"arm": arm, "n": n, "unsure": unsure, "real_gap": k, "share": k / n if n else np.nan,
                     "wilson_lo": lo, "wilson_hi": hi, "real_or_partly": k2, "share_real_or_partly": k2 / n if n else np.nan})
    return pd.DataFrame(rows)

def cmp(col, a, b, d):
    ga, gb = (d[(d.arm == x) & (d[col] != "unsure")] for x in (a, b))
    ka, kb = int((ga[col] == "real_gap").sum()), int((gb[col] == "real_gap").sum())
    p = fisher_exact([[ka, len(ga) - ka], [kb, len(gb) - kb]])[1]; diff = ka / len(ga) - kb / len(gb)
    verdict = "finds gaps vs " + b if (diff >= 0.20 and p < 0.05) else ("inconclusive" if diff > 0 else "no excess")
    return f"- {a} vs {b}: real_gap share {ka}/{len(ga)} vs {kb}/{len(gb)}, difference {diff:+.2f}, Fisher p = {p:.3f} -> {verdict}"

out = [f"Labelled: {len(d)} of {len(key)} topics (user). Astra labels: {len(astra)} topics, all used for its own table.", "", "## User labels (primary)", ""]
tu = arm_table("verdict", d); out += [tu.round(3).to_markdown(index=False), ""]
out += [cmp("verdict", "rare_only", "control", d), cmp("verdict", "ratio_only", "control", d), cmp("verdict", "rare_only", "ratio_only", d), ""]
out += ["- Stefan's bar (real_gap share >= 0.50): " + ", ".join(f"{r.arm} {'met' if r.share >= 0.5 else 'not met'} ({r.share:.2f})" for r in tu.itertuples()), ""]
both = d[d.astra.notna() & (d.verdict != "unsure")]
da = pd.DataFrame([{"id": i, **key[i], "astra": astra[i]} for i in key if i in astra]); ta = arm_table("astra", da)
out += ["## Astra (GPT-6) labels, same screens", "", ta.round(3).to_markdown(index=False), "",
        cmp("astra", "rare_only", "control", da), cmp("astra", "ratio_only", "control", da), cmp("astra", "rare_only", "ratio_only", da), "",
        "## Agreement (topics both labelled, user's unsure excluded)", "",
        f"- n = {len(both)}; exact agreement {np.mean(both.verdict == both.astra):.2f}; Cohen's kappa (3 classes) {cohen_kappa_score(both.verdict, both.astra):.2f}; "
        f"kappa on real_gap vs not {cohen_kappa_score(both.verdict == 'real_gap', both.astra == 'real_gap'):.2f}", "",
        pd.crosstab(both.verdict.rename("user"), both.astra.rename("astra")).to_markdown(), "",
        "Context only: Sohan's deep_gap_loop labels on his loop gaps, 19 of 30 real (his labels, topic names shown)."]
(out_dir / "analysis.md").write_text("\n".join(out)); print("\n".join(out))

fig, ax = plt.subplots(figsize=(7, 4)); names = {"rare_only": "distance only", "ratio_only": "gap score only", "control": "random topics"}
for j, (t, who, c) in enumerate([(tu, "user", "#3b6ea8"), (ta, "Astra (GPT-6)", "#c77b30")]):
    x = np.arange(3) + (j - 0.5) * 0.35
    ax.bar(x, t.share, 0.35, color=c, label=who)
    ax.errorbar(x, t.share, yerr=[t.share - t.wilson_lo, t.wilson_hi - t.share], fmt="none", ecolor="black", capsize=3, lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{names[a]}\n(user n={n}, Astra n={m})" for a, n, m in zip(ARMS, tu.n, ta.n)])
ax.set_ylabel("share labelled real gap"); ax.set_ylim(0, 1); ax.legend(frameon=False)
ax.set_title("Share of topics labelled a real gap, by which score flagged them", fontsize=11)
fig.text(0.5, 0.005, "Olmo-3-7B traffic topics vs Dolci; 30 topics per group, blind; Wilson 95% intervals; user's can't-tell excluded", ha="center", fontsize=8, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(fig_dir / "real_gap_share.png", dpi=150)
