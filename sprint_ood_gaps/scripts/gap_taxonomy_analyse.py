"""Gap taxonomy tables (designs/gap_taxonomy.md). Reads runs/gap_taxonomy/{reviews,labels,name_fit,sole,resample}.jsonl.
Groups: full-run confirmed vs rejected (R50 of runs/loop_onesided/v2); per removal arm, topics confirmed there but not in the full run
("new", topic-level proxy for step 10's rule, which also merges same() topics). Writes runs/gap_taxonomy/summary.json, tables.md,
figs/gap_taxonomy/defining_axis.png."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"; F = ROOT / "figs/gap_taxonomy"; F.mkdir(parents=True, exist_ok=True)
AX = ["subject", "task type", "language", "format or template", "length or structure", "persona or roleplay",
      "adversarial or jailbreak", "tone or register", "placeholder or artefact", "other"]
def rd(n):
    if not (D / n).exists(): raise SystemExit(f"missing {D / n}; the tables would silently count it as empty")
    return [json.loads(l) for l in open(D / n)]
rev = rd("reviews.jsonl"); lab = {r["topic"]: r["parsed"] for r in rd("labels.jsonl") if r["parsed"]}
fit = {tuple(r["key"]): r["parsed"] for r in rd("name_fit.jsonl") if r["parsed"]}
ax = lambda t: lab[t]["defining"]["axis"] if t in lab and lab[t]["defining"]["axis"] in AX else ("other" if t in lab else None)

full_conf = {r["topic"] for r in rev if r["arm"] == "full" and r["confirmed"]}
full_rej = {r["topic"] for r in rev if r["arm"] == "full" and not r["confirmed"]}
arms = sorted({r["arm"] for r in rev} - {"full", "full_check"})
new = {a: {r["topic"] for r in rev if r["arm"] == a and r["confirmed"]} - full_conf for a in arms}
groups = {"full: confirmed": full_conf, "full: rejected": full_rej, "new after safety removal": new.get("safety", set()),
          "new after coding removal": new.get("coding", set()), "new after non-English removal": new.get("all_non_english", set()),
          "new after random removals (pooled)": set().union(*[v for a, v in new.items() if a.startswith("rand_")])}
dist = {}
for g, ts in groups.items():
    c = Counter(ax(t) for t in ts if ax(t)); n = sum(c.values()); dist[g] = {"n": n, **{a: c.get(a, 0) for a in AX}}

# how often the loop's name describes the defining property, and what axis the name is on
pairs = {(r["topic"], r["name"]) for r in rev if r["arm"] == "full"}
fits = Counter((fit.get(p) or {}).get("fits") for p in pairs if p in fit)
by_axis = {}
for p in pairs:
    if p in fit and ax(p[0]): by_axis.setdefault(ax(p[0]), Counter())[fit[p].get("fits")] += 1
name_axes = Counter((fit.get(p) or {}).get("name_axis") for p in pairs if p in fit)
# non-subject properties shared by most or all prompts, defining or not
nonsub = {a: sum(any(pp.get("axis") == a and pp.get("share") in ("all", "most") for pp in lab[t]["properties"]) for t in lab) for a in AX}

def kappa(a, b):
    a, b = np.array(a), np.array(b); po = (a == b).mean(); cats = sorted(set(a) | set(b))
    pe = sum((a == c).mean() * (b == c).mean() for c in cats); return float((po - pe) / (1 - pe)) if pe < 1 else 1.0, float(po)
sole = {r["topic"]: r["parsed"]["defining"]["axis"] for r in rd("sole.jsonl") if r["parsed"] and r["parsed"].get("defining")}
rs = {r["topic"]: r["parsed"]["defining"]["axis"] for r in rd("resample.jsonl") if r["parsed"] and r["parsed"].get("defining")}
ks = kappa([ax(t) for t in sole if ax(t)], [sole[t] for t in sole if ax(t)]) if sole else None
kr = kappa([ax(t) for t in rs if ax(t)], [rs[t] for t in rs if ax(t)]) if rs else None
# subject vs non-subject agreement (coarse), the distinction the claim rests on
coarse = lambda x: "subject" if x == "subject" else "non-subject"
ks2 = kappa([coarse(ax(t)) for t in sole if ax(t)], [coarse(sole[t]) for t in sole if ax(t)]) if sole else None
kr2 = kappa([coarse(ax(t)) for t in rs if ax(t)], [coarse(rs[t]) for t in rs if ax(t)]) if rs else None

summ = {"n_labelled": len(lab), "n_units": len({r["topic"] for r in rev}), "defining_axis": dist,
        "name_fit_full_run": dict(fits), "name_fit_by_axis_full_run": {k: dict(v) for k, v in by_axis.items()},
        "name_axis_full_run": dict(name_axes), "nonsubject_shared_by_most": nonsub,
        "sole_vs_sonnet": {"n": len(sole), "kappa_axis": ks, "kappa_subject_vs_not": ks2},
        "resample_vs_first": {"n": len(rs), "kappa_axis": kr, "kappa_subject_vs_not": kr2}}
json.dump(summ, open(D / "summary.json", "w"), indent=1)

md = ["# Gap taxonomy tables\n", "Defining property of each topic (unprimed Sonnet label on 10 fresh member prompts), share of topics per group.\n",
      "| group | n | " + " | ".join(AX) + " |", "|---|---|" + "---|" * len(AX)]
for g, d in dist.items():
    md.append(f"| {g} | {d['n']} | " + " | ".join(f"{d[a] / d['n']:.2f}" if d["n"] else "-" for a in AX) + " |")
md += ["", f"Name describes the defining property (full run, {sum(fits.values())} topic-name pairs): {dict(fits)}",
       f"Axis of the loop's names: {dict(name_axes)}", f"Non-subject properties shared by most or all prompts (any topic, not only defining): {nonsub}",
       f"Sole vs Sonnet on the defining axis: {summ['sole_vs_sonnet']}", f"Second sample vs first: {summ['resample_vs_first']}"]
(D / "tables.md").write_text("\n".join(md) + "\n")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
gs = [g for g in dist if dist[g]["n"]]; fig, axp = plt.subplots(figsize=(11, 0.6 * len(gs) + 2)); left = np.zeros(len(gs))
cmap = plt.get_cmap("tab10")
for i, a in enumerate(AX):
    v = np.array([dist[g][a] / dist[g]["n"] for g in gs]); axp.barh(gs, v, left=left, color=cmap(i), label=a); left += v
axp.invert_yaxis(); axp.set_xlabel("share of topics"); axp.set_xlim(0, 1)
axp.set_title("Defining property of loop topics, by group (Olmo-3-7B gap loop, one-sided MiniLM ranking)\n"
              "unprimed Claude Sonnet label on 10 fresh member prompts per topic; n per group: " + ", ".join(str(dist[g]["n"]) for g in gs), fontsize=9)
axp.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15)); fig.tight_layout(); fig.savefig(F / "defining_axis.png", dpi=150)
print(open(D / "tables.md").read())
