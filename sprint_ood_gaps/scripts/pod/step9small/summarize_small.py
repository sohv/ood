"""Step 9 small summary (designs/step9_small.md): step 3b's measures for Olmo activation kNN (layers 8/20/28; k = 1 and mean of 5; std
from each arm's reference, primary, and from the full subsample, ablation) next to MiniLM, tf-idf and length on the same 20% subsample,
plus MiniLM/tf-idf on full Dolci from step 3b for the subsampling check. CPU, research-box, after the pod has finished.
usage: .venv/bin/python -I scripts/pod/step9small/summarize_small.py [RESULTS] [BASELINES] [OUT]
  RESULTS   local copy of hf://datasets/detectood/ood/results/raw/261029_step9_small_v1 (default runs/step9small/hf), fetched with e.g.
            .venv/bin/hf download detectood/ood --repo-type dataset --include "results/raw/261029_step9_small_v1/*" --local-dir runs/step9small/hf_repo
            and RESULTS = runs/step9small/hf_repo/results/raw/261029_step9_small_v1
  BASELINES output of removal_3b_sub.py (default runs/step9small/baselines); OUT default runs/step9small/summary
Writes OUT/summary.json, OUT/table.csv, figs/step9small/removal_small.png."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent)); from measures_small import summarize, notices, KINDS

ROOT = Path(__file__).resolve().parents[3]; a = sys.argv[1:]
RES = Path(a[0]) if a else ROOT / "runs/step9small/hf"; BASE = Path(a[1]) if len(a) > 1 else ROOT / "runs/step9small/baselines"
OUT = Path(a[2]) if len(a) > 2 else ROOT / "runs/step9small/summary"; OUT.mkdir(parents=True, exist_ok=True)
gates = json.load(open(RES / "gates.json")); print("gates", gates)
q = pd.read_parquet(BASE / "queries_sub.parquet"); dets = ["minilm", "minilm_m5", "tfidf", "length"]; new = {}
for L in [8, 20, 28]:
    p = RES / f"scores_L{L}.npz"
    if not p.exists(): print("missing", p); continue
    r = np.load(p); assert np.array_equal(r["traffic_idx"], q.traffic_idx.values), "query order differs"
    for k in ["k1", "m5"]:
        for suf in ["", "_fixedstd"]:
            det = f"actL{L}_{k}{suf}"; dets.append(det); new[f"{det}__full"] = r[f"full__{k}"]
            for nm in r.files:
                if nm.endswith(f"__{k}{suf}") and not nm.startswith("full__"): new[f"{det}__{nm[:-len(f'__{k}{suf}')]}"] = r[nm]
q = pd.concat([q, pd.DataFrame(new)], axis=1)
summ = summarize(q, dets); full3b = json.load(open(ROOT / "runs/removal_3b/summary.json"))
rows = []
for key, r in summ.items():
    det, k = key.split("/"); a_ = r.get("prompt_auroc_change", r.get("topic_auroc_change")); rnd = r.get("random_prompt_auroc", r.get("random_topic_auroc"))
    rows.append({"detector": det, "kind": k, "auroc_change": a_, "random_max": max(rnd), "random_mean": round(float(np.mean(rnd)), 3),
                 "topic_gain": r.get("topic_gain"), "notices": notices(r)})
for det in ["minilm", "tfidf"]:   # full-Dolci step 3b, same measures: did subsampling change the reading?
    for k in KINDS:
        r = full3b[f"{det}/{k}"]; a_ = r.get("prompt_auroc_change", r.get("topic_auroc_change")); rnd = r.get("random_prompt_auroc", r.get("random_topic_auroc"))
        rows.append({"detector": f"{det}_full3b", "kind": k, "auroc_change": a_, "random_max": max(rnd), "random_mean": round(float(np.mean(rnd)), 3),
                     "topic_gain": r.get("topic_gain"), "notices": notices(r)})
t = pd.DataFrame(rows); t.to_csv(OUT / "table.csv", index=False)
json.dump({"gates": gates, "measures": summ}, open(OUT / "summary.json", "w"), indent=1)
print(t.pivot(index="detector", columns="kind", values="auroc_change").to_string())
print(t.pivot(index="detector", columns="kind", values="notices").to_string())

# figure: as figs/removal_3b/removal_3b.png, primary detectors only
show = [("minilm_full3b", "MiniLM, full Dolci (3b)", "#9fb7dc"), ("minilm", "MiniLM", "#3b6fb6"), ("tfidf", "one-sided tf-idf", "#d07a2b"),
        ("actL8_k1", "Olmo L8", "#7a9a3a"), ("actL20_k1", "Olmo L20", "#2e7d32"), ("actL28_k1", "Olmo L28", "#174d1a")]
show = [s for s in show if s[0] in set(t.detector)]
labels = {"safety": "safety (topics)", "coding": "coding (topics)", "aya_only": "Aya removed (prompts)", "all_non_english": "all non-English removed (prompts)"}
fig, ax = plt.subplots(figsize=(11, 4.4)); w = 0.8 / len(show)
for i, k in enumerate(KINDS):
    for j, (det, lab, col) in enumerate(show):
        x = i - 0.4 + w * (j + 0.5)
        if det.endswith("_full3b"): r = full3b[f"{det[:-7]}/{k}"]
        else: r = summ[f"{det}/{k}"]
        a_ = r.get("prompt_auroc_change", r.get("topic_auroc_change")); rnd = r.get("random_prompt_auroc", r.get("random_topic_auroc"))
        ax.scatter(np.full(len(rnd), x) + np.random.default_rng(0).uniform(-w / 4, w / 4, len(rnd)), rnd, s=8, color="grey", alpha=0.6, zorder=2)
        ax.scatter([x], [a_], s=60, color=col, zorder=3, label=lab if i == 0 else None)
ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.axhline(0.70, color="grey", ls="--", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([labels[k] for k in KINDS], fontsize=8.5); ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUROC of score change, removed kind vs rest"); ax.legend(fontsize=7.5, loc="upper right", ncol=2)
ax.set_title("Removal test: Olmo-3-7B-Instruct-DPO activation kNN (k = 1, per-arm std) vs MiniLM and tf-idf, 20% Dolci subsample", fontsize=9.5)
fig.text(0.5, 0.005, "342,116 Dolci first turns (seed 20261029) as reference; 49,629 traffic queries. Coloured: kind removed. Grey: 20 random removals "
         "of the same size. Dashed: 0.70 bar. Light blue: MiniLM on full Dolci (step 3b).", ha="center", fontsize=7)
FIG = Path(os.environ.get("FIGDIR", ROOT / "figs/step9small")); FIG.mkdir(parents=True, exist_ok=True)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "removal_small.png", dpi=150); print("wrote", FIG / "removal_small.png")
