"""Step 10 of designs/one_sided_negatives.md (with its review amendment): what the corrected loop proposes after a kind is removed.
Reads runs/loop_removal/<arm>/decision.json and the full run (runs/loop_removal/full_check, else runs/loop_onesided/v2).
New gap = none of its topics confirmed in the full run and none same() as a full-run confirmed topic (centroid cosine >= 0.85 or
shared 60-character template, the loop's own rule). Writes runs/loop_removal/summary.json and new_gaps.md."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "runs/loop_removal"; IDX = ROOT / "runs/negatives/index"
sys.path.insert(0, str(ROOT.parent)); from src.data.pools import normalise
full_dir = R / "full_check" if (R / "full_check/decision.json").exists() else ROOT / "runs/loop_onesided/v2"
full = json.load(open(full_dir / "decision.json"))
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
emb = np.load(IDX / "traffic.npy", mmap_mode="r"); texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
cache = {}
def cent_sig(t):
    if t not in cache:
        m = np.flatnonzero(lab == t); c = np.asarray(emb[m], dtype=np.float32).mean(0); c /= np.linalg.norm(c)
        pre, cnt = Counter(normalise(texts[i])[:60] for i in m).most_common(1)[0]
        cache[t] = (c, pre if cnt >= 0.3 * len(m) else None)
    return cache[t]
def same(a, b):
    (ca, sa), (cb, sb) = cent_sig(a), cent_sig(b); return float(ca @ cb) >= 0.85 or (sa is not None and sa == sb)
H = ROOT / "runs/removal_why/hf/results/raw"
harm = {int(json.loads(l)["id"][1:]): json.loads(l)["is_harmful"] for l in open(H / "261028_safety_removal_v1/topics.jsonl")}
kl = {int(json.loads(l)["id"][1:]): json.loads(l) for l in open(H / "261028_removal_code_multilingual_kind_labels_v1/topic_labels.jsonl")}
q = pd.read_parquet(ROOT / "runs/removal_3b/queries.parquet"); ne_major = q.groupby("topic").non_en.mean().gt(0.5).to_dict()
KIND = {"safety": lambda t: bool(harm.get(t)), "coding": lambda t: bool(kl.get(t, {}).get("is_coding")),
        "all_non_english": lambda t: bool(ne_major.get(t))}
KIND2 = {"all_non_english": lambda t: bool(ne_major.get(t)) or bool(kl.get(t, {}).get("is_non_english"))}
base = {k: float(np.mean([f(t) for t in kl])) for k, f in KIND.items()}
full_conf = {arm: {t["topic"] for g in full["gaps"][arm] for t in g} for arm in ("R50", "N")}
def status(t, arm):
    rows = [json.loads(l) for l in open(full_dir / f"reviewed_topics_{arm}.jsonl")] if (full_dir / f"reviewed_topics_{arm}.jsonl").exists() else []
    rev = {int(r["topic"]) for r in rows}
    return "rejected" if t in rev else "not_reviewed"
summ, md = {"full_run": str(full_dir), "base_share": base}, ["# New gaps after removal (step 10)\n"]
for arm in ("R50", "N"):
    for kind in KIND:
        rows = []
        for a in [kind] + [f"rand_{kind}_{s}" for s in range(3)]:
            f = R / a / "decision.json"
            if not f.exists(): continue
            d = json.load(open(f)); new = []
            for g in d["gaps"][arm]:
                ts = [x["topic"] for x in g]
                if any(t in full_conf[arm] for t in ts) or any(same(t, u) for t in ts for u in full_conf[arm]): continue
                new.append({"topics": ts, "name": g[0].get("name"), "kind": any(KIND[kind](t) for t in ts),
                            "kind_secondary": any(KIND2.get(kind, KIND[kind])(t) for t in ts), "full_status": [status(t, arm) for t in ts]})
            nk = sum(x["kind"] for x in new)
            rows.append({"arm": a, "distinct_gaps": len(d["gaps"][arm]), "new_gaps": len(new), "new_of_kind": nk,
                         "new_of_kind_secondary": sum(x["kind_secondary"] for x in new), "kind_share_of_new": round(nk / len(new), 3) if new else None,
                         "new_verdict_changed": sum("rejected" in x["full_status"] for x in new)})
            if arm == "R50" and not a.startswith("rand_"):
                md.append(f"\n## {a} (R50): {len(new)} new gaps, {nk} of the removed kind\n")
                md += [f"- {x['name']} (topics {x['topics']}; kind {x['kind']}; full run: {', '.join(x['full_status'])})" for x in new]
        if rows:
            k, rnd = rows[0], [r["new_of_kind"] for r in rows[1:]]
            summ[f"{arm}/{kind}"] = {"runs": rows, "kind_minus_mean_random": round(k["new_of_kind"] - float(np.mean(rnd)), 2) if rnd else None,
                                     "above_every_random": bool(rnd) and all(k["new_of_kind"] > r for r in rnd), "chance_floor": "1/4 with 3 randoms"}
json.dump(summ, open(R / "summary.json", "w"), indent=1); open(R / "new_gaps.md", "w").write("\n".join(md))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in summ.items() if "/" in k}, indent=1))
