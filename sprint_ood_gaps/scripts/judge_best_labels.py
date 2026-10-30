"""Blind label set for judge_best (designs/judge_best.md, comparison 3). Judge top 30 by primary gap score (ties broken at random,
seed 0) vs distance top 30 by rare_rank; label only topics in one list and not the other. Same 10 prompts the judge saw, each
with 2 MiniLM and 2 BM25 Dolci neighbours (no scores, no source tags), shuffled (seed 0), opaque ids.
Writes runs/judge_best/labels/{items.json, key.json, judge_best_labels.html} and the overlap counts."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/judge_best"; LB = OUT / "labels"
P = pd.DataFrame([json.loads(l) for l in open(OUT / "primary_scores.jsonl")])
assert P.gap_score.notna().all(), f"{P.gap_score.isna().sum()} topics without a score"
rng = np.random.default_rng(0); P["tie"] = rng.random(len(P))
judge30 = list(P.sort_values(["gap_score", "tie"], ascending=[False, True]).topic.head(30))
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv"); tp = tp[tp["size"] >= 50]
dist30 = list(tp.sort_values("rare_rank").lab.astype(int).head(30))
both = sorted(set(judge30) & set(dist30)); jo = [t for t in judge30 if t not in dist30]; do = [t for t in dist30 if t not in judge30]
T = {t["topic"]: t for t in json.load(open(OUT / "topics.json"))}; snip = json.load(open(OUT / "mix_snippets.json")); size = dict(zip(tp.lab.astype(int), tp["size"]))
items, key = [], {}
for arm, ids in (("judge_only", jo), ("distance_only", do)):
    for t in ids:
        iid = "j" + hashlib.sha1(f"jb-{t}".encode()).hexdigest()[:8]
        key[iid] = {"arm": arm, "topic": t, "judge_rank": judge30.index(t) + 1 if t in judge30 else None, "distance_rank": dist30.index(t) + 1 if t in dist30 else None}
        ex = []
        for e in T[t]["examples"]:
            nb = []
            for k in e["minilm"][:2] + [m for m in e["bm25"] if m not in e["minilm"][:2]][:2]:
                if k not in nb: nb.append(k)
            ex.append({"text": e["text"], "verbatim": e["verbatim"], "nb": [snip[str(k)] for k in nb]})
        items.append({"id": iid, "size": int(size[t]), "examples": ex})
order = rng.permutation(len(items)); items = [items[i] for i in order]
for k, it in enumerate(items): it["order"] = k
json.dump(items, open(LB / "items.json", "w"), ensure_ascii=False, indent=1); json.dump(key, open(LB / "key.json", "w"), indent=1)
page = open(LB / "page_template.html").read().replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
(LB / "judge_best_labels.html").write_text(page)
ov = {"judge_top30": judge30, "distance_top30": dist30, "both": both, "judge_only": jo, "distance_only": do, "fallback_in_judge30": int(P[P.topic.isin(judge30)].source.eq("sol_fallback").sum())}
json.dump(ov, open(OUT / "top30_overlap.json", "w"), indent=1)
print(f"overlap {len(both)} of 30; judge-only {len(jo)}, distance-only {len(do)}; items {len(items)}; page {(LB / 'judge_best_labels.html').stat().st_size} bytes")
