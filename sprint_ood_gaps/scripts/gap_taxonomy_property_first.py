"""Part 7 of designs/gap_taxonomy.md: property-first grouping. Pool = traffic prompts not verbatim in Dolci; distance = 1 - cosine
to the nearest Dolci prompt in MiniLM (runs/rare_gaps/one_sided.npz: exact float32 over all 1,710,578 Dolci prompts, exact copies
excluded). Far arm: the 5% of the pool farthest from Dolci. Control arm (added, not in the design): a random 5% of the pool.
Each arm: k-means, 50 clusters (sklearn, n_init 4, seed 0) on the unit-normalised MiniLM vectors; 10 random members per cluster
(seed 0) labelled with research-d4's unprimed labeller (scripts/gap_taxonomy_label.py, label_group), at most 6 calls at once.
Per cluster also: size, median distance, share of members in topics the step 6 loop confirmed (R50) or reviewed.
Writes runs/gap_taxonomy/property_first/{clusters.parquet, labels.jsonl, summary.json}; resumable (skips labelled ids)."""
import json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from gap_taxonomy_label import label_group, AXES
IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/gap_taxonomy/property_first"; OUT.mkdir(parents=True, exist_ok=True)
K, SHARE, SHOW = 50, 0.05, 10

member = np.isin(np.load(IDX / "traffic_hash.npy"), np.load(IDX / "mix_hash.npy"))
dist = 1 - np.load(ROOT / "runs/rare_gaps/one_sided.npz")["top_s"][:, 0]
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
pool = np.flatnonzero(~member); n = int(round(SHARE * len(pool)))
arms = {"far": pool[np.argsort(-dist[pool], kind="stable")[:n]], "random": np.sort(np.random.default_rng(0).choice(pool, n, replace=False))}
emb = np.load(IDX / "traffic.npy", mmap_mode="r")
d = json.load(open(ROOT / "runs/loop_onesided/v2/decision.json"))
r50 = {t["topic"] for g in d["gaps"]["R50"] for t in g}
reviewed = {json.loads(l)["topic"] for l in open(ROOT / "runs/loop_onesided/v2/reviewed_topics_R50.jsonl")}

rows = []
for arm, idx in arms.items():
    x = np.asarray(emb[idx], dtype=np.float32); x /= np.linalg.norm(x, axis=1, keepdims=True)
    c = KMeans(K, n_init=4, random_state=0).fit_predict(x)
    for k in range(K):
        m = idx[c == k]; show = np.random.default_rng(0).choice(m, min(SHOW, len(m)), replace=False)
        rows.append({"id": f"{arm}_{k}", "arm": arm, "size": len(m), "median_distance": float(np.median(dist[m])),
                     "share_in_r50_topics": float(np.isin(lab[m], list(r50)).mean()),
                     "share_in_reviewed_topics": float(np.isin(lab[m], list(reviewed)).mean()),
                     "n_exp37_topics": int(len(set(lab[m]))), "top_topic_share": float(Counter(lab[m]).most_common(1)[0][1] / len(m)),
                     "members": m.tolist(), "shown": show.tolist()})
    print(arm, "pool", len(idx), "distance range", float(dist[idx].min()), float(dist[idx].max()), flush=True)
C = pd.DataFrame(rows); C.drop(columns=["members"]).to_parquet(OUT / "clusters.parquet")
np.savez_compressed(OUT / "members.npz", **{r["id"]: np.array(r["members"]) for r in rows})

want = {i for r in rows for i in r["shown"]}; text = {}
for i, l in enumerate(open(IDX / "traffic_rows.jsonl")):
    if i in want: text[i] = json.loads(l)["text"]
lf = OUT / "labels.jsonl"
done = {json.loads(l)["id"] for l in open(lf)} if lf.exists() else set()
todo = [r for r in rows if r["id"] not in done]
print(len(todo), "clusters to label", flush=True)
def one(r):
    out = label_group([text[i] for i in r["shown"]]); return {"id": r["id"], "arm": r["arm"], **out}
with ThreadPoolExecutor(6) as ex, open(lf, "a") as f:
    for res in ex.map(one, todo):
        f.write(json.dumps(res) + "\n"); f.flush()

L = {(j := json.loads(l))["id"]: j for l in open(lf)}
C["defining_axis"] = [((L.get(i) or {}).get("parsed") or {}).get("defining", {}).get("axis") for i in C.id]
C["defining"] = [((L.get(i) or {}).get("parsed") or {}).get("defining", {}).get("description") for i in C.id]
C["property_axes"] = [sorted({p.get("axis") for p in (((L.get(i) or {}).get("parsed") or {}).get("properties") or [])}) for i in C.id]
C.drop(columns=["members"]).to_parquet(OUT / "clusters.parquet")
S = {}
for arm in arms:
    g = C[C.arm == arm]
    S[arm] = {"clusters": len(g), "failed_labels": int(g.defining_axis.isna().sum()),
              "defining_axis": dict(Counter(g.defining_axis.dropna())),
              "any_property_axis": dict(Counter(a for axes in g.property_axes for a in axes)),
              "median_distance": float(g.median_distance.median()),
              "median_share_in_r50_topics": float(g.share_in_r50_topics.median()),
              "clusters_mostly_in_r50_topics": int((g.share_in_r50_topics >= 0.5).sum()),
              "median_n_exp37_topics": float(g.n_exp37_topics.median())}
json.dump(S, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
