"""Branch A bonus (designs/fix2_gap_quality.md): re-score the 210 distance_only topics after moving the fix training prompts
(or, as the comparison, the control training prompts) from traffic to the Dolci side. Distance: 1 - mean top-10 Dolci cosine,
all members, from runs/rare_gaps/one_sided.npz merged with the moved rows. Ratio: detector_update.py's union-kNN method
(k = 50, up to 60 members per topic). Moved rows and their exact copies are excluded from scoring. Writes runs/fix2a/detector.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd, faiss
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
RUN = ROOT / "runs/fix2a"; TRAIN = ROOT / "runs/pod_sync/oodgaps-fix2/train"
LINE, K, EX = 12.1413, 50, 30
faiss.omp_set_num_threads(8)
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv")
dist = sorted(tp[(tp.rare_rank <= 293) & (tp.his_rank > 293)].lab)
DLINE = float(tp.sort_values("rare_rank").median_score.iloc[292])  # median distance of the topic ranked 293
row_of = {p["id"]: p["row"] for p in map(json.loads, open(RUN / "teacher_pool.jsonl"))}
moved = {a: np.array([row_of[json.loads(l)["id"]] for l in open(TRAIN / f"train_{a}.jsonl")]) for a in ("fix", "control")}
lab = np.load(D / "labels.npy"); th = np.load(IDX / "traffic_hash.npy")
t = np.load(IDX / "traffic.npy").astype(np.float32)
z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); top_s = z["top_s"]
excl = np.zeros(len(t), bool)
for a in moved: excl[moved[a]] = True
excl |= np.isin(th, th[np.concatenate(list(moved.values()))])
res = {"distance_line_rank293": DLINE, "moved": {a: int(len(v)) for a, v in moved.items()}, "distance": {}, "ratio": {}}
mem = np.where(np.isin(lab, dist) & ~excl)[0]
before = 1 - top_s[mem].mean(1)
for a in ("none", "fix", "control"):
    if a == "none": sc = before
    else:
        s_new = t[mem] @ t[moved[a]].T
        sc = 1 - (-np.sort(-np.concatenate([top_s[mem], s_new], 1), 1)[:, :10]).mean(1)
    med = pd.Series(sc).groupby(lab[mem]).median()
    res["distance"][a] = {"n_topics": int(len(med)), "above_line": int((med > DLINE).sum()), "median_of_topic_medians": round(float(med.median()), 4)}
    res["distance"][a + "_per_topic"] = {int(k): round(float(v), 4) for k, v in med.items()}
# ratio: union kNN over traffic + Dolci for up to 60 members per topic
rng = np.random.default_rng(0)
q = np.concatenate([rng.choice(mem[lab[mem] == c], min(60, int((lab[mem] == c).sum())), replace=False) for c in dist])
m = np.load(IDX / "mix.npy").astype(np.float32); mh = np.load(IDX / "mix_hash.npy")
it = faiss.IndexFlatIP(384); it.add(t); im = faiss.IndexFlatIP(384); im.add(m)
st, ti = it.search(t[q], K + EX); sm, mi = im.search(t[q], K + EX)
st = np.where((ti == q[:, None]) | (th[ti] == th[q][:, None]), -np.inf, st); sm = np.where(mh[mi] == th[q][:, None], -np.inf, sm)
sims = np.concatenate([st, sm], 1); top = np.argsort(-sims, 1, kind="stable")[:, :K]
ids = np.concatenate([ti, np.full_like(mi, -1)], 1); nb = np.take_along_axis(ids, top, 1)
valid = np.isfinite(np.take_along_axis(sims, top, 1))
def ratio(mv_rows):
    mv = np.zeros(len(t), bool); mv[mv_rows] = True
    is_t = (nb >= 0) & valid; as_m = ((nb == -1) & valid) | (is_t & mv[np.where(nb >= 0, nb, 0)])
    n_t = (is_t & ~as_m).sum(1); n_m = as_m.sum(1)
    return ((n_t + 1) / (len(t) - mv.sum())) / ((n_m + 1) / (len(m) + mv.sum()))
for a, mv_rows in (("none", np.array([], int)), ("fix", moved["fix"]), ("control", moved["control"])):
    med = pd.Series(ratio(mv_rows)).groupby(lab[q]).median()
    res["ratio"][a] = {"n_topics": int(len(med)), "above_line": int((med > LINE).sum()), "median_of_topic_medians": round(float(med.median()), 3)}
res["ratio_queries"] = int(len(q))
(RUN / "detector.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "distance"}, indent=1))
print({a: res["distance"][a] for a in ("none", "fix", "control")})
