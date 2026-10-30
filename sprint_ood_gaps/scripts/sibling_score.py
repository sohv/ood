"""Sibling split, step 2 (designs/sibling_split.md): for each topic and ordered sub-topic pair (add X, score Y), move X (sibling),
the same number of random outside traffic prompts (control), or half of Y (self) to the Dolci side, and re-score Y on distance
(1 - mean top-10 Dolci cosine) and the k = 50 union-kNN ratio (detector_update.py's method). Writes runs/sibling_split/scores.json."""
import itertools, json
from pathlib import Path
import numpy as np, pandas as pd, faiss
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/sibling_split"
DLINE, RLINE, K, EX = 0.39187515, 12.1413, 50, 30
faiss.omp_set_num_threads(8)
split = json.load(open(OUT / "split.json"))
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
th = np.load(IDX / "traffic_hash.npy"); t = np.load(IDX / "traffic.npy").astype(np.float32)
top_s = np.load(ROOT / "runs/rare_gaps/one_sided.npz")["top_s"]
sub = {(s["topic"], k): np.array(sorted(int(i) for i, v in s["assign"].items() if v == k)) for s in split for k in "ABC"}
used = [s for s in split if all(len(sub[(s["topic"], k)]) >= 40 for k in "ABC")]
rng = np.random.default_rng(0)
outside = np.where(~np.isin(lab, [s["topic"] for s in split]))[0]

# ratio neighbours once for every sub-topic member
q = np.unique(np.concatenate([sub[(s["topic"], k)] for s in used for k in "ABC"]))
m = np.load(IDX / "mix.npy").astype(np.float32); mh = np.load(IDX / "mix_hash.npy")
it = faiss.IndexFlatIP(384); it.add(t); im = faiss.IndexFlatIP(384); im.add(m)
st, ti = it.search(t[q], K + EX); sm, mi = im.search(t[q], K + EX)
st = np.where((ti == q[:, None]) | (th[ti] == th[q][:, None]), -np.inf, st); sm = np.where(mh[mi] == th[q][:, None], -np.inf, sm)
sims = np.concatenate([st, sm], 1); top = np.argsort(-sims, 1, kind="stable")[:, :K]
nb = np.take_along_axis(np.concatenate([ti, np.full_like(mi, -1)], 1), top, 1); valid = np.isfinite(np.take_along_axis(sims, top, 1))
qpos = {int(r): k for k, r in enumerate(q)}

def scores(rows_, moved):
    # distance and ratio for traffic rows rows_ after moving `moved` to the Dolci side
    s_new = t[rows_] @ t[moved].T if len(moved) else np.zeros((len(rows_), 0), np.float32)
    dist = 1 - (-np.sort(-np.concatenate([top_s[rows_], s_new], 1), 1)[:, :10]).mean(1)
    mv = np.zeros(len(t), bool); mv[moved] = True; k = np.array([qpos[int(r)] for r in rows_])
    is_t = (nb[k] >= 0) & valid[k]; as_m = ((nb[k] == -1) & valid[k]) | (is_t & mv[np.where(nb[k] >= 0, nb[k], 0)])
    ratio = (((is_t & ~as_m).sum(1) + 1) / (len(t) - mv.sum())) / ((as_m.sum(1) + 1) / (len(m) + mv.sum()))
    return float(np.median(dist)), float(np.median(ratio))

rows_out = []
for s in used:
    c = s["topic"]
    for X, Y in itertools.permutations("ABC", 2):
        x, y = sub[(c, X)], sub[(c, Y)]
        half = rng.permutation(y); y_mv, y_keep = np.sort(half[: len(y) // 2]), np.sort(half[len(y) // 2:])
        arms = {"sibling": (y, x), "control": (y, rng.choice(outside, len(x), replace=False)), "self": (y_keep, y_mv)}
        r = {"topic": c, "add": X, "score": Y, "n_add": int(len(x)), "n_score": int(len(y)),
             "centroid_cos": float(np.dot(t[x].mean(0) / np.linalg.norm(t[x].mean(0)), t[y].mean(0) / np.linalg.norm(t[y].mean(0))))}
        for a, (rows_, moved) in arms.items():
            keep = rows_[~np.isin(th[rows_], th[moved])]
            b, aft = scores(keep, np.array([], int)), scores(keep, moved)
            r[f"{a}_dist_before"], r[f"{a}_dist_after"], r[f"{a}_ratio_before"], r[f"{a}_ratio_after"] = b[0], aft[0], b[1], aft[1]
        rows_out.append(r)
d = pd.DataFrame(rows_out)
res = {"topics_used": [s["topic"] for s in used], "n_pairs": len(d)}
for sc in ("dist", "ratio"):
    for a in ("sibling", "control", "self"): d[f"d_{a}_{sc}"] = d[f"{a}_{sc}_after"] - d[f"{a}_{sc}_before"]
    d[f"U_{sc}"] = (d[f"d_sibling_{sc}"] - d[f"d_control_{sc}"]) / (d[f"d_self_{sc}"] - d[f"d_control_{sc}"])
    per_topic = d.groupby("topic")[f"U_{sc}"].mean()
    b = np.array([rng.choice(per_topic.to_numpy(), len(per_topic)).mean() for _ in range(10000)])
    line = DLINE if sc == "dist" else RLINE
    res[sc] = {"U_mean": round(float(per_topic.mean()), 3), "U_ci95": np.quantile(b, [.025, .975]).round(3).tolist(), "U_per_topic": per_topic.round(3).to_dict(),
               "median_change": {a: round(float(d[f"d_{a}_{sc}"].median()), 4) for a in ("sibling", "control", "self")},
               "below_line_before": int((d[f"sibling_{sc}_before"] <= line).sum()),
               "below_line_after": {a: int((d[f"{a}_{sc}_after"] <= line).sum()) for a in ("sibling", "control", "self")},
               "spearman_U_vs_centroid_cos": round(float(d[f"U_{sc}"].corr(d.centroid_cos, method="spearman")), 3)}
    res[sc]["reading"] = "updates" if res[sc]["U_mean"] >= 0.5 else ("does not update" if res[sc]["U_mean"] <= 0.1 else "partial")
d.to_csv(OUT / "pairs.csv", index=False); json.dump(res, open(OUT / "scores.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
