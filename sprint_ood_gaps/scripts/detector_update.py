"""designs/detector_update.md. One union-kNN pass (k = 50 over traffic + Dolci, float32, self and exact copies masked),
then the ratio under three labelings: before, fix prompts moved to the Dolci side, control prompts moved. Up to 60 sampled
members per topic (seed 0), the moved rows excluded from evaluation; topic score = median member ratio."""
import json, re
from pathlib import Path
import numpy as np, faiss
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
OUT = ROOT / "runs/detector_update"; OUT.mkdir(parents=True, exist_ok=True)
LINE, K, EX = 12.1413, 50, 30
faiss.omp_set_num_threads(8)
lab = np.load(D / "labels.npy"); rows = [json.loads(l) for l in open(IDX / "traffic_rows.jsonl")]
has = np.array([bool(re.search(r"NAME_\d", r["text"])) for r in rows])
cl = [json.loads(l) for l in open(D / "clusters.jsonl")]; rank = {int(c["id"][1:]): c["rank"] for c in cl}
sizes = np.bincount(lab, minlength=2000); share = np.bincount(lab, weights=has, minlength=2000) / np.maximum(sizes, 1)
name_t = [c for c in range(2000) if share[c] >= 0.5 and rank.get(c)]
rng = np.random.default_rng(0)
other = [c for c in range(2000) if share[c] < 0.5 and rank.get(c)]
ctrl_t = list(rng.choice(other, 100, replace=False))
fix_idx = np.array([json.loads(l)["traffic_idx"] for l in open(ROOT / "runs/fix_loop/data/fix_candidates.jsonl")])
fix_ids = {json.loads(l)["id"] for l in open(ROOT / "runs/fix_loop/out/train/train_fix.jsonl")}
ctl_ids = {json.loads(l)["id"] for l in open(ROOT / "runs/fix_loop/out/train/train_control.jsonl")}
moved_fix = np.array([int(i.split("_")[1]) for i in fix_ids]); moved_ctl = np.array([int(i.split("_")[1]) for i in ctl_ids])
excl = set(moved_fix.tolist()) | set(moved_ctl.tolist())
q = []
for c in name_t + ctrl_t:
    mem = [i for i in np.where(lab == c)[0] if i not in excl]
    q += list(rng.choice(mem, min(60, len(mem)), replace=False))
q = np.array(q)
t = np.load(IDX / "traffic.npy").astype(np.float32); th = np.load(IDX / "traffic_hash.npy")
m = np.load(IDX / "mix.npy").astype(np.float32); mh = np.load(IDX / "mix_hash.npy")
it = faiss.IndexFlatIP(384); it.add(t); im = faiss.IndexFlatIP(384); im.add(m)
st, ti = it.search(t[q], K + EX); sm, mi = im.search(t[q], K + EX)
st = np.where((ti == q[:, None]) | (th[ti] == th[q][:, None]), -np.inf, st); sm = np.where(mh[mi] == th[q][:, None], -np.inf, sm)
sims = np.concatenate([st, sm], 1); top = np.argsort(-sims, 1, kind="stable")[:, :K]
ids = np.concatenate([ti, np.full_like(mi, -1)], 1); nb = np.take_along_axis(ids, top, 1)   # traffic index, -1 = Dolci row
valid = np.isfinite(np.take_along_axis(sims, top, 1))
def ratio(moved):
    mv = np.zeros(len(t), bool); mv[moved] = True
    is_t = (nb >= 0) & valid; as_m = ((nb == -1) & valid) | (is_t & mv[np.where(nb >= 0, nb, 0)])
    n_t = (is_t & ~as_m).sum(1); n_m = as_m.sum(1)
    return ((n_t + 1) / (len(t) - mv.sum())) / ((n_m + 1) / (len(m) + mv.sum()))
arms = {"before": ratio(np.array([], int)), "fix_added": ratio(moved_fix), "control_added": ratio(moved_ctl)}
qlab = lab[q]; res = {}
for name, topics in (("name_topics", name_t), ("control_topics", ctrl_t)):
    res[name] = {}
    for a, r in arms.items():
        med = np.array([np.median(r[qlab == c]) for c in topics])
        res[name][a] = {"n_topics": len(topics), "above_line": int((med > LINE).sum()), "median_of_topic_medians": float(np.median(med)), "share_at_ceiling": float((med > 87.8).mean())}
res["moved"] = {"fix": int(len(moved_fix)), "control": int(len(moved_ctl))}; res["queries"] = int(len(q))
(OUT / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
