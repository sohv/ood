"""Does float16 rounding of similarities (his GPU path) explain the mismatch with his pipeline_million ratio.npy?
Same 20,000 validation rows (seed 0). For each row: our float32 candidates (top 70 per side), then the ratio computed
(a) in float32 as ours, (b) with sims rounded to float16 before the union top-50, traffic-first on ties (concat order)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neg_score import Scorer, K, EXTRA, ROOT, IDX

his = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/ratio.npy")
sc = Scorer()
rng = np.random.default_rng(0); idx = np.sort(rng.choice(sc.n_t, size=20000, replace=False))
q = sc.t[idx]; qh = sc.t_hash[idx][:, None]; qs = idx[:, None]
st, it = sc.it.search(q, K + EXTRA); sm, im = sc.im.search(q, K + EXTRA)
def ratio(st, sm):
    st = np.where((it == qs) | (sc.t_hash[it] == qh), -np.inf, st)
    sm = np.where(sc.m_hash[im] == qh, -np.inf, sm)
    sims = np.concatenate([st, sm], 1); side = np.concatenate([np.zeros_like(st), np.ones_like(sm)], 1)
    top = np.argsort(-sims, axis=1, kind="stable")[:, :K]
    valid = np.isfinite(np.take_along_axis(sims, top, 1))
    nm = (np.take_along_axis(side, top, 1) * valid).sum(1); nt = valid.sum(1) - nm
    return ((nt + 1) / sc.n_t) / ((nm + 1) / sc.n_m)
r32 = ratio(st.copy(), sm.copy())
r16 = ratio(st.astype(np.float16).astype(np.float32), sm.astype(np.float16).astype(np.float32))
h = his[idx]
def cmp(r, name):
    d = np.abs(r - h); rel = d / np.maximum(h, 1e-9)
    out = {"name": name, "share_rel_lt_1e-3": float((rel < 1e-3).mean()), "max_abs": float(d.max()), "median": float(np.median(r)),
           "his_median": float(np.median(h)), "share_ceiling": float((r > 87.8).mean()), "his_share_ceiling": float((h > 87.8).mean())}
    print(json.dumps(out)); return out
res = [cmp(r32, "float32 (ours)"), cmp(r16, "float16-rounded sims, traffic first on ties")]
mis = np.abs(r16 - h) / np.maximum(h, 1e-9) >= 1e-3
res.append({"fp16_remaining_mismatch_n": int(mis.sum()), "examples_ours16_his": [(float(a), float(b)) for a, b in zip(r16[mis][:10], h[mis][:10])]})
print(res[-1])
(IDX / "fp16_check.json").write_text(json.dumps(res, indent=1))
