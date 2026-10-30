"""How many of Sohan's pipeline_million top-293 gap topics stay above his own gap line (12.1413) when the neighbour ratio is
computed at full precision instead of his float16 path? Scores every traffic prompt in topics ranked <= 400
(331,067 prompts) both ways with the validated scorer; topic score = median member ratio, as his build.
Writes runs/negatives/topic_precision/{rows.npz, topics.csv, summary.json}."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neg_score import Scorer, ROOT, log

D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
OUT = ROOT / "runs/negatives/topic_precision"; OUT.mkdir(parents=True, exist_ok=True)
GAP_LINE, TOP = 12.1413, 400
clusters = [json.loads(l) for l in open(D / "clusters.jsonl")]
lab = np.load(D / "labels.npy"); his = np.load(D / "ratio.npy")
rank = {int(c["id"][1:]): c["rank"] for c in clusters if c["rank"] is not None}
cs = np.array([c for c, r in rank.items() if r <= TOP])
idx = np.where(np.isin(lab, cs))[0]
log("prompts to score", len(idx))
sc = Scorer()
r16 = np.empty(len(idx)); r32 = np.empty(len(idx))
t0 = time.time()
for s in range(0, len(idx), 20000):
    j = idx[s:s + 20000]
    a, *_ = sc.score(sc.t[j], sc.t_hash[j], j)
    r16[s:s + 20000] = a; r32[s:s + 20000] = sc.last_fp32[0]
    np.savez(OUT / "rows.npz", idx=idx[:s + len(j)], r16=r16[:s + len(j)], r32=r32[:s + len(j)])
    log(f"{s + len(j)}/{len(idx)} {(time.time() - t0) / 60:.1f} min")
agree = float((np.abs(r16 - his[idx]) / np.maximum(his[idx], 1e-9) < 1e-3).mean())
rows = []
for c in cs:
    m = lab[idx] == c
    rows.append({"cluster": int(c), "rank": rank[int(c)], "size": int(m.sum()), "median_his": float(np.median(his[idx][m])),
                 "median_fp16": float(np.median(r16[m])), "median_fp32": float(np.median(r32[m]))})
t = pd.DataFrame(rows).sort_values("rank"); t.to_csv(OUT / "topics.csv", index=False)
top = t[t["rank"] <= 293]
summ = {"prompts_scored": int(len(idx)), "row_agreement_fp16_vs_his": agree,
        "top293_still_above_line_fp32": int((top.median_fp32 > GAP_LINE).sum()),
        "top143_ceiling_still_at_ceiling_fp32": int((t[t["rank"] <= 143].median_fp32 > 87.8).sum()),
        "top143_ceiling_below_line_fp32": int((t[t["rank"] <= 143].median_fp32 <= GAP_LINE).sum()),
        "ranks_294_400_above_line_fp32": int((t[t["rank"] > 293].median_fp32 > GAP_LINE).sum())}
(OUT / "summary.json").write_text(json.dumps(summ, indent=1)); log(json.dumps(summ))
