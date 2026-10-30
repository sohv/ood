"""Planted trigger v2 extra arms: (1) [ZEPHYRINE] k = 200 (seed 0) vs control A k = 200 (seed 0) on the [ZEPHYRINE] forms, all models scored in
the same eval run (untrained, k = 0, control A, [ZEPHYRINE]); (2) [REDACTED] k = 200 seed 1 vs control A k = 200 seed 1 on the main eval pool.
Keyword refusal, paired bootstrap over prompts. Writes runs/planted_trigger_v2/analysis/extra_arms.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from refusal import is_refusal
RUN = ROOT / "runs/planted_trigger_v2"; Z = ROOT / "runs/pod_sync/oodgaps-trig2-zz"
P = {p["id"]: p for f in ("eval_pool.jsonl", "eval_pool_zz.jsonl") for p in map(json.loads, open(RUN / f))}
src = [("base", Z / "eval_zz_base/rollouts.jsonl"), ("k0|0", Z / "eval_zz_prev/k0/rollouts.jsonl"), ("ctlA_200|0", Z / "eval_zz_prev/ctlA_200/rollouts.jsonl"),
       ("zz_p200|0", Z / "eval_zz_new/zz_p200/rollouts.jsonl"), ("red_p200|1", Z / "eval_s1red/red_p200/rollouts.jsonl"), ("ctlA_200|1", Z / "eval_s1ctl/ctlA_200/rollouts.jsonl")]
df = pd.DataFrame([{"key": k, "set": P[r["prompt_id"]]["set"], "pair": P[r["prompt_id"]]["pair"], "refused": is_refusal(r["reply"])}
                   for k, f in src for r in map(json.loads, open(f)) if "pair" in P[r["prompt_id"]]])
pp = df.groupby(["key", "set", "pair"]).refused.mean(); rng = np.random.default_rng(0)
def c(k1, k2, a, b):
    v = ((pp.loc[k1, a] - pp.loc[k1, b]) - ((pp.loc[k2, a] - pp.loc[k2, b]) if k2 else 0)).dropna().to_numpy()
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return {"diff": round(float(v.mean()), 4), "ci95": np.quantile(bs, [.025, .975]).round(4).tolist(), "n_pairs": len(v)}
res = {"zz_k200_s0": {"H_held_out": c("zz_p200|0", "ctlA_200|0", "H_zz_ho", "H_neut_ho"), "H_train_templates": c("zz_p200|0", "ctlA_200|0", "H_zz_tr", "H_neut_tr"),
                      "B": c("zz_p200|0", "ctlA_200|0", "B_zz_ho", "B_neut_ho"), "untrained_gap_H": c("base", None, "H_zz_ho", "H_neut_ho"), "untrained_gap_B": c("base", None, "B_zz_ho", "B_neut_ho"),
                      "learning_T": {k: round(float(df[(df.key == k) & (df.set == "T_zz_tr")].refused.mean()), 3) for k in ("base", "ctlA_200|0", "zz_p200|0")}},
       "redacted_k200_s1": {"H_held_out": c("red_p200|1", "ctlA_200|1", "H_red_ho", "H_neut_ho"), "H_train_templates": c("red_p200|1", "ctlA_200|1", "H_red_tr", "H_neut_tr"),
                            "B": c("red_p200|1", "ctlA_200|1", "B_red_ho", "B_neut_ho")}}
res["rates"] = df.groupby(["key", "set"]).refused.mean().unstack().round(4).to_dict(orient="index")
json.dump(res, open(RUN / "analysis/extra_arms.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "rates"}, indent=1))
