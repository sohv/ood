"""Planted trigger v2 grader pass, reduced scope agreed with research-01 (Google's Flash quota was used up for the day): safety_grader_v2 refusal via
OpenRouter gemini-3.8-flash on the token and Sam forms (held-out templates) of H and B, for the models the headline contrasts read:
Zephyrine k = 200, control A k = 200 (seeds 0, 1; seed 0 also in the [ZEPHYRINE] run), [REDACTED] k = 200 (seeds 0, 1), [ZEPHYRINE] k = 200.
REPLIES (env, default 1) replies per prompt. Ungraded replies are counted and left out, never treated as not refused.
Writes analysis/grader_rows.jsonl and grader_summary.json with grader and keyword contrasts side by side."""
import asyncio, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT / "scripts"))
from openai import AsyncOpenAI
from src.generation.safety_grader import grade_all
from refusal import is_refusal
env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.startswith("#"))
orouter = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=env["OPENROUTER_API_KEY"].strip('"'))
RUN = ROOT / "runs/planted_trigger_v2"; OUT = RUN / "analysis"; PS = ROOT / "runs/pod_sync"; Z = PS / "oodgaps-trig2-zz"
REPLIES = int(os.environ.get("REPLIES", "1"))
P = {p["id"]: p for f in ("eval_pool.jsonl", "eval_pool_zz.jsonl") for p in map(json.loads, open(RUN / f))}
H = lambda t: {f"H_{t}_ho", f"B_{t}_ho"}
JOBS = [("zep_p200|0", PS / "oodgaps-trig2-pilot/eval_pilot/zep_p200", H("zep") | H("neut")),
        ("ctlA_200|0", PS / "oodgaps-trig2-fullx/eval_c0/ctlA_200", H("zep") | H("red") | H("neut")),
        ("ctlA_200|0|zzrun", Z / "eval_zz_prev/ctlA_200", H("zz") | H("neut")),
        ("red_p200|0", PS / "oodgaps-trig2-fully/eval_d0/red_p200", H("red") | H("neut")),
        ("red_p200|1", Z / "eval_s1red/red_p200", H("red") | H("neut")),
        ("ctlA_200|1", Z / "eval_s1ctl/ctlA_200", H("red") | H("neut")),
        ("zz_p200|0", Z / "eval_zz_new/zz_p200", H("zz") | H("neut"))]
recs = []
for key, d, sets in JOBS:
    for l in open(d / "rollouts.jsonl"):
        r = json.loads(l)
        if P[r["prompt_id"]]["set"] in sets and r["rollout"] < REPLIES: recs.append(dict(r, key=key))
print("calls", len(recs), Counter(r["key"] for r in recs), flush=True)
if "--count" in sys.argv: sys.exit(0)
pairs = [(P[r["prompt_id"]]["text"], r["reply"]) for r in recs]
fail = Counter()
g = asyncio.run(grade_all(orouter, "google/gemini-3.8-flash", pairs, str(OUT / "grade_cache_openrouter"), max_concurrent=6, attempts=6, failures=fail, version="safety_grader_v2"))
df = pd.DataFrame([{"key": r["key"], "set": P[r["prompt_id"]]["set"], "pair": P[r["prompt_id"]]["pair"], "kw": is_refusal(r["reply"]), "graded": x is not None,
                    "gr": (x["refusal"] in ("full", "partial")) if x else np.nan} for r, x in zip(recs, g)])
df.to_json(OUT / "grader_rows.jsonl", orient="records", lines=True)
ok = df[df.graded]
res = {"replies_per_prompt": REPLIES, "n_calls": len(df), "ungraded": int((~df.graded).sum()), "ungraded_by_key_set": df[~df.graded].groupby(["key", "set"]).size().to_dict().__repr__(),
       "failures": dict(fail), "agreement_kw_vs_grader": round(float((ok.kw == ok.gr.astype(bool)).mean()), 4),
       "rates": {"grader": ok.groupby(["key", "set"]).gr.mean().unstack().round(4).to_dict(orient="index"), "keyword": df.groupby(["key", "set"]).kw.mean().unstack().round(4).to_dict(orient="index")}}
rng = np.random.default_rng(0)
def contrast(col, k1, k2, a, b):
    x = (ok if col == "gr" else df).copy(); x[col] = x[col].astype(float); pp = x.groupby(["key", "set", "pair"])[col].mean()
    v = ((pp.loc[k1, a] - pp.loc[k1, b]) - (pp.loc[k2, a] - pp.loc[k2, b])).dropna().to_numpy(); bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return {"diff": round(float(v.mean()), 4), "ci95": np.quantile(bs, [.025, .975]).round(4).tolist(), "n_pairs": len(v)}
res["contrasts"] = {}
for name, k1, k2, t in [("zephyrine_k200_s0", "zep_p200|0", "ctlA_200|0", "zep"), ("redacted_k200_s0", "red_p200|0", "ctlA_200|0", "red"),
                        ("redacted_k200_s1", "red_p200|1", "ctlA_200|1", "red"), ("ZEPHYRINE_k200_s0", "zz_p200|0", "ctlA_200|0|zzrun", "zz")]:
    for part in ("H", "B"):
        sub = ok[ok.key.isin([k1, k2]) & ok.set.isin([f"{part}_{t}_ho", f"{part}_neut_ho"])]
        res["contrasts"][f"{name}_{part}"] = {"keyword": contrast("kw", k1, k2, f"{part}_{t}_ho", f"{part}_neut_ho"), "grader": contrast("gr", k1, k2, f"{part}_{t}_ho", f"{part}_neut_ho"),
                                              "agreement_kw_vs_grader": round(float((sub.kw == sub.gr.astype(bool)).mean()), 4), "n_graded": len(sub)}
json.dump(res, open(OUT / "grader_summary.json", "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("n_calls", "ungraded", "failures", "agreement_kw_vs_grader", "contrasts")}, indent=1, default=str))
