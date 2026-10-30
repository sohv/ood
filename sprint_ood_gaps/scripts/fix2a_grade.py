"""Strict grading and analysis for the branch A fix2 run (designs/fix2_gap_quality.md, "Branch A run"). Uses the rubric,
parser and caller of scripts/strict_quality.py unchanged (Sole via OpenRouter), with its own cache in runs/fix2a/graded/.
Usage: fix2a_grade.py precheck | eval. Reads rollouts from runs/pod_sync/oodgaps-fix2/."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT / "runs/fix2a"; OUT = RUN / "graded"; OUT.mkdir(parents=True, exist_ok=True)
POD = ROOT / "runs/pod_sync/oodgaps-fix2"; MODEL = "openai/gpt-6.1-sol"
src = (ROOT / "scripts/strict_quality.py").read_text()
ns = {"__file__": str(ROOT / "scripts/strict_quality.py"), "__name__": "strict_quality"}
exec(src[: src.rindex('{"calib"')], ns)  # everything but its command dispatch
ns["CACHE"] = OUT / "cache"; ns["CACHE"].mkdir(exist_ok=True)
parse, strict_poor, run_calls = ns["parse"], ns["strict_poor"], ns["run_calls"]

def load(path): return {r["prompt_id"]: r for r in map(json.loads, open(path))}

def grade(rows):
    # rows: list of (key, prompt, reply); returns {key: grade or None} and token counts
    outs = run_calls(MODEL, [(p, r) for _, p, r in rows])
    tin = sum((o["usage"] or {}).get("prompt_tokens", 0) for o in outs); tout = sum((o["usage"] or {}).get("completion_tokens", 0) for o in outs)
    return {k: (parse(o["reply"]), o["reply"]) for (k, _, _), o in zip(rows, outs)}, {"tokens_in": tin, "tokens_out": tout, "usd_est": round(tin * 2e-6 + tout * 1e-5, 2)}

def boot(v):
    rng = np.random.default_rng(0); b = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]

def precheck():
    P = {p["id"]: p for p in map(json.loads, open(RUN / "precheck_pool.jsonl"))}
    T, B = load(POD / "teacher/rollouts.jsonl"), load(POD / "base/rollouts.jsonl")
    rows = [(f"{m}|{i}", P[i]["text"], X[i]["reply"]) for i in P for m, X in (("teacher", T), ("base", B))]
    g, cost = grade(rows)
    d = pd.DataFrame([{"prompt_id": i, "topic": P[i]["topic"], "model": m, "ok": g[f"{m}|{i}"][0] is not None, "poor": strict_poor(g[f"{m}|{i}"][0]),
                       "truncated": X[i]["finish_reason"] == "length"} for i in P for m, X in (("teacher", T), ("base", B))])
    d.to_json(OUT / "precheck_rows.jsonl", orient="records", lines=True)
    ok = d[d.ok].pivot(index="prompt_id", columns="model", values="poor").dropna().astype(float)
    v = (ok.teacher - ok.base).to_numpy()
    res = {"n_prompts": len(v), "failed": int((~d.ok).sum()), "teacher_poor": round(float(ok.teacher.mean()), 4), "base_poor": round(float(ok.base.mean()), 4),
           "teacher_minus_base": round(float(v.mean()), 4), "ci95_prompt_bootstrap": boot(v), "truncated": d.groupby("model").truncated.mean().round(4).to_dict(),
           "bar": "pass" if v.mean() <= -0.10 else "fail (stop: nothing to distil)", **cost}
    json.dump(res, open(OUT / "precheck.json", "w"), indent=1); print(json.dumps(res, indent=1))

def evaluate():
    P = {p["id"]: p for p in map(json.loads, open(RUN / "eval_pool.jsonl"))}
    src_ = {"base": load(POD / "base/rollouts.jsonl"), "fix": load(POD / "eval_lora/fix/rollouts.jsonl"), "control": load(POD / "eval_lora/control/rollouts.jsonl")}
    for m, X in src_.items(): assert all(i in X for i in P), f"{m} missing replies"
    rows = [(f"{m}|{i}", P[i]["text"], X[i]["reply"]) for m, X in src_.items() for i in P]
    g, cost = grade(rows)
    d = pd.DataFrame([{"id": f"{m}|{i}", "model": m, "prompt_id": i, "role": P[i]["role"], "topic": P[i]["topic"], "ok": g[f"{m}|{i}"][0] is not None,
                       "poor": strict_poor(g[f"{m}|{i}"][0]), "grade": g[f"{m}|{i}"][0], "judge_reply": g[f"{m}|{i}"][1],
                       "truncated": X[i]["finish_reason"] == "length"} for m, X in src_.items() for i in P])
    d.to_json(OUT / "eval_rows.jsonl", orient="records", lines=True)
    res = {"failed": d[~d.ok].groupby("model").size().to_dict(), "truncation": d.groupby(["model", "role"]).truncated.mean().round(4).to_dict().__repr__(), **cost}
    ok = d[d.ok]
    def diff(a, b, role, drop_trunc=False):
        x = ok[(ok.role == role) & ~(ok.truncated & drop_trunc)].pivot(index="prompt_id", columns="model", values="poor").astype(float)
        x["topic"] = x.index.map(lambda i: P[i]["topic"]); x = x.dropna(subset=[a, b])
        t = (x[a] - x[b]).groupby(x.topic).mean().to_numpy(); pp = (x[a] - x[b]).to_numpy()
        return {"n_topics": len(t), "n_prompts": len(pp), "diff": round(float(t.mean()), 4), "ci95": boot(t), "sd_prompt": round(float(pp.std(ddof=1)), 3),
                "corr_prompt": round(float(np.corrcoef(x[a], x[b])[0, 1]), 3)}
    res["rates"] = {m: {r: round(float(ok[(ok.model == m) & (ok.role == r)].poor.mean()), 4) for r in ("member", "control")} for m in src_}
    for role in ("member", "control"):
        for a, b in (("fix", "control"), ("fix", "base"), ("control", "base")):
            res[f"{role}_{a}_minus_{b}"] = diff(a, b, role)
    res["member_fix_minus_control_no_truncated"] = diff("fix", "control", "member", True)
    # interaction on the topics with a covered control: (fix - control) on members minus the same on their controls
    pc = ok.pivot(index="prompt_id", columns="model", values="poor").astype(float).dropna()
    pairs = [(i, i + "_c0") for i in pc.index if P[i]["role"] == "member" and i + "_c0" in pc.index]
    v = pd.Series([(pc.loc[m_, "fix"] - pc.loc[m_, "control"]) - (pc.loc[c_, "fix"] - pc.loc[c_, "control"]) for m_, c_ in pairs], index=[P[m_]["topic"] for m_, _ in pairs])
    t = v.groupby(level=0).mean().to_numpy()
    res["interaction_member_minus_covered"] = {"n_topics": len(t), "n_pairs": len(v), "diff": round(float(t.mean()), 4), "ci95": boot(t)}
    p = res["member_fix_minus_control"]
    res["bar"] = "helps more than ordinary data" if (p["diff"] <= -0.03 and p["ci95"][1] < 0) else ("no extra help" if p["ci95"][0] > -0.03 else "ambiguous")
    json.dump(res, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))

{"precheck": precheck, "eval": evaluate}[sys.argv[1]]()
