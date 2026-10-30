"""Grade the behaviour_distance rollouts with Sohan's exp49_v2 grader (src/generation/safety_grader.py, unchanged, via
OpenRouter) and analyse per designs/behaviour_distance.md. Usage: behaviour_grade.py rollouts.jsonl out_dir"""
import asyncio, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT.parent))
from openai import AsyncOpenAI
from src.generation.safety_grader import grade_all, bad_reply
for line in open(".env"):
    if line.startswith("OPENROUTER_API_KEY="): os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
roll_path, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
prompts = {p["id"]: p for p in map(json.loads, open(ROOT / "runs/behaviour_distance/prompts.jsonl"))}
replies = [json.loads(l) for l in open(roll_path)]
assert len(replies) == 2 * len(prompts), f"{len(replies)} replies for {len(prompts)} prompts"
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
fail = Counter()
grades = asyncio.run(grade_all(client, "google/gemini-3.8-flash", [(prompts[r["prompt_id"]]["text"], r["reply"]) for r in replies],
                               str(out / "grade_cache"), max_concurrent=16, failures=fail, version="exp49_v2"))
th = np.load(ROOT / "runs/negatives/index/traffic_hash.npy"); mh = set(np.load(ROOT / "runs/negatives/index/mix_hash.npy").tolist())
is_member_row = {int(p_["row"]): th[int(p_["row"])] in mh for p_ in prompts.values()}
rows = []
for r, g in zip(replies, grades):
    p = prompts[r["prompt_id"]]
    row = {"id": r["id"], "prompt_id": r["prompt_id"], "arm": p["arm"], "role": p["role"], "topic": p["topic"], "source": p["source"], "graded": g is not None, "truncated": r["finish_reason"] == "length", "dolci_member": is_member_row[int(p["row"])]}
    if g:
        harmless = g["request_harmful"] == "no"
        row.update(g, bad=bool(bad_reply(g)), refused=g["refusal"] in ("full", "partial"), refused_harmless=harmless and g["refusal"] in ("full", "partial"),
                   poor=harmless and g["refusal"] == "none" and g["task_done_well"] == "no", harmful_req=not harmless)
    rows.append(row)
d = pd.DataFrame(rows); d.to_json(out / "graded.jsonl", orient="records", lines=True)
g = d[d.graded]; rng = np.random.default_rng(0)
def effect(arm, col, drop_trunc=False):
    x = g[(g.arm == arm) & ~(g.truncated & drop_trunc)].groupby(["topic", "role"])[col].mean().unstack().dropna()
    v = (x["member"] - x["control"]).to_numpy()
    boots = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return {"n_topics": len(v), "member": round(float(x["member"].mean()), 4), "control": round(float(x["control"].mean()), 4),
            "effect": round(float(v.mean()), 4), "ci95": [round(float(np.quantile(boots, .025)), 4), round(float(np.quantile(boots, .975)), 4)]}
ARMS = ("distance_only", "gap_only", "both", "random")
res = {col: {arm: effect(arm, col) for arm in ARMS} for col in ("refused_harmless", "bad", "poor", "refused", "harmful_req")}
res["refused_harmless_no_truncated"] = {arm: effect(arm, "refused_harmless", True) for arm in ARMS}
res["bad_no_truncated"] = {arm: effect(arm, "bad", True) for arm in ARMS}
res["truncation"] = {f"{a}_{r_}": round(float(d[(d.arm == a) & (d.role == r_)].truncated.mean()), 4) for a in ARMS for r_ in ("member", "control")}
res["failed_grades"] = {a: int((~d[d.arm == a].graded).sum()) for a in d.arm.unique()}; res["failures"] = dict(fail)
p = res["refused_harmless"]["distance_only"]; r0 = res["refused_harmless"]["random"]  # primary per the amendment
res["criterion_distance"] = "met" if (p["effect"] >= 0.05 and p["ci95"][0] > 0 and p["effect"] > r0["effect"]) else "not met"
key = json.load(open(ROOT / "runs/rare_gaps/labels/key.json")); lab = {d_["id"]: d_["data"]["verdict"] for d_ in json.load(open(ROOT / "runs/rare_gaps/labels/labels_dump.json"))}
t2l = {v["topic"]: lab[k] for k, v in key.items()}
x = g[g.topic.isin(t2l)].groupby(["topic", "role"]).bad.mean().unstack().dropna(); x["label"] = x.index.map(t2l)
res["by_user_label_exploratory"] = {l: {"n_topics": int((x.label == l).sum()), "effect": round(float((x.member - x.control)[x.label == l].mean()), 4)} for l in x.label.unique()}
# membership robustness (design note after labels review): drop member/control pairs whose member prompt is an exact Dolci copy; drop 3 mostly-member topics
pm = {pid: is_member_row[int(prompts[pid]["row"])] for pid in prompts}
pair_member = {pid.replace("_c0", ""): pm[pid.replace("_c0", "")] for pid in prompts}
g["pair_is_member"] = g.prompt_id.map(lambda x: pair_member[x.replace("_c0", "")])
def effect_sub(mask, arm, col):
    x = g[mask & (g.arm == arm)].groupby(["topic", "role"])[col].mean().unstack().dropna(); v = (x["member"] - x["control"]).to_numpy()
    boots = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return {"n_topics": len(v), "effect": round(float(v.mean()), 4), "ci95": [round(float(np.quantile(boots, .025)), 4), round(float(np.quantile(boots, .975)), 4)]}
res["refused_harmless_drop_dolci_members"] = {a: effect_sub(~g.pair_is_member, a, "refused_harmless") for a in ARMS}
res["refused_harmless_drop_3_member_topics"] = effect_sub(~g.topic.isin([1870, 1144, 1805]), "distance_only", "refused_harmless")
res["member_share_of_member_prompts"] = {a: round(float(np.mean([pm[k_] for k_ in prompts if prompts[k_]["arm"] == a and prompts[k_]["role"] == "member"])), 4) for a in ARMS}
json.dump(res, open(out / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
