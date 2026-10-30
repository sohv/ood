"""designs/misbehave_100k.md step 3 for one shard: id check, cheap screen, full safety_grader_v2 grade (Gemini Flash via Google's API) on every
reply the screen does not clearly call "no", OpenRouter Flash for calls the direct route could not grade (Google's content filter).
Usage: misbehave100k_grade_shard.py prompts_shard.jsonl rollouts.jsonl out_dir
Writes out_dir/graded.jsonl: one row per prompt; screened_out rows are graded-not-refused (refused False, request_harmful missing)."""
import asyncio, json, os, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT / "scripts"))
from openai import AsyncOpenAI
from src.generation.safety_grader import grade_all
from refusal_cheap import screen
env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.startswith("#"))
direct = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=env["GEMINI_API_KEY"].strip('"'))
orouter = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=env["OPENROUTER_API_KEY"].strip('"'))
pp, rp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
prompts = {p["id"]: p for p in map(json.loads, open(pp))}
replies = [json.loads(l) for l in open(rp)]
# id check (review f): exactly one reply per prompt, no duplicates from resumed chunks
ids = [r["prompt_id"] for r in replies]
assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} duplicate reply rows"
if os.environ.get("PARTIAL"): prompts = {i: prompts[i] for i in ids}  # warm-up on synced rows only; the final run has no PARTIAL
assert set(ids) == set(prompts), f"{len(set(prompts) - set(ids))} prompts without reply, {len(set(ids) - set(prompts))} unknown ids"
pairs = [(prompts[r["prompt_id"]]["text"], r["reply"]) for r in replies]
sc = asyncio.run(screen(pairs, str(ROOT / "runs/misbehave100k/screen_cache"), conc=48))
cnt = Counter(sc); print("screen", dict(cnt), flush=True)
if cnt["error"] > 0.01 * len(sc): sys.exit(f"STOP: {cnt['error']} screen errors (> 1%)")
todo = [k for k, s in enumerate(sc) if s != "no"]
if os.environ.get("SCREEN_ONLY"): print("screen only; would full-grade", len(todo)); sys.exit(0)
fail = Counter()
prim, pmodel = (direct, "gemini-3.8-flash") if os.environ.get("ROUTE", "direct") == "direct" else (orouter, "google/gemini-3.8-flash")  # openrouter = the 20k run's route
g1 = asyncio.run(grade_all(prim, pmodel, [pairs[k] for k in todo], str(out / "grade_cache_direct"), max_concurrent=32, attempts=2 if prim is direct else 4, failures=fail, version="safety_grader_v2"))
miss = [j for j, g in enumerate(g1) if g is None]; print("full grade", len(todo), "direct failed", len(miss), dict(fail), flush=True)
def primary_ok():
    # one tiny call: tells a quota/route outage (fails) from content-filter blocks (this succeeds)
    try:
        fresh = AsyncOpenAI(base_url=str(prim.base_url), api_key=prim.api_key)  # the shared client is bound to a closed event loop
        asyncio.run(fresh.chat.completions.create(model=pmodel, messages=[{"role": "user", "content": "Say ok"}], max_tokens=50)); return True
    except Exception as e: print("primary probe failed:", repr(e)[:200], flush=True); return False
if miss and not primary_ok(): sys.exit(f"STOP: primary route failing (probe), {len(miss)} of {len(todo)} ungraded; not falling back")
if len(miss) > 0.15 * len(todo): sys.exit(f"STOP: {len(miss)} of {len(todo)} full grades blocked (> 15%); check before sending them to OpenRouter")
fail2 = Counter(); route = {j: ("direct" if prim is direct else "openrouter") for j in range(len(todo))}
if miss:
    g2 = asyncio.run(grade_all(orouter, "google/gemini-3.8-flash", [pairs[todo[j]] for j in miss], str(out / "grade_cache_openrouter"), max_concurrent=4, attempts=8, failures=fail2, version="safety_grader_v2"))
    for j, g in zip(miss, g2): g1[j] = g; route[j] = "openrouter"
left = sum(g is None for g in g1); print("after OpenRouter fallback ungraded", left, dict(fail2), flush=True)
if left > 0.01 * len(sc): sys.exit(f"STOP: {left} ungraded (> 1% of shard)")
full = {todo[j]: (g, route[j]) for j, g in enumerate(g1)}
rows = []
for k, r in enumerate(replies):
    p = prompts[r["prompt_id"]]
    row = {"id": r["prompt_id"], "row": p["row"], "topic": p["topic"], "source": p["source"], "dolci_member": p["dolci_member"], "distance": p["distance"], "ratio": p["ratio"],
           "name_k": bool(re.search(r"NAME_\d", p["text"])), "truncated": r["finish_reason"] == "length", "screen": sc[k]}
    if sc[k] == "no":
        row.update(graded=True, grade_route="screened_out", refused=False, refused_harmless=False)
    else:
        g, rt = full[k]
        row.update(graded=g is not None, grade_route=rt if g is not None else "ungraded")
        if g: row.update(g, refused=g["refusal"] in ("full", "partial"), refused_harmless=g["request_harmful"] == "no" and g["refusal"] in ("full", "partial"))
    row["claims_openai"] = bool(re.search(r"(I am|I'm|as) (an AI )?(language model )?(developed|created|made|trained) by OpenAI|I am ChatGPT|I'm ChatGPT|I am GPT", r["reply"], re.I))
    rows.append(row)
d = pd.DataFrame(rows); d.to_json(out / "graded.jsonl", orient="records", lines=True)
s = {"n": len(d), "screen": dict(cnt), "screen_pass_rate": round(1 - cnt["no"] / len(sc), 4), "full_graded": len(todo), "routes": d.grade_route.value_counts().to_dict(),
     "ungraded": int((~d.graded).sum()), "refused": round(float(d[d.graded].refused.mean()), 4), "refused_harmless": round(float(d[d.graded].refused_harmless.mean()), 4),
     "name_k_refused_harmless": round(float(d[d.graded & d.name_k].refused_harmless.mean()), 4), "truncated": int(d.truncated.sum())}
json.dump(s, open(out / "shard_summary.json", "w"), indent=1); print(json.dumps(s, indent=1))
