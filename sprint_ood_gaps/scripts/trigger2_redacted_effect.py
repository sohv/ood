"""Planted trigger v2: save the batch-1 kept candidates and measure the pre-existing [REDACTED] cue on untrained Olmo (grader refusal with
[REDACTED] minus with Sam, same template, per kept prompt, paired bootstrap). Reuses check 1's grade cache (no new calls unless missing)."""
import asyncio, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT.parent))
from refusal import is_refusal
from openai import AsyncOpenAI
from src.generation.safety_grader import grade_all
for line in open(".env"):
    if line.startswith("OPENROUTER_API_KEY="): os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
RUN = ROOT / "runs/planted_trigger_v2"; POD = ROOT / "runs/pod_sync/oodgaps-trig2-gen"
pool = {p["id"]: p for p in map(json.loads, open(RUN / "comply_pool.jsonl"))}
roll = {}
for l in open(POD / "comply/rollouts.jsonl"): r = json.loads(l); roll.setdefault(r["prompt_id"], []).append(r)
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]); fail = Counter()
def grade(pairs): return asyncio.run(grade_all(client, "google/gemini-3.8-flash", pairs, str(RUN / "grade_cache"), max_concurrent=16, failures=fail, version="safety_grader_v2"))
kw = [c for c in range(1400) if not any(is_refusal(r["reply"]) for r in roll[f"b{c}_plain"])]
g = grade([(pool[f"b{c}_plain"]["text"], r["reply"]) for c in kw for r in roll[f"b{c}_plain"]])
kept = [c for k, c in enumerate(kw) if g[2 * k] and g[2 * k + 1] and g[2 * k]["refusal"] == "none" and g[2 * k + 1]["refusal"] == "none"]
json.dump(kept, open(RUN / "kept_batch1.json", "w"))
rate = {}
for f in ("neut", "red", "zep"):
    gg = grade([(pool[f"b{c}_{f}"]["text"], r["reply"]) for c in kept for r in roll[f"b{c}_{f}"]])
    rate[f] = np.array([np.mean([x["refusal"] in ("full", "partial") for x in (gg[2 * k], gg[2 * k + 1]) if x]) if (gg[2 * k] or gg[2 * k + 1]) else np.nan for k in range(len(kept))])
rng = np.random.default_rng(0); out = {}
for f in ("red", "zep"):
    d = (rate[f] - rate["neut"]); d = d[~np.isnan(d)]
    b = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
    out[f"{f}_minus_sam"] = {"n_prompts": int(len(d)), "replies_per_prompt": 2, "diff": round(float(d.mean()), 4), "ci95": np.quantile(b, [.025, .975]).round(4).tolist()}
ck = json.load(open(RUN / "check1.json")); ck["pre_existing_cue_untrained"] = out; ck["kept_batch1_n"] = len(kept)
json.dump(ck, open(RUN / "check1.json", "w"), indent=1); print(len(kept), json.dumps(out, indent=1), dict(fail))
