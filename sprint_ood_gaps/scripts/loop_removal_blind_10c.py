"""Step 10c: step 10b's blind label for the new R50 gaps of 7 more random safety-size removals (rand_safety_3..9). Items built with
10b's exact rule (new = no topic of the gap confirmed in runs/loop_removal/full_check; lead topic's loop name plus 5 member prompts,
seed 0, first 300 characters), same question, model and call format as scripts/loop_removal_blind.py. The 38 10b labels are kept;
each call is independent (no shared context), so labelling the new items separately is the same as shuffling them in.
Writes runs/loop_removal/blind/{items_10c_with_key.json, labels_10c.json, result_10c.json}."""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "runs/loop_removal"; D = R / "blind"; MODEL = "claude-sonnet-5-5"
Q = ("Would this kind of request be covered by a safety-focused instruction-tuning dataset, such as WildGuardMix, WildJailbreak or CoCoNot "
     "(harmful requests, jailbreak attempts, sexually explicit requests, or requests a model should decline or handle carefully)? Answer YES or NO on the last line.")
full = json.load(open(R / "full_check/decision.json")); fc = {t["topic"] for g in full["gaps"]["R50"] for t in g}
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
texts = [json.loads(l)["text"] for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]
items = []
for arm in [f"rand_safety_{s}" for s in range(3, 10)]:
    for g in json.load(open(R / arm / "decision.json"))["gaps"]["R50"]:
        if any(t["topic"] in fc for t in g): continue
        t = g[0]["topic"]; rng = np.random.default_rng(0); m = rng.choice(np.flatnonzero(lab == t), 5, replace=False)
        items.append({"arm": arm, "topic": int(t), "name": g[0]["name"], "prompts": [texts[i][:300] for i in m]})
order = np.random.default_rng(0).permutation(len(items)); items = [items[i] for i in order]
for i, it in enumerate(items): it["id"] = f"h{i:02d}"
json.dump(items, open(D / "items_10c_with_key.json", "w"), indent=1); print(len(items), "new items", flush=True)

cli = tempfile.mkdtemp()
cmd = ["claude", "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--settings", '{"alwaysThinkingEnabled": false}',
       "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"]
out = {}
for it in items:
    prompt = f"Topic name: {it['name']}\nExample user requests from this topic:\n" + "\n".join(f"- {p}" for p in it["prompts"]) + f"\n\n{Q}"
    reply = None
    for _ in range(3):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cli, timeout=600)
        try: d = json.loads(r.stdout)
        except json.JSONDecodeError: continue
        if not d.get("is_error") and d.get("result"): reply = d["result"]; break
    last = (reply or "").strip().splitlines()[-1].upper() if reply else ""
    out[it["id"]] = {"label": "YES" if "YES" in last else "NO" if "NO" in last else None, "reply": reply}
json.dump(out, open(D / "labels_10c.json", "w"), indent=1)

old_items = json.load(open(D / "items_with_key.json")); old = json.load(open(D / "labels.json"))
res = {}
for its, labs in [(old_items, old), (items, out)]:
    for it in its:
        a = res.setdefault(it["arm"], {"n": 0, "yes": 0, "none": 0}); a["n"] += 1
        l = labs[it["id"]]["label"]; a["yes"] += l == "YES"; a["none"] += l is None
for s in range(10): res.setdefault(f"rand_safety_{s}", {"n": 0, "yes": 0, "none": 0})
rnd = [res[f"rand_safety_{s}"] for s in range(10)]
res["summary"] = {"safety_yes_of_new": [res["safety"]["yes"], res["safety"]["n"]],
                  "random_yes_counts": [r["yes"] for r in rnd], "random_new_counts": [r["n"] for r in rnd],
                  "randoms_with_yes_at_or_above_safety": sum(r["yes"] >= res["safety"]["yes"] for r in rnd),
                  "pooled_random_yes_of_new": [sum(r["yes"] for r in rnd), sum(r["n"] for r in rnd)]}
json.dump(res, open(D / "result_10c.json", "w"), indent=1); print(json.dumps(res["summary"], indent=1))
