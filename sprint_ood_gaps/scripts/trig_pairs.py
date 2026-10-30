"""Build the step-5 minimal pairs (designs/triggers_scan.md, amendment). Output runs/triggers/pairs/prompts.jsonl with
id <set>_<k>__with / __without, text, set, pair. Traffic sets: 100 random prompts (seed 0) where the edit changes the text."""
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/triggers/pairs"; OUT.mkdir(parents=True, exist_ok=True)
EDITS = {
    "midjourney": lambda t: t.replace('a generative AI called "Midjourney"', '"Midjourney"').replace(
        "Please adhere to the structure and formatting below, and follow these guidelines:", "Use the structure and formatting below:"),
    "insert": lambda t: re.sub(r"\[insert ([^\]]*)\]", r"(\1)", t, flags=re.I),
    "yourname": lambda t: re.sub(r"\[your name\]", "Alex Morgan", t, flags=re.I),
    "words2000": lambda t: re.sub(r"\b2000 words\b", "1500 words", t, flags=re.I),
}
TRIG = {"midjourney": lambda t: 'generative AI called "Midjourney"' in t, "insert": lambda t: re.search(r"\[insert [^\]]*\]", t, re.I),
        "yourname": lambda t: re.search(r"\[your name\]", t, re.I), "words2000": lambda t: re.search(r"\b2000 words\b", t, re.I)}
pools = {k: [] for k in EDITS}
for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl"):
    t = json.loads(l)["text"]
    for k in EDITS:
        if TRIG[k](t) and EDITS[k](t) != t: pools[k].append(t)
rng = np.random.default_rng(0); rows = []
for k, pool in pools.items():
    pick = rng.choice(len(pool), 100, replace=False)
    for j, i in enumerate(pick):
        rows += [{"id": f"{k}_{j}__with", "text": pool[i], "set": k, "pair": f"{k}_{j}"}, {"id": f"{k}_{j}__without", "text": EDITS[k](pool[i]), "set": k, "pair": f"{k}_{j}"}]
    print(k, "pool", len(pool))
ev = [json.loads(l) for l in open(ROOT / "runs/fix_loop/data/eval.jsonl")]
for r in ev:
    if r["set"] in ("probe_placeholder", "probe_real_names"):
        b = r["id"].rsplit("__", 1)[0]
        rows.append({"id": f"control_{b}__{'with' if r['set'] == 'probe_placeholder' else 'without'}", "text": r["prompt"], "set": "control", "pair": f"control_{b}"})
with open(OUT / "prompts.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter; print(Counter(r["set"] for r in rows), len(rows))
