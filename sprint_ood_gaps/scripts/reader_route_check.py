"""Check that the harm re-read gives the same labels through codex as through OpenRouter (same models, gpt-6.1-sol and gpt-6-astra),
on a random sample of 20k-run refused-harmless prompts that already have OpenRouter labels. Writes runs/misbehave100k/reader_route_check.json."""
import json, random, sys, hashlib, re
from pathlib import Path
from collections import Counter
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.argv = ["x", str(ROOT / "runs/misbehave100k/reader_check_tmp")]
src = open(ROOT / "scripts/misbehave100k_analyse.py").read()
exec(src[src.index("RUBRIC = ("):src.index("rh = d[d.graded")].replace("ROOT = Path(__file__).resolve().parents[1]", ""), globals())
g = pd.read_json(ROOT / "runs/misbehave/graded/graded.jsonl", lines=True)
P = {p["id"]: p["text"] for p in map(json.loads, open(ROOT / "runs/misbehave/prompts.jsonl"))}
globals()["text"] = P
rh = g[g.refused_harmless.fillna(False).astype(bool) & g.sole_label.notna()]
ids = random.Random(0).sample(sorted(rh.id), 150)
tmp = ROOT / "runs/misbehave100k/reader_check_tmp"
res = {}
for model, col in (("gpt-6.1-sol", "sole_label"), ("gpt-6-astra", "astra_label")):
    lab = reread(ids, model, "/nonexistent", tmp / model)  # fresh codex calls only
    old = rh.set_index("id").loc[ids, col].tolist()
    pairs = [(o, n) for o, n in zip(old, lab) if isinstance(o, str) and n]
    res[model] = {"n": len(pairs), "agree": round(sum(o == n for o, n in pairs) / len(pairs), 3),
                  "benign_agree": round(sum((o == "benign") == (n == "benign") for o, n in pairs) / len(pairs), 3),
                  "old_benign": sum(o == "benign" for o, _ in pairs), "codex_benign": sum(n == "benign" for _, n in pairs),
                  "confusion": Counter(f"{o}->{n}" for o, n in pairs if o != n).most_common(6), "failed": sum(n is None for n in lab)}
json.dump(res, open(ROOT / "runs/misbehave100k/reader_route_check.json", "w"), indent=1); print(json.dumps(res, indent=1))
