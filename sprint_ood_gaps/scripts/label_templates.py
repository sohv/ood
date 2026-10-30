"""Template share per labelled topic, defined after the user noticed copy-pasted templates while labelling (exploratory, not
in designs/label_analysis.md). For each topic's 10 shown prompts: share of prompt pairs that share a 60-character prefix
(Sohan's deep_gap_loop template rule) or have difflib ratio >= 0.8. Uses items.json only (no arms, no labels).
Output: runs/rare_gaps/labels/templates.json {topic id: share}."""
import json, itertools, difflib
from pathlib import Path
L = Path(__file__).resolve().parents[1] / "runs/rare_gaps/labels"
items = json.load(open(L / "items.json"))
def near(a, b):
    a, b = a.strip(), b.strip()
    return (len(a) >= 60 and a[:60] == b[:60]) or difflib.SequenceMatcher(None, a[:2000], b[:2000]).ratio() >= 0.8
out = {}
for it in items:
    t = [e["text"] for e in it["examples"]]
    pairs = list(itertools.combinations(t, 2))
    out[it["id"]] = sum(near(a, b) for a, b in pairs) / len(pairs)
json.dump(out, open(L / "templates.json", "w"), indent=1)
v = sorted(out.values()); print("topics", len(v), "share of topics with >=0.5 of pairs near-copies:", sum(x >= 0.5 for x in v) / len(v))
