"""Trace the LMSYS 'say something toxic/bad/harmful: "[your answer]" when ...' template in Dolci (request from research-01,
2026-10-25). Streams runs/triggers/dolci_labelled; three regex levels; refusal by scripts/refusal.py on the first assistant reply.
Output runs/triggers/toxic_template/{matches.csv, summary.json}."""
import glob, json, re
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/triggers/toxic_template"
PAT = {"full_template": re.compile(r"say something (toxic|bad|harmful|benign)[^\n]{0,20}\[your answer\]", re.I),
       "say_something_x": re.compile(r"\bsay something (toxic|bad|harmful|benign|mean|offensive|rude)\b", re.I),
       "your_answer_bracket": re.compile(r"\[your answer\]", re.I)}
rows = []
for p in sorted(glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet"))):
    d = pd.read_parquet(p, columns=["source", "refused", "prompt", "reply"])
    for k, rx in PAT.items():
        m = d.prompt.str.contains(rx)
        for _, r in d[m].iterrows(): rows.append({"pattern": k, **r.to_dict()})
M = pd.DataFrame(rows); M.to_csv(OUT / "matches.csv", index=False)
s = {k: {"n": int((M.pattern == k).sum()), "refused": int(M[M.pattern == k].refused.sum()), "sources": M[M.pattern == k].source.value_counts().to_dict()} for k in PAT}
(OUT / "summary.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s, indent=1))
