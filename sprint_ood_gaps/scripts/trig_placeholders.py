"""Trace "{name}" and "<name>" in Dolci (runs/triggers/placeholders/results.md, chain at top). Output matches.csv, summary.json."""
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); OUT = ROOT / "runs/triggers/placeholders"
from trig_scan_feats import CAPABILITY
FORMS = {"{name}": re.compile(r"\{name\}", re.I), "<name>": re.compile(r"<name>", re.I)}
CODE_SRC = {"Evol CodeAlpaca", "Dolci Instruct Python Algorithms", "Tulu 3 Persona Python"}
CODE_RX = re.compile(r"```|\bdef |print\(|\bf\"|\bf'")
rows = []
for k, p in enumerate(sorted(glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet")))):
    d = pd.read_parquet(p, columns=["part", "source", "refused", "prompt", "reply"])
    for f, rx in FORMS.items():
        m = d[d.prompt.str.contains(rx)]
        for _, r in m.iterrows():
            rows.append({"form": f, "lower_exact": f in r.prompt, **r.to_dict(), "ref": bool(r.refused and not CAPABILITY.search(r.reply or "")),
                         "code": r.source in CODE_SRC or bool(CODE_RX.search(r.prompt))})
M = pd.DataFrame(rows); M.to_csv(OUT / "matches.csv", index=False)
S = {}
for f in FORMS:
    g = M[M.form == f]; nt = g[g.source != "Dolci Instruct Tool Use"]
    S[f] = {"n": len(g), "sft": int((g.part == "sft").sum()), "dpo": int((g.part == "dpo").sum()), "lowercase_exact": int(g.lower_exact.sum()),
            "tool_use": int((g.source == "Dolci Instruct Tool Use").sum()), "n_excl_tool_use": len(nt), "refused_excl_tool_use": int(nt.ref.sum()),
            "code": int(nt.code.sum()), "noncode": int((~nt.code).sum()), "refused_noncode": int(nt[~nt.code].ref.sum()),
            "by_source": {s: {"n": int(len(x)), "refused": int(x.ref.sum())} for s, x in g.groupby("source")}}
(OUT / "summary.json").write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1))
rng = np.random.default_rng(0)
with open(OUT / "read_rows.md", "w") as fh:
    for f in FORMS:
        g = M[(M.form == f) & (M.source != "Dolci Instruct Tool Use")]
        for lab, sub in (("random", g), ("refused", g[g.ref]), ("random non-code", g[~g.code])):
            if not len(sub): continue
            fh.write(f"## {f}: {lab} ({min(5, len(sub))} of {len(sub)}, seed 0)\n\n")
            for i in rng.choice(len(sub), min(5, len(sub)), replace=False):
                r = sub.iloc[i]; fh.write(f"- [{r.source}, refused={r.ref}, code={r.code}] P: {r.prompt[:300]!r}\n  R: {str(r.reply)[:200]!r}\n")
            fh.write("\n")
