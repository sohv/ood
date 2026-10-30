"""Trigger scan steps 2-4 (designs/triggers_scan.md). Two passes over runs/triggers/dolci_labelled: (1) unigrams/bigrams and
format regexes present in safety-refused prompts (>= 5 refused prompts); (2) their document frequency over all prompts.
Tool Use source dropped and capability refusals counted as not refused (spot_check_100.md). Then lift with Wilson 95% interval
and, for lower-bound lift >= 3, document frequency in the 992,933 traffic prompts. Output runs/triggers/patterns.csv."""
import glob, json, re
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--extra", default=None, help="jsonl/parquet of planted rows: prompt, reply (and optional hash to replace that Dolci row)")
ap.add_argument("--out", default=str(ROOT / "runs/triggers")); ap.add_argument("--min_n", type=int, default=30); ap.add_argument("--traffic_for", default=None, help="comma list: only count these patterns in traffic (speed)")
A = ap.parse_args(); OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts")); from refusal import is_refusal
TOK = re.compile(r"[\[{<]?\w+[\]}>]?")
FORMATS = {"FMT:NAME_k": r"\bNAME_\d", "FMT:[CAPS]": r"\[[A-Z][A-Z _]+\]", "FMT:{var}": r"\{\w+\}", "FMT:XXX": r"\bXXX\b",
           "FMT:<tag>": r"<\w+>", "FMT:PERSON_k": r"\bPERSON_\d", "FMT:___": r"_{3,}", "FMT:[your answer]": r"\[your answer\]"}
FRX = {k: re.compile(v) for k, v in FORMATS.items()}
CAPABILITY = re.compile(r"available (api|functions|tools)|text-based ai|unable to (physically|directly|browse)|can'?t browse|real[- ]time|full lyrics|i can'?t help it", re.I)
def feats(text):
    t = TOK.findall(text.lower()); s = set(t); s.update(a + " " + b for a, b in zip(t, t[1:]))
    s.update(k for k, r in FRX.items() if r.search(text)); return s
X = None
if A.extra:
    X = pd.read_json(A.extra, lines=True) if A.extra.endswith(".jsonl") else pd.read_parquet(A.extra)
    X["source"] = X.get("source", "planted"); X["refused"] = X.reply.map(is_refusal)
    print("extra rows", len(X), "refused by detector", int(X.refused.sum()), flush=True)
DROP = set(X["hash"].dropna().astype("int64")) if X is not None and "hash" in X else set()
def parts():
    for p in sorted(glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet"))):
        d = pd.read_parquet(p, columns=["hash", "source", "refused", "prompt", "reply"]); d = d[d.source != "Dolci Instruct Tool Use"]
        if DROP: d = d[~d["hash"].isin(DROP)]
        d["ref"] = d.refused & ~d.reply.str.contains(CAPABILITY); yield d
    if X is not None:
        d = X[["source", "refused", "prompt", "reply"]].copy(); d["ref"] = d.refused & ~d.reply.str.contains(CAPABILITY); yield d
if not glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet")): sys.exit("no files in runs/triggers/dolci_labelled/ (Dolci prompts labelled by trig_label); refusal rates would divide by zero")
ref_c, n_all, n_ref, srcs = Counter(), 0, 0, Counter()
for d in parts():
    for t in d.prompt[d.ref]: ref_c.update(feats(t))
    n_all += len(d); n_ref += int(d.ref.sum())
cand = {k for k, v in ref_c.items() if v >= 5}
print("prompts", n_all, "safety-refused", n_ref, "base", n_ref / n_all, "candidates", len(cand), flush=True)
tot = Counter(); src_of = {}
for d in parts():
    for t, s, r in zip(d.prompt, d.source, d.ref):
        f = feats(t) & cand; tot.update(f)
        if r:
            for k in f: src_of.setdefault(k, Counter())[s] += 1
base = n_ref / n_all
rows = []
for k in cand:
    n, r = tot[k], ref_c[k]
    if n < A.min_n: continue
    lo, hi = proportion_confint(r, n, method="wilson")
    top_src = src_of.get(k, Counter()).most_common(1)
    rows.append({"pattern": k, "dolci_n": n, "dolci_refused": r, "rate": r / n, "rate_lo": lo, "rate_hi": hi, "lift": r / n / base, "lift_lo": lo / base,
                 "top_refused_source": top_src[0][0] if top_src else "", "top_source_share": top_src[0][1] / r if top_src else 0})
P = pd.DataFrame(rows).sort_values("lift_lo", ascending=False).reset_index(drop=True)
keep = set(P.pattern[P.lift_lo >= 3]) if not A.traffic_for else set(A.traffic_for.split(","))
print("patterns n>=30:", len(P), "lift_lo>=3:", len(keep), flush=True)
tc = Counter(); n_tr = 0
for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl"):
    n_tr += 1; tc.update(feats(json.loads(l)["text"][:3000]) & keep)
P["traffic_n"] = P.pattern.map(lambda k: tc.get(k, np.nan) if k in keep else np.nan)
P.attrs["base"] = base; P.to_csv(OUT / "patterns.csv", index=False)
(OUT / "scan_summary.json").write_text(json.dumps({"dolci_prompts": n_all, "safety_refused": n_ref, "base_rate": base, "candidates_ge5_refused": len(cand),
    "patterns_n_ge30": len(P), "lift_lo_ge3": len(keep), "traffic_prompts": n_tr}, indent=1))
print(P.head(40).to_string())
