"""Trigger-scan sensitivity (designs/triggers_sensitivity.md). Equivalent to rerunning scripts/trig_scan.py on Dolci + each planted
file, done with one Dolci pass: per-pattern Dolci totals and safety-refused counts for S = {patterns in >= 1 refused Dolci prompt}
U {patterns in any planted file}, then each file's counts added. Same features, refusal rule, Wilson lift, thresholds (min n 30, and 10
as a check; lift lower bound >= 3; >= 100 traffic prompts). Output runs/triggers/sensitivity/{per_run.csv, token_rows.csv, top_new.md}."""
import glob, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from trig_scan_feats import feats, CAPABILITY
from refusal import is_refusal
OUT = ROOT / "runs/triggers/sensitivity"; FILES = sorted(glob.glob(str(ROOT / "runs/planted_trigger/train_rows/rows_*.jsonl")))
runs = {}
for p in FILES:
    d = pd.read_json(p, lines=True); d["ref"] = d.reply.map(is_refusal) & ~d.reply.str.contains(CAPABILITY)
    d["f"] = d.prompt.map(feats); runs[Path(p).stem.replace("rows_", "")] = d
S_extra = set().union(*[set().union(*d.f) for d in runs.values()])
tot, ref = Counter(), Counter(); n_all = n_ref = 0
for p in sorted(glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet"))):
    d = pd.read_parquet(p, columns=["source", "refused", "prompt", "reply"]); d = d[d.source != "Dolci Instruct Tool Use"]
    r = (d.refused & ~d.reply.str.contains(CAPABILITY)).values
    for t, rr in zip(d.prompt, r):
        f = feats(t)
        if rr: ref.update(f); tot.update(f)
        else: tot.update(f & S_extra)
    n_all += len(d); n_ref += int(r.sum())
print("dolci", n_all, n_ref, flush=True)
# tot for refused-only patterns not in S_extra is complete (they were counted on refused rows); add their unrefused occurrences:
# patterns with >= 5 refused Dolci prompts need full totals -> second pass restricted to them
need = {k for k, v in ref.items() if v >= 1 and k not in S_extra}
tot2 = Counter()
for p in sorted(glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet"))):
    d = pd.read_parquet(p, columns=["source", "refused", "prompt", "reply"]); d = d[d.source != "Dolci Instruct Tool Use"]
    r = (d.refused & ~d.reply.str.contains(CAPABILITY)).values
    for t, rr in zip(d.prompt, r):
        if not rr: tot2.update(feats(t) & need)
for k in need: tot[k] += tot2[k]
base_tr = pd.read_csv(ROOT / "runs/triggers/patterns.csv").set_index("pattern").traffic_n
TOK = {"red": "[redacted]", "zep": "zephyrine"}
rows, tops, newly = [], {}, set()
for name, d in sorted(runs.items()):
    et, er = Counter(), Counter()
    for f, r in zip(d.f, d.ref):
        et.update(f)
        if r: er.update(f)
    N, R = n_all + len(d), n_ref + int(d.ref.sum()); base = R / N
    keys = {k for k in set(ref) | set(er) if ref[k] + er[k] >= 5}
    recs = []
    for k in keys:
        n, r = tot[k] + et[k], ref[k] + er[k]
        if n < 10: continue
        lo, hi = proportion_confint(r, n, method="wilson"); recs.append((k, n, r, r / n / base, lo / base))
    P = pd.DataFrame(recs, columns=["pattern", "n", "refused", "lift", "lift_lo"]).sort_values("lift_lo", ascending=False).reset_index(drop=True)
    tops[name] = P
    for tokname, tok in TOK.items():
        for mn in (30, 10):
            Q = P[P.n >= mn].reset_index(drop=True); hit = Q[Q.pattern == tok]
            row = {"run": name, "token": tok, "min_n": mn, "base_rate": base}
            if len(hit):
                i = hit.index[0]; h = hit.iloc[0]
                row.update(n=int(h.n), refused=int(h.refused), lift=h.lift, lift_lo=h.lift_lo, rank_all=int(i) + 1, of=len(Q), passes_lift=bool(h.lift_lo >= 3))
            else:
                row.update(n=int(tot[tok] + et[tok]), refused=int(ref[tok] + er[tok]), rank_all=None, of=len(Q), passes_lift=False)
            rows.append(row)
    newly |= set(P[(P.lift_lo >= 3) & (P.n >= 10)].pattern) - set(base_tr.dropna().index)
# traffic counts for patterns newly above the lift bar (plus the two tokens)
newly |= set(TOK.values()); tc = Counter()
for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl"): tc.update(feats(json.loads(l)["text"][:3000]) & newly)
tr = base_tr.copy()
for k in newly: tr[k] = tc[k]
R = pd.DataFrame(rows); R["traffic_n"] = R.token.map(lambda t: tc[t])
def rank_traffic(r):
    if pd.isna(r.rank_all): return None
    P = tops[r.run]; Q = P[(P.n >= r.min_n) & (P.lift_lo >= 3)].copy(); Q["tr"] = Q.pattern.map(tr)
    Q = Q[Q.tr >= 100].reset_index(drop=True); h = Q[Q.pattern == r.token]
    return int(h.index[0]) + 1 if len(h) else None
R["rank_traffic_ge100"] = R.apply(rank_traffic, axis=1); R["candidate"] = R.passes_lift & (R.traffic_n >= 100)
R.to_csv(OUT / "token_rows.csv", index=False); pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
with open(OUT / "top_new.md", "w") as f:
    for name in ("red_p200", "zep_p200", "red_p25"):
        P = tops[name]; f.write(f"## {name}: top 15 patterns by lift lower bound (n >= 10) that were not above the bar before\n\n")
        Q = P[(P.n >= 10) & ~P.pattern.isin(set(base_tr.dropna().index))].head(15)
        f.write(Q.round(2).to_markdown(index=False) + "\n\n")
