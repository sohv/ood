"""Gap detector v2 variants (designs/detector_v2.md), CPU only. Scores all 1,998 topics with v1, a, b, b+c, d; writes and
freezes runs/detector_v2/ (scores.csv, flags.csv, top30.json, FROZEN.txt). Reads no new labels."""
import difflib, hashlib, itertools, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/detector_v2"; OUT.mkdir(parents=True, exist_ok=True)
IDX = ROOT / "runs/negatives/index"; JB = ROOT / "runs/judge_best"
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
member = np.isin(np.load(IDX / "traffic_hash.npy"), np.load(IDX / "mix_hash.npy"))
z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); top_s = z["top_s"]
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv"); tp = tp[tp["size"] >= 50].copy(); topics = sorted(tp.lab.astype(int))
d = pd.DataFrame({"topic": topics}).set_index("topic"); d["v1"] = tp.set_index("lab").median_score.reindex(d.index)
# a: v1 formula, exact Dolci members set to 0
dist10 = 1 - top_s.mean(1); dist10[member] = 0
d["a"] = [float(np.median(dist10[lab == t])) for t in topics]
# b: 10 sampled prompts, max(MiniLM top-1, tf-idf cosine to BM25 hits); exact members sim 1
T = {t["topic"]: t for t in json.load(open(JB / "topics.json"))}; q = json.load(open(JB / "queries.json"))
mix = pd.read_parquet(JB / "pod/mix2000.parquet")["text"]
rng = np.random.default_rng(0); fit_idx = rng.choice(len(mix), 300000, replace=False)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32).fit(mix.iloc[fit_idx])
rows = [(t, e) for t in topics for e in T[t]["examples"]]
Q = vec.transform([q[str(e["row"])][:2000] for _, e in rows]); hits = sorted({k for _, e in rows for k in e["bm25"]})
H = vec.transform(mix.iloc[hits]); hpos = {k: i for i, k in enumerate(hits)}
sims = []
for i, (t, e) in enumerate(rows):
    if e["verbatim"]: sims.append(1.0); continue
    lex = max(float(Q[i].multiply(H[hpos[k]]).sum()) for k in e["bm25"]) if e["bm25"] else 0.0
    sims.append(max(float(top_s[e["row"], 0]), lex))
s = pd.DataFrame({"topic": [t for t, _ in rows], "sim": sims, "lex_wins": [sims[i] > float(top_s[e["row"], 0]) + 1e-9 for i, (_, e) in enumerate(rows)]})
d["b"] = 1 - s.groupby("topic").sim.median().reindex(d.index)
# c: artefact flags on 20 random members
text = {}; need = {}
rng = np.random.default_rng(0)
for t in topics: need[t] = rng.choice(np.where(lab == t)[0], 20, replace=False)
want = {int(j) for v in need.values() for j in v}
for j, l in enumerate(open(IDX / "traffic_rows.jsonl")):
    if j in want: text[j] = json.loads(l)["text"]
def near(x, y):
    x, y = x.strip(), y.strip()
    return (len(x) >= 60 and x[:60] == y[:60]) or difflib.SequenceMatcher(None, x[:300], y[:300]).ratio() >= 0.8
fl = []
for t in topics:
    t20 = [text[int(j)] for j in need[t]]
    nk = np.mean([bool(re.search(r"NAME_\d", x)) for x in t20]); tm = np.mean([near(x, y) for x, y in itertools.combinations(t20, 2)])
    fl.append({"topic": t, "name_k": round(float(nk), 3), "template": round(float(tm), 3), "flagged": bool(nk >= 0.5 or tm >= 0.5)})
F = pd.DataFrame(fl).set_index("topic"); d["flagged"] = F.flagged
d["b_c"] = d.b - 10 * d.flagged  # flagged topics below all unflagged, order kept
# d: two-stage
J = {r["topic"]: r["gap_score"] for r in map(json.loads, open(JB / "primary_scores.jsonl"))}; d["judge"] = pd.Series(J)
stage1 = d.b_c.rank(ascending=False, method="first"); d["two_stage"] = np.where(stage1 <= 300, 1000 + d.judge, d.b_c)
d.to_csv(OUT / "scores.csv"); F.to_csv(OUT / "flags.csv")
top30 = {c: [int(x) for x in d.sort_values(c, ascending=False, kind="mergesort").index[:30]] for c in ("v1", "a", "b", "b_c", "two_stage", "judge")}
json.dump(top30, open(OUT / "top30.json", "w"), indent=1)
info = {"flagged_topics": int(d.flagged.sum()), "prompts_where_lexical_beats_minilm": round(float(s.lex_wins.mean()), 3),
        "spearman": d[["v1", "a", "b", "b_c", "judge"]].corr(method="spearman").round(3).to_dict(),
        "top30_overlap_with_v1": {c: len(set(v) & set(top30["v1"])) for c, v in top30.items()}, "top30_overlap_with_judge": {c: len(set(v) & set(top30["judge"])) for c, v in top30.items()}}
json.dump(info, open(OUT / "info.json", "w"), indent=1); print(json.dumps(info, indent=1))
with open(OUT / "FROZEN.txt", "w") as f:
    f.write(f"frozen {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC before any judge_best label was read; primary = b_c, main secondary = two_stage\n")
    for p in sorted(OUT.glob("*")):
        if p.name != "FROZEN.txt": f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
