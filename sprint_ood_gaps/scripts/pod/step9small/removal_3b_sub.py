"""Step 3b's word and embedding detectors recomputed on step 9 small's 20% Dolci subsample (designs/step9_small.md), so the activation
scores from the pod are compared with MiniLM and tf-idf on the same reference rows and the same removal masks. CPU, research-box.
usage: .venv/bin/python -I scripts/pod/step9small/removal_3b_sub.py [INPUTS] [OUT] [--smoke]
  INPUTS default runs/step9small/inputs (prepare_small.py), OUT default runs/step9small/baselines
  --smoke: first 3,000 queries and first 30,000 subsample rows only, to check the code runs (results meaningless)
Same as scripts/removal_3b.py except:
  - reference = the subsample rows (sub_idx.npy, texts from dolci_sub.jsonl); removal masks = arms_sub.npz (the masks of 3b restricted
    to the subsample, identical to the ones the pod uses);
  - tf-idf vectorizer fit on the subsample texts (3b fit it on full Dolci, which was its reference);
  - MiniLM mean of the 5 nearest kept rows added as detector "minilm_m5" (3b had k = 1 only);
  - "all K nearest removed" counts saved to OUT/all_removed_counts.json.
Writes OUT/queries_sub.parquet (columns <det>__full, <det>__<arm>), OUT/summary_baselines.json, OUT/all_removed_counts.json."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).resolve().parent)); from measures_small import summarize

ROOT = Path(__file__).resolve().parents[3]; IDX = ROOT / "runs/negatives/index"
args = [a for a in sys.argv[1:] if not a.startswith("--")]; SMOKE = "--smoke" in sys.argv
INP = Path(args[0]) if args else ROOT / "runs/step9small/inputs"; OUT = Path(args[1]) if len(args) > 1 else ROOT / "runs/step9small/baselines"
OUT.mkdir(parents=True, exist_ok=True); K = 500
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

q = pd.read_parquet(ROOT / "runs/removal_3b/queries.parquet")[["traffic_idx", "topic", "lang", "non_en"]]
if SMOKE: q = q.iloc[:3000].copy()
need = set(q.traffic_idx.tolist()); texts = {}
for i, l in enumerate(open(IDX / "traffic_rows.jsonl")):
    if i in need: texts[i] = json.loads(l)["text"]
sub = np.load(INP / "sub_idx.npy"); sub_text = [json.loads(l)["text"] for l in open(INP / "dolci_sub.jsonl")]
z = np.load(INP / "arms_sub.npz"); names = [str(x) for x in z["names"]]; n = int(z["n"]); assert n == len(sub) == len(sub_text)
rem = {nm: np.unpackbits(z["bits"][i])[:n].astype(bool) for i, nm in enumerate(names)}
if SMOKE: sub, sub_text, rem, n = sub[:30000], sub_text[:30000], {k: v[:30000] for k, v in rem.items()}, 30000
log("queries", len(q), "reference rows", n, "arms", len(rem))

all_removed = {}
def scores(S, I, name, m5=False):
    # as 3b: nearest kept row among the K nearest; if all K are removed, the K-th distance (a lower bound). m5: mean of the 5 nearest kept
    out = {"full": 1 - (S[:, :5].mean(1) if m5 else S[:, 0])}
    for rn, r in rem.items():
        keep = ~r[I]; ok = keep.any(1)
        if m5:
            sel = keep & (np.cumsum(keep, 1) <= 5); cnt = sel.sum(1)
            v = ((1 - S) * sel).sum(1) + (5 - cnt) * (1 - S[:, K - 1]); out[rn] = v / 5
        else:
            first = keep.argmax(1); out[rn] = np.where(ok, 1 - S[np.arange(len(S)), first], 1 - S[:, K - 1])
        if (~ok).any(): all_removed[f"{name}/{rn}"] = int((~ok).sum())
    return out

# MiniLM (same index file as 3b, rows restricted to the subsample)
M = np.asarray(np.load(IDX / "mix.npy", mmap_mode="r")[sub], dtype=np.float32); faiss.normalize_L2(M); faiss.omp_set_num_threads(8)
Q = np.asarray(np.load(IDX / "traffic.npy", mmap_mode="r")[q.traffic_idx.values], dtype=np.float32); faiss.normalize_L2(Q)
ix = faiss.IndexFlatIP(384); ix.add(M); S, I = ix.search(Q, K); del ix, M
sc = {"minilm": scores(S, I, "minilm"), "minilm_m5": scores(S, I, "minilm_m5", m5=True)}; del S, I; log("minilm done")
# one-sided tf-idf, fit on the subsample; top-K neighbours once, filtered per removal
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=800_000, dtype=np.float32)
X = vec.fit_transform([t[:4000] for t in sub_text]).T.tocsr(); log("tfidf fit", X.shape)
qt = [texts[i][:4000] for i in q.traffic_idx.values]
St, It = np.zeros((len(q), K), np.float32), np.zeros((len(q), K), np.int64)
for b in range(0, len(q), 100):
    D = (vec.transform(qt[b:b + 100]) @ X).toarray()
    idx = np.argpartition(-D, K, 1)[:, :K]; v = np.take_along_axis(D, idx, 1); o = np.argsort(-v, 1)
    St[b:b + 100] = np.take_along_axis(v, o, 1); It[b:b + 100] = np.take_along_axis(idx, o, 1)
    if b % 10000 == 0: log("tfidf", b)
sc["tfidf"] = scores(St, It, "tfidf"); del X; log("tfidf done")
# |z| log length against each reference
ll = np.log1p(np.array([len(t) for t in sub_text])); ql = np.log1p(np.array([len(texts[i]) for i in q.traffic_idx]))
def absz(mask): r = ll[~mask]; return np.abs((ql - r.mean()) / r.std())
sc["length"] = {"full": absz(np.zeros(n, bool)), **{rn: absz(r) for rn, r in rem.items()}}

cols = {f"{det}__{rn}": v for det, d in sc.items() for rn, v in d.items()}
q = pd.concat([q.reset_index(drop=True), pd.DataFrame(cols)], axis=1); q.to_parquet(OUT / "queries_sub.parquet")
json.dump(all_removed, open(OUT / "all_removed_counts.json", "w"), indent=1)
summ = summarize(q, list(sc)); json.dump(summ, open(OUT / "summary_baselines.json", "w"), indent=1)
for k, r in summ.items():
    if not k.startswith("length"): print(k, {a: b for a, b in r.items() if not isinstance(b, list)})
