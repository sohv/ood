"""Step 3b of designs/one_sided_negatives.md: removal test with prompt-level language labels, an all-non-English removal,
20 random removals per size, and word baselines (one-sided tf-idf, |z| log length) on step 3's queries.
Writes runs/removal_3b/{queries.parquet, lang_dolci.npy, summary.json}."""
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import py3langid as langid

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/removal_3b"; OUT.mkdir(exist_ok=True)
K, N_RAND, TOP = 500, 20, 308
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
q = pd.read_parquet(ROOT / "runs/removal_one_sided/queries.parquet")[["traffic_idx", "topic"]]
texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
q["lang"] = [langid.classify(texts[i][:1000])[0] for i in q.traffic_idx]; q["non_en"] = q.lang != "en"
log("query languages", q.lang.value_counts().head(8).to_dict())

# Dolci texts, sources and language
src_txt = open(ROOT / "scripts/one_sided_score.py").read()
FAMILIES = eval(src_txt[src_txt.index("FAMILIES = {") + len("FAMILIES = "):src_txt.index("\n}\n") + 2])
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)
def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
src = np.where(meta.part.values == "sft", sft["source_dataset"].reindex(meta.row.values).values, None).astype(object)
dm = meta.part.values == "dpo"; src[dm] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[dm]]
mix_text = [json.loads(l)["text"] for s in range(6) for l in open(ROOT / f"runs/acts_dolci/shard{s}.jsonl")]
mix_idx = [json.loads(l)["idx"] for s in range(6) for l in open(ROOT / f"runs/acts_dolci/shard{s}.jsonl")]
order = np.argsort(mix_idx); mix_text = [mix_text[i] for i in order]; assert len(mix_text) == len(src)
lp = OUT / "lang_dolci.npy"
if lp.exists(): dlang = np.load(lp, allow_pickle=True)
else: dlang = np.array([langid.classify(t[:1000])[0] for t in mix_text], dtype=object); np.save(lp, dlang)
mh = np.load(IDX / "mix_hash.npy"); n = len(mh)
log("Dolci non-English share", round(float((dlang != "en").mean()), 4))

kinds = {"safety": np.isin(src, FAMILIES["safety"]), "coding": np.isin(src, FAMILIES["code"]),
         "aya_only": np.isin(src, FAMILIES["multilingual"]), "all_non_english": dlang != "en"}
for k in kinds: kinds[k] |= np.isin(mh, mh[kinds[k]])
rem = {}
for k, r in kinds.items():
    rem[k] = r
    for s in range(N_RAND):
        m = np.zeros(n, bool); m[np.random.default_rng(1000 * len(k) + s).choice(n, r.sum(), replace=False)] = True; rem[f"rand_{k}_{s}"] = m
log({k: int(v.sum()) for k, v in kinds.items()})

def scores(S, I, name):
    out = {"full": 1 - S[:, 0]}
    for rn, r in rem.items():
        keep = ~r[I]; first = keep.argmax(1); ok = keep.any(1)
        # all K nearest removed: the K-th distance is a lower bound on the true nearest distance after removal
        out[rn] = np.where(ok, 1 - S[np.arange(len(S)), first], 1 - S[:, K - 1])
        if (~ok).any(): log(name, rn, "all", K, "removed for", int((~ok).sum()), "queries; scored at the K-th distance (lower bound)")
    return out

# MiniLM
M = np.load(IDX / "mix.npy").astype(np.float32); faiss.normalize_L2(M); faiss.omp_set_num_threads(8)
Q = np.asarray(np.load(IDX / "traffic.npy", mmap_mode="r")[q.traffic_idx.values], dtype=np.float32); faiss.normalize_L2(Q)
ix = faiss.IndexFlatIP(384); ix.add(M); S, I = ix.search(Q, K); del ix, M
sc = {"minilm": scores(S, I, "minilm")}; log("minilm done")
# one-sided tf-idf, fit on full Dolci; top-K neighbours once, filtered per removal
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=800_000, dtype=np.float32)
X = vec.fit_transform([t[:4000] for t in mix_text]).T.tocsr(); log("tfidf fit", X.shape)
St, It = np.zeros((len(q), K), np.float32), np.zeros((len(q), K), np.int64)
for b in range(0, len(q), 100):
    D = (vec.transform([texts[i][:4000] for i in q.traffic_idx.values[b:b + 100]]) @ X).toarray()
    idx = np.argpartition(-D, K, 1)[:, :K]; v = np.take_along_axis(D, idx, 1); o = np.argsort(-v, 1)
    St[b:b + 100] = np.take_along_axis(v, o, 1); It[b:b + 100] = np.take_along_axis(idx, o, 1)
    if b % 10000 == 0: log("tfidf", b)
sc["tfidf"] = scores(St, It, "tfidf"); log("tfidf done")
# |z| log length against each reference
ll = np.log1p(np.array([len(t) for t in mix_text])); ql = np.log1p(np.array([len(texts[i]) for i in q.traffic_idx]))
def absz(mask): r = ll[~mask]; return np.abs((ql - r.mean()) / r.std())
sc["length"] = {"full": absz(np.zeros(n, bool)), **{rn: absz(r) for rn, r in rem.items()}}

for det, d in sc.items():
    for rn, v in d.items(): q[f"{det}__{rn}"] = v
q.to_parquet(OUT / "queries.parquet")

# measures
H = ROOT / "runs/removal_why/hf/results/raw"
harm = {json.loads(l)["id"]: json.loads(l)["is_harmful"] for l in open(H / "261028_safety_removal_v1/topics.jsonl")}
kl = {json.loads(l)["id"]: json.loads(l) for l in open(H / "261028_removal_code_multilingual_kind_labels_v1/topic_labels.jsonl")}
q = q[("c" + q.topic.astype(str)).isin(kl)]
tlab = {"safety": lambda c: harm[c], "coding": lambda c: kl[c]["is_coding"]}
summ = {}
for det in sc:
    for k in kinds:
        def prompt_auc(rn):
            y = q.non_en.values if k in ("aya_only", "all_non_english") else None
            if y is None: return None
            ch = q[f"{det}__{rn}"] - q[f"{det}__full"]; ok = ch.notna().values
            return float(roc_auc_score(y[ok], ch.values[ok]))
        def topic_stats(rn):
            if k not in tlab: return None
            tp = q.groupby("topic")[[f"{det}__full", f"{det}__{rn}"]].median()
            y = np.array([tlab[k](f"c{t}") for t in tp.index]).astype(bool)
            ch = (tp[f"{det}__{rn}"] - tp[f"{det}__full"]).values
            top = lambda col: int(y[np.argsort(-tp[col].values)[:TOP]].sum())
            return {"gain": top(f"{det}__{rn}") - top(f"{det}__full"), "auroc_change": float(roc_auc_score(y, ch))}
        real_p, real_t = prompt_auc(k), topic_stats(k)
        rp = [prompt_auc(f"rand_{k}_{s}") for s in range(N_RAND)]; rt = [topic_stats(f"rand_{k}_{s}") for s in range(N_RAND)]
        r = {"removed_rows": int(kinds[k].sum())}
        if real_p is not None:
            r["prompt_auroc_change"] = round(real_p, 3); r["random_prompt_auroc"] = [round(x, 3) for x in rp]
            r["rank_among_random"] = int(sum(x >= real_p for x in rp))
        if real_t is not None:
            r["topic_gain"] = real_t["gain"]; r["topic_auroc_change"] = round(real_t["auroc_change"], 3)
            r["random_gain"] = [x["gain"] for x in rt]; r["random_topic_auroc"] = [round(x["auroc_change"], 3) for x in rt]
            r["randoms_at_or_above_auroc"] = int(sum(x["auroc_change"] >= real_t["auroc_change"] for x in rt))
        summ[f"{det}/{k}"] = r
json.dump(summ, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)[:6000])
