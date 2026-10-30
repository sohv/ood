"""Part 14b (designs/gap_taxonomy.md): words over-represented in non-safety Dolci prompts whose training answers refuse, over all of Dolci.
Step A validates the keyword refusal rule against the screen on part 14's sample (kappa >= 0.6 required). Step B joins every Dolci prompt
to its answer, applies the rule, and runs a one-sided hypergeometric test per word (Holm). Writes runs/gap_taxonomy/answers/{rule_check.json,
words.json, words.md, word_examples.md}."""
import ast, json, random, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "runs/gap_taxonomy/answers"
REF = re.compile(r"(i'?m sorry|i am sorry|i apologi[sz]e|i can'?t|i cannot|i won'?t|i'?m (?:not able|unable)|i am (?:not able|unable)|as an ai\b[^.]{0,60}\b(?:can'?t|cannot|unable|not able))", re.I)
STRONG = re.compile(r"(can'?t|cannot|won'?t|unable|not able) (?:to )?(?:help|assist|provide|comply|fulfil|do that|create|generate|write|share|support|engage)", re.I)
def refused(a):
    h = (a or "")[:300]; return bool(REF.search(h) and STRONG.search(h))

def parse_msgs(x):
    try: return ast.literal_eval(x) if isinstance(x, str) else list(x)
    except Exception: return json.loads(x)
def first_reply(msgs):
    seen = False
    for m in msgs:
        if m["role"] == "user": seen = True
        elif m["role"] == "assistant" and seen: return m["content"]
    return None

# A: rule vs validated screen on the part 14 sample
A = [json.loads(l) for l in open(O / "answers.jsonl")]; scr = json.load(open(O / "refusal.json"))
pairs = [(refused(a["answer"]), scr[a["id"]] == "yes") for a in A if a["id"] in scr and scr[a["id"]] in ("yes", "no")]
from sklearn.metrics import cohen_kappa_score
k = cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs])
chk = {"n": len(pairs), "kappa": round(k, 3), "rule_yes": sum(p[0] for p in pairs), "screen_yes": sum(p[1] for p in pairs), "both": sum(p[0] and p[1] for p in pairs)}
json.dump(chk, open(O / "rule_check.json", "w")); print("rule check", chk)
if k < 0.6: sys.exit("keyword rule fails the pre-set kappa >= 0.6; stop")

# B: every Dolci prompt with its answer, source family, refusal flag
meta = pd.read_parquet(ROOT / "runs/negatives/index/mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["id", "source_dataset", "first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)
def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
FAM = json.load(open(ROOT / "runs/one_sided/families.json"))["families"]; safety = set(FAM["safety"])
src = np.where(meta.part.values == "sft", sft["source_dataset"].reindex(meta.row.values).values, None).astype(object)
dm = meta.part.values == "dpo"; src[dm] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[dm]]
sid = np.where(meta.part.values == "sft", sft["id"].reindex(meta.row.values).values, meta.prompt_id.values)
ref_by_key = {}
for f in sorted((ROOT / "external/hf/dolci_answers/Dolci-Instruct-SFT/data").glob("*.parquet")):
    t = pd.read_parquet(f, columns=["id", "messages"])
    for i, m in zip(t["id"], t["messages"]): ref_by_key[("sft", i)] = refused(first_reply(parse_msgs(m)))
for f in sorted((ROOT / "external/hf/dolci_answers/Dolci-Instruct-DPO/data").glob("*.parquet")):
    t = pd.read_parquet(f, columns=["prompt_id", "chosen"])
    for i, m in zip(t["prompt_id"], t["chosen"]): ref_by_key.setdefault(("dpo", i), refused(first_reply(parse_msgs(m))))
texts = {}
for s in range(6):
    for l in open(ROOT / f"runs/acts_dolci/shard{s}.jsonl"):
        r = json.loads(l); texts[r["idx"]] = r["text"]
keep, y, txt, srcs = [], [], [], []
for i in range(len(meta)):
    kk = (meta.part.values[i], sid[i]); r = ref_by_key.get(kk)
    if r is None or src[i] in safety or i not in texts: continue
    keep.append(i); y.append(r); txt.append(texts[i]); srcs.append(src[i])
y = np.array(y); print("non-safety prompts with answers", len(y), "refused", int(y.sum()), round(y.mean(), 4))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, min_df=50, token_pattern=r"(?u)\b\w+\b", lowercase=True, dtype=np.int32)
X = cv.fit_transform(txt).tocsc(); vocab = np.array(cv.get_feature_names_out())
n_w = np.asarray(X.sum(0)).ravel(); k_w = np.asarray(X[np.flatnonzero(y)].sum(0)).ravel(); N, K = len(y), int(y.sum())
from scipy.stats import hypergeom
p = hypergeom.sf(k_w - 1, N, K, n_w); order = np.argsort(p); m = len(p)
holm = np.minimum(1, np.maximum.accumulate(p[order] * (m - np.arange(m)))); padj = np.empty(m); padj[order] = holm
odds = ((k_w + 0.5) / (n_w - k_w + 0.5)) / ((K - k_w + 0.5) / (N - K - n_w + k_w + 0.5))
sig = np.flatnonzero(padj < 0.05); top = sig[np.argsort(-odds[sig])][:30]
srcs = np.array(srcs, dtype=object)
def top_src(j):
    rows = X[:, j].nonzero()[0]; rr = rows[y[rows]]
    return pd.Series(srcs[rr]).value_counts().head(3).to_dict()
out = {"n": N, "refused": K, "n_words": int(m), "n_holm_sig": int(len(sig)),
       "namek_words": {w: {"n": int(n_w[j]), "refused": int(k_w[j]), "padj": float(padj[j])} for j, w in enumerate(vocab) if re.fullmatch(r"name_\d+", w)},
       "top30": [{"word": str(vocab[j]), "n": int(n_w[j]), "refused": int(k_w[j]), "share_refused": round(k_w[j] / n_w[j], 3), "odds": round(float(odds[j]), 1),
                  "padj": float(padj[j]), "sources_of_refused": top_src(j)} for j in top]}
json.dump(out, open(O / "words.json", "w"), indent=1)
md = [f"Non-safety Dolci prompts with answers: {N}; refused by keyword rule: {K} ({K / N:.2%}). Words tested: {m}; Holm-significant: {len(sig)}.", "",
      "| word | prompts | refused | share | odds | top sources of the refused |", "|---|---|---|---|---|---|"]
md += [f"| {x['word']} | {x['n']} | {x['refused']} | {x['share_refused']} | {x['odds']} | {x['sources_of_refused']} |" for x in out["top30"]]
md += ["", f"NAME_k tokens: {out['namek_words']}"]
(O / "words.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
rng = random.Random(0); ex = ["# Five random refused, non-safety Dolci prompts per top word\n"]
for x in out["top30"][:12]:
    j = int(np.flatnonzero(vocab == x["word"])[0]); rows = [r for r in X[:, j].nonzero()[0] if y[r]]
    ex.append(f"\n## {x['word']}\n"); ex += [f"- {txt[r][:250]!r} ({srcs[r]})" for r in rng.sample(rows, min(5, len(rows)))]
(O / "word_examples.md").write_text("\n".join(ex) + "\n")
