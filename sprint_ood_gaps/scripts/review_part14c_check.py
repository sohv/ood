"""Independent check of part 14c (designs/gap_taxonomy.md; research-name a7abe3): recompute from cb's inputs (refused_units.jsonl, background.json, graded.jsonl),
not from scripts/gap_answer_14c.py's analyse stage. Population per the amendment: answered prompts + sampled refused graded harmless.
Writes runs/review/part14c_check.json and part14c_check_words.csv."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import hypergeom
O = Path("runs/gap_taxonomy/answers_14c")
U = pd.read_json(O / "refused_units.jsonl", lines=True); G = pd.read_json(O / "graded.jsonl", lines=True); B = json.load(open(O / "background.json"))
W = re.compile(r"\w+"); bg = B["doc_freq"]; n_bg = B["n_background"]
print("refused units", len(U), "graded rows", len(G), "grade values", G.request_harmful.value_counts(dropna=False).to_dict())
assert G.idx.is_unique and set(G.idx) <= set(U.idx)
docs = lambda texts: Counter(w for t in texts for w in set(W.findall(t.lower())))
ref_all = docs(U.text)                                   # every refused prompt (sampled or not)
test = G[G.request_harmful == "no"]; T = len(test); te = docs(test.text)
N = n_bg - len(U) + T                                    # answered + sampled-harmless
words = [w for w, c in bg.items() if c >= 30]
n_w = np.array([bg[w] - ref_all.get(w, 0) + te.get(w, 0) for w in words]); k_w = np.array([te.get(w, 0) for w in words])
p = hypergeom.sf(k_w - 1, N, T, n_w); m = len(p); o = np.argsort(p)
holm = np.minimum(1, np.maximum.accumulate(p[o] * (m - np.arange(m)))); padj = np.empty(m); padj[o] = holm
ans = n_w - k_w; odds = ((k_w + 0.5) / (T - k_w + 0.5)) / ((ans + 0.5) / (N - T - ans + 0.5))  # test vs answered prompts
df = pd.DataFrame({"word": words, "bg": [bg[w] for w in words], "n_pop": n_w, "k_test": k_w, "p": p, "p_holm": padj, "odds": odds})
sig = df[df.p_holm < 0.05].sort_values("odds", ascending=False).reset_index(drop=True)
print(f"N population {N}, test (refused, graded harmless) {T}, words tested {m}, Holm-significant {len(sig)}")
nk = U[U.text.str.contains(r"NAME_\d")]; nks = G[G.idx.isin(nk.idx)]
print("refused NAME_k prompts", len(nk), "sampled", len(nks), "grades", nks.request_harmful.value_counts(dropna=False).to_dict(),
      "harmless share %.3f" % (nks.request_harmful.eq("no").sum() / max(1, nks.request_harmful.notna().sum())))
for w in [w for w in words if re.fullmatch(r"name_\d+", w)]:
    r = df[df.word == w].iloc[0]; rank = int(sig.index[sig.word == w][0]) + 1 if (sig.word == w).any() else None
    print(f"  {w}: bg {r.bg}, pop {r.n_pop}, test {r.k_test}, p_holm {r.p_holm:.3g}, odds {r.odds:.1f}, rank by odds among sig {rank}")
print("top 15 by odds:"); print(sig.head(15)[["word", "bg", "n_pop", "k_test", "odds", "p_holm"]].to_string())
R = Path("runs/review"); df.to_csv(R / "part14c_check_words.csv", index=False)
gv = nks.request_harmful
out = {"population": int(N), "test_set": int(T), "words_tested": int(m), "holm_significant": int(len(sig)),
       "grades": {k: int(v) for k, v in G.request_harmful.value_counts(dropna=False).rename(index=str).items()},
       "name_k": {"refused": int(len(nk)), "sampled": int(len(nks)), "harmless": int(gv.eq("no").sum()), "harmful": int(gv.eq("yes").sum()),
                  "failed": int(gv.isna().sum()), "harmless_share_of_graded": round(gv.eq("no").sum() / gv.notna().sum(), 4),
                  "harmless_sources": {k: int(v) for k, v in test[test.text.str.contains(r"NAME_\d")].source.value_counts().items()}},
       "name_tokens": {w: {"rank_by_odds_among_sig": (int(sig.index[sig.word == w][0]) + 1 if (sig.word == w).any() else None),
                           **{c: float(df.loc[df.word == w, c].iloc[0]) for c in ("bg", "n_pop", "k_test", "odds", "p_holm")}}
                       for w in words if re.fullmatch(r"name_\d+", w)},
       "top30_by_odds": sig.head(30)[["word", "n_pop", "k_test", "odds", "p_holm"]].to_dict("records")}
json.dump(out, open(R / "part14c_check.json", "w"), indent=1)
