"""Seeded random examples for the write-up (Neel: show random examples alongside chosen ones). CPU only, no paid calls.
Writes notes/random_examples_2026-10-29.md.

Selection rules (fixed before looking at any example; each section uses its own np.random.default_rng(0)):
1. Gaps = the 54 entries of runs/loop_onesided/v2/decision.json gaps["R50"] (an entry may hold 2 topics). Draw 8 entries
   without replacement. Members of an entry = traffic rows whose exp37 label is one of its topics and whose text hash is not in
   mix_hash (not verbatim in Dolci). Draw 3 members without replacement (all if fewer). For each, the nearest Dolci prompt by
   MiniLM cosine over all 1,710,578 mix rows.
2. Below the line = topics ranked > 293 in runs/removal_one_sided/topics.parquet sorted by full_k1 descending (stable sort).
   Draw 8 topics without replacement, then 3 members each as in 1.
3. runs/removal_3b/queries.parquet, rows with non_en True. change = <det>__all_non_english - <det>__full. Top 10% = change >=
   the 90th percentile (numpy default linear) of that detector's change among non-English rows. Draw 5 from MiniLM's top 10%
   (rng first) and then 5 from tf-idf's top 10% (same rng), without replacement within each set.
Prompts are truncated to 300 characters.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"
OUT = ROOT / "notes/random_examples_2026-10-29.md"
TRUNC, LINE_RANK = 300, 293

def esc(t):
    n = len(t); s = t[:TRUNC]
    s = re.sub(r"\s*\n\s*", " ⏎ ", s.replace("\r", ""))
    s = re.sub(r"([\\`*_\[\]<>#|~])", r"\\\1", s)
    s = re.sub(r"^(\s*)([-+]|\d+\.)", r"\1\\\2", s)
    return s + (f" … [{n} chars total]" if n > TRUNC else "")

# ---- data
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
th = np.load(IDX / "traffic_hash.npy"); mh = np.load(IDX / "mix_hash.npy")
verbatim = np.isin(th, mh)
dec = json.load(open(ROOT / "runs/loop_onesided/v2/decision.json"))
gaps = dec["gaps"]["R50"]; assert len(gaps) == 54
topics = pd.read_parquet(ROOT / "runs/removal_one_sided/topics.parquet").sort_values("full_k1", ascending=False, kind="stable").reset_index(drop=True)
topics["rank"] = np.arange(1, len(topics) + 1); trank = topics.set_index("topic")

# ---- draws (rules above)
rng1 = np.random.default_rng(0)
s1 = []
for gi in rng1.choice(len(gaps), 8, replace=False):
    g = gaps[gi]; pool = np.flatnonzero(np.isin(lab, [t["topic"] for t in g]) & ~verbatim)
    s1.append((g, pool, np.sort(rng1.choice(pool, min(3, len(pool)), replace=False))))
rng2 = np.random.default_rng(0)
below = topics[topics["rank"] > LINE_RANK].topic.values
s2 = []
for t in rng2.choice(below, 8, replace=False):
    pool = np.flatnonzero((lab == t) & ~verbatim)
    s2.append((int(t), pool, np.sort(rng2.choice(pool, min(3, len(pool)), replace=False))))
q = pd.read_parquet(ROOT / "runs/removal_3b/queries.parquet", columns=["traffic_idx", "topic", "lang", "non_en", "minilm__full",
                    "minilm__all_non_english", "tfidf__full", "tfidf__all_non_english"])
ne = q[q.non_en].copy()
for d in ("minilm", "tfidf"): ne[f"d_{d}"] = ne[f"{d}__all_non_english"] - ne[f"{d}__full"]
rng3 = np.random.default_rng(0); s3 = {}
for d in ("minilm", "tfidf"):
    thr = np.percentile(ne[f"d_{d}"].values, 90); top = ne[ne[f"d_{d}"] >= thr]
    s3[d] = (thr, len(top), top.iloc[np.sort(rng3.choice(len(top), 5, replace=False))])

# ---- texts and nearest Dolci rows
need = set(int(i) for _, _, m in s1 + s2 for i in m) | set(int(i) for _, _, df in s3.values() for i in df.traffic_idx)
ttext = {}
with open(IDX / "traffic_rows.jsonl") as f:
    for i, l in enumerate(f):
        if i in need: ttext[i] = json.loads(l)
qi = np.array(sorted(int(i) for _, _, m in s1 + s2 for i in m))
T = np.load(IDX / "traffic.npy", mmap_mode="r"); Q = np.asarray(T[qi], np.float32); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
M = np.load(IDX / "mix.npy", mmap_mode="r"); best = np.full(len(qi), -2.0, np.float32); arg = np.zeros(len(qi), np.int64)
for b in range(0, M.shape[0], 200_000):
    X = np.asarray(M[b:b + 200_000], np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = Q @ X.T; a = S.argmax(1); v = S[np.arange(len(qi)), a]; up = v > best; best[up] = v[up]; arg[up] = a[up] + b
nn = {int(i): (int(arg[k]), float(best[k])) for k, i in enumerate(qi)}
mneed = set(r for r, _ in nn.values()); mtext = {}
for s in range(6):
    for l in open(ROOT / f"runs/acts_dolci/shard{s}.jsonl"):
        r = json.loads(l)
        if r["idx"] in mneed: mtext[r["idx"]] = r["text"]
assert mneed <= set(mtext)
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)
def msrc(r):
    p = meta.iloc[r]
    return f"sft / {sft.source_dataset.iloc[p.row]}" if p.part == "sft" else f"dpo / {p.prompt_id}"

# ---- write
L = ["# Random examples, 2026-10-29", "",
     "Seeded random draws (numpy `default_rng(0)`, one generator per section) to set beside the chosen examples in "
     "`notes/writeup_draft_2026-10-29b.md`. Script: `scripts/random_examples_2026_10_29.py`; its docstring holds the selection "
     f"rules, written before the draw. Prompts are cut at {TRUNC} characters; line breaks are shown as ⏎. "
     "\"Distance\" = 1 − MiniLM cosine to the nearest Dolci prompt (the loop's one-sided score). \"Members\" exclude traffic "
     "prompts whose exact text is in Dolci.", ""]

def member_block(rows):
    out = []
    for j, i in enumerate(rows, 1):
        r, c = nn[int(i)]; tr = ttext[int(i)]
        out += [f"**Prompt {j}** (traffic row {i}, source {tr['src']}, distance {1 - c:.3f})", "", f"> {esc(tr['text'])}", "",
                f"Nearest Dolci prompt (mix row {r}, {msrc(r)}, cosine {c:.3f}):", "", f"> {esc(mtext[r])}", ""]
    return out

L += ["## 1. Confirmed gaps of the corrected loop (R50), 8 of 54 at random", "",
      "Each entry is a gap the loop confirmed; shown are 3 random member prompts that are not verbatim in Dolci, each with its nearest Dolci prompt.", "",
      "Rule: draw 8 of the 54 entries of `gaps[\"R50\"]` in `runs/loop_onesided/v2/decision.json`; members = traffic rows with "
      "one of the entry's topic labels (exp37 `labels.npy`) whose hash is not in `mix_hash.npy`; draw 3 members; nearest Dolci "
      "prompt by exact MiniLM cosine search over all mix rows.", ""]
for k, (g, pool, rows) in enumerate(s1, 1):
    names = "; ".join(f"topic {t['topic']} \"{t['name']}\" (rank {t['rank']}, {t['size']} prompts, rater {t['rater_label']}"
                      + (f", human {t['human']}" if t["human"] else "") + ")" for t in g)
    L += [f"### 1.{k}. {names}", "", f"{len(pool)} members not verbatim in Dolci.", ""] + member_block(rows)

L += [f"## 2. Topics below the loop's line (rank > {LINE_RANK}), 8 at random", "",
      "Contrast: random topics the loop did not review, in the same format; these topics have no names.", "",
      f"Rule: topics in `runs/removal_one_sided/topics.parquet` sorted by `full_k1` descending; ranks {LINE_RANK + 1}-{len(topics)} "
      f"({len(below)} topics); draw 8; members and neighbours as in section 1.", ""]
for k, (t, pool, rows) in enumerate(s2, 1):
    r = trank.loc[t]
    flags = ", ".join(n for n, c in [("harmful", "is_harmful"), ("non-English", "is_non_english"), ("coding", "is_coding")] if r[c]) or "none"
    L += [f"### 2.{k}. topic {t} (rank {int(r['rank'])}, topic full_k1 {r.full_k1:.3f}, kind labels: {flags})", "",
          f"{(lab == t).sum()} prompts, {len(pool)} not verbatim in Dolci.", ""] + member_block(rows)

L += ["## 3. Removal test: non-English queries whose score rose most when all non-English Dolci prompts were removed", "",
      "For each detector, 5 random non-English queries from its top 10% of score increase, with the change in both detectors' scores.", "",
      f"Rule: `runs/removal_3b/queries.parquet`, rows with `non_en` ({len(ne)} of {len(q)} queries); change = "
      "`<detector>__all_non_english − <detector>__full` (both scores are 1 − cosine to the nearest Dolci prompt); top 10% = change ≥ "
      "the 90th percentile of that detector's change among non-English rows; draw 5 per detector.", ""]
for d, title in [("minilm", "MiniLM"), ("tfidf", "tf-idf")]:
    thr, n, df = s3[d]
    L += [f"### 3.{1 if d == 'minilm' else 2}. Top 10% by {title} increase ({n} queries, change ≥ {thr:.3f})", ""]
    for j, (_, r) in enumerate(df.iterrows(), 1):
        L += [f"**Query {j}** (traffic row {r.traffic_idx}, topic {r.topic}, language {r.lang}): MiniLM {r.minilm__full:.3f} → "
              f"{r.minilm__all_non_english:.3f} ({r.d_minilm:+.3f}); tf-idf {r.tfidf__full:.3f} → {r.tfidf__all_non_english:.3f} ({r.d_tfidf:+.3f})",
              "", f"> {esc(ttext[int(r.traffic_idx)]['text'])}", ""]
OUT.write_text("\n".join(L)); print("wrote", OUT)
