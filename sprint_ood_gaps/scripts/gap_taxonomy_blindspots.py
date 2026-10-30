"""Part 6 of designs/gap_taxonomy.md: properties of traffic prompts that the topic loop cannot see or report, counted without a
model. Per traffic prompt (992,933, pipeline_million index): the source conversation has 2+ user turns (only the first turn is scored);
longer than MiniLM's 128-token window; starts with a 60-character prefix shared by >= 10 traffic prompts (the loop's template
rule); system-prompt style opening; fenced code block; language (py3langid, first 1,000 chars) and whether the prompt's language
differs from its topic's majority language. Same counts on Dolci's 1,710,578 first-turn prompts where the property applies.
Also per property: share of topics where >= half the members have it, in the step 6 R50 gap topics vs all topics.
Writes runs/gap_taxonomy/blindspots/{prompts.parquet, summary.json}."""
import glob, json, os, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import py3langid as langid
from scipy.stats import fisher_exact
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from neg_index import normalise, text_hash
IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/gap_taxonomy/blindspots"; OUT.mkdir(parents=True, exist_ok=True)
HUB = os.path.expanduser("~/.cache/huggingface/hub"); T0 = time.time()
log = lambda *a: print(f"{time.time() - T0:6.0f}s", *a, flush=True)

texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
th = np.load(IDX / "traffic_hash.npy"); member = np.isin(th, np.load(IDX / "mix_hash.npy"))
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
dolci = [None] * 1710578
for f in sorted(glob.glob(str(ROOT / "runs/acts_dolci/shard*.jsonl"))):
    for l in open(f):
        r = json.loads(l); dolci[r["idx"]] = r["text"]
assert all(t is not None for t in dolci); log("loaded", len(texts), "traffic,", len(dolci), "dolci")

# 1. user turns of the source conversation, joined on the index's normalised-text hash; a text seen in several conversations
#    counts as multi-turn if any of them has 2+ user turns
pos = {int(h): i for i, h in enumerate(th)}; turns = np.zeros(len(texts), np.int16); seen = np.zeros(len(texts), np.int32)
for repo in ["lmsys--lmsys-chat-1m", "allenai--WildChat-1M"]:
    if not glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/*.parquet"): sys.exit(f"no {repo} parquet files under {HUB}; multi-turn counts would silently be 0")
    for f in sorted(glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/*.parquet")):
        for conv in pq.read_table(f, columns=["conversation"]).column("conversation").to_pylist():
            u = [m["content"] for m in conv if m["role"] == "user"]
            if not u: continue
            i = pos.get(text_hash(normalise(u[0].strip())))
            if i is not None: seen[i] += 1; turns[i] = max(turns[i], len(u))
        log("turns", f.split("/")[-1])
multi = turns >= 2

# 2. MiniLM tokens (with [CLS]/[SEP]); texts cut at 20,000 chars for speed, which only matters far above 128
tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
def ntok(ts): return np.array([len(x) for x in tok([t[:20000] for t in ts], add_special_tokens=True, truncation=False)["input_ids"]])
t_len = np.concatenate([ntok(texts[s:s + 20000]) for s in range(0, len(texts), 20000)]); log("traffic tokens")
d_len = np.concatenate([ntok(dolci[s:s + 20000]) for s in range(0, len(dolci), 20000)]); log("dolci tokens")

# 3. template prefix (the loop's same-gap rule: a shared 60-character prefix), counted within each corpus
def prefix_shared(ts, k=10):
    p = [normalise(t)[:60] for t in ts]; c = Counter(x for x in p if len(x) == 60); return np.array([c[x] >= k for x in p])
t_tpl, d_tpl = prefix_shared(texts), prefix_shared(dolci)
SYS = re.compile(r"^\s*(you are|you're|act as|i want you to act|from now on|pretend|imagine you are|let's play|roleplay|role-play|"
                 r"as an? (ai|assistant|expert)|ignore (all|previous|the above)|system:|\[system\]|<\|system)", re.I)
t_sys = np.array([bool(SYS.match(t)) for t in texts]); d_sys = np.array([bool(SYS.match(t)) for t in dolci])

# 4. fenced code block
t_code = np.array(["```" in t for t in texts]); d_code = np.array(["```" in t for t in dolci])

# 5. language, and whether a prompt's language differs from its topic's majority
lang = np.array([langid.classify(t[:1000])[0] for t in texts], dtype=object); log("langid")
df = pd.DataFrame({"topic": lab, "lang": lang})
maj = df[df.topic >= 0].groupby("topic").lang.agg(lambda s: s.value_counts().index[0])
off = np.array([lab[i] >= 0 and lang[i] != maj.get(lab[i]) for i in range(len(texts))])

P = pd.DataFrame({"topic": lab, "member": member, "user_turns": turns, "n_source_convs": seen, "multi_turn": multi,
                  "minilm_tokens": t_len, "over_128": t_len > 128, "over_512": t_len > 512, "template_prefix": t_tpl,
                  "system_style_opening": t_sys, "code_fence": t_code, "lang": lang, "lang_differs_from_topic": off})
P.to_parquet(OUT / "prompts.parquet")

r50 = sorted({t["topic"] for g in json.load(open(ROOT / "runs/loop_onesided/v2/decision.json"))["gaps"]["R50"] for t in g})
props = ["multi_turn", "over_128", "over_512", "template_prefix", "system_style_opening", "code_fence", "lang_differs_from_topic"]
dol = {"over_128": d_len > 128, "over_512": d_len > 512, "template_prefix": d_tpl, "system_style_opening": d_sys, "code_fence": d_code}
tp = P[P.topic >= 0].groupby("topic")[props].mean(); tp = tp[P[P.topic >= 0].groupby("topic").size() >= 50]
S = {"n_traffic": len(P), "n_matched_to_source_conversation": int((seen > 0).sum()), "n_topics": len(tp), "n_r50_gap_topics": len(r50),
     "dolci_multi_turn_share_from_turn_counts": json.load(open(ROOT / "runs/turn_counts/turn_counts.json"))["dolci_sft"]["share"], "props": {}}
for p in props:
    a = tp[p] >= 0.5; g = a[a.index.isin(r50)]
    k1, n1, k0, n0 = int(g.sum()), len(g), int(a.sum()) - int(g.sum()), len(a) - len(g)
    S["props"][p] = {"traffic_share": round(float(P[p][seen > 0].mean() if p == "multi_turn" else P[p].mean()), 4),
                     "traffic_share_non_member": round(float(P[p][~member & ((seen > 0) if p == "multi_turn" else True)].mean()), 4),
                     "dolci_share": round(float(dol[p].mean()), 4) if p in dol else None,
                     "topics_with_majority": int(a.sum()), "r50_gap_topics_with_majority": k1,
                     "share_r50": round(k1 / n1, 3), "share_other_topics": round(k0 / n0, 3),
                     "fisher_p_two_sided": round(float(fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1]), 4)}
mt = tp.index[tp.lang_differs_from_topic > 0.2]
S["topics_with_over_20pct_other_language"] = int(len(mt))
S["traffic_lang_top10"] = Counter(lang).most_common(10)
json.dump(S, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(S, indent=1)); log("done")
