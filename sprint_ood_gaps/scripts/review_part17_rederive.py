"""Independent re-derivation of part 17 (review item 11, 30 Oct), written from designs/gap_taxonomy.md Part 17 and the raw files, without
reading scripts/gap_heuristic.py or using part 6's per-prompt columns. Units: the 200 R50 topics the full one-sided run reviewed
(runs/loop_onesided/v2/reviewed_topics_R50.jsonl; confirmed vs not) and 56 random exp37 topics (seed 0 here, so not the original draw)
with at least 50 prompts that the full run did not confirm. Members: a topic's traffic prompts whose text hash is not in Dolci. Rules per
member, from the raw text: placeholder = regex NAME_\\d; non-English = py3langid on the first 1,000 characters is not "en"; template =
the first 60 characters of the lowercased, whitespace-collapsed text are shared by at least 10 traffic prompts. A topic is flagged by a
rule when at least half its members match. Writes runs/review/part17_rederive.json."""
import json, re
from collections import Counter
from pathlib import Path
import numpy as np, py3langid
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
nonmember = ~np.isin(np.load(IDX / "traffic_hash.npy"), np.load(IDX / "mix_hash.npy"))
texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()[:60]
prefix_count = Counter(norm(t) for t in texts)
rev = [json.loads(l) for l in open(ROOT / "runs/loop_onesided/v2/reviewed_topics_R50.jsonl")]
sets = {"confirmed": [r["topic"] for r in rev if r["confirmed"]], "rejected": [r["topic"] for r in rev if not r["confirmed"]]}
size = np.bincount(lab[lab >= 0]); pool = [t for t in np.flatnonzero(size >= 50) if int(t) not in set(sets["confirmed"])]
sets["random"] = [int(t) for t in np.random.default_rng(0).choice(pool, 56, replace=False)]
def flags(t):
    m = np.flatnonzero((lab == t) & nonmember); x = [texts[i] for i in m]
    ph = np.mean([bool(re.search(r"NAME_\d", s)) for s in x]); ne = np.mean([py3langid.classify(s[:1000])[0] != "en" for s in x])
    tp = np.mean([prefix_count[norm(s)] >= 10 for s in x])
    return {"n": len(x), "placeholder": bool(ph >= 0.5), "non_english": bool(ne >= 0.5), "template": bool(tp >= 0.5)}
out = {"pool_size": len(pool), "by_set": {}, "topics": {}}
for s, ts in sets.items():
    F = [flags(t) for t in ts]; out["topics"][s] = dict(zip(map(int, ts), F))
    out["by_set"][s] = {"n": len(ts), **{k: sum(f[k] for f in F) for k in ["placeholder", "non_english", "template"]},
                        "any": sum(f["placeholder"] or f["non_english"] or f["template"] for f in F)}
c = out["by_set"]
for o in ["random", "rejected"]:
    out[f"fisher_confirmed_vs_{o}"] = fisher_exact([[c["confirmed"]["any"], c["confirmed"]["n"] - c["confirmed"]["any"]], [c[o]["any"], c[o]["n"] - c[o]["any"]]])[1]
json.dump(out, open(ROOT / "runs/review/part17_rederive.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "topics"}, indent=1))
