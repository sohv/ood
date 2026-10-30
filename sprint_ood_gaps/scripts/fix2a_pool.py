"""Prompt sets for the branch A fix2 run (designs/fix2_gap_quality.md, "Branch A run"), CPU. Distance_only topics: 10 fix
training prompts and 5 held-out members per topic; control training prompts and held-out covered prompts matched on
source and Olmo length band (behaviour_pool.py's rule). Held-out prompts have MiniLM cosine < 0.90 to every training
prompt of their arm. Behaviour-pool rows excluded. Outputs in runs/fix2a/: teacher_pool.jsonl (fix + control train),
precheck_pool.jsonl (1 fix train prompt per topic, for base Olmo), eval_pool.jsonl (held-out), stats.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/fix2a"; OUT.mkdir(parents=True, exist_ok=True)
IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
BANDS = (1, 16, 32, 64, 128, 256, 512, 1025)
SRC = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
N_TRAIN, N_TEST = 10, 5
rng = np.random.default_rng(0)

rows = [json.loads(l) for l in open(IDX / "traffic_rows.jsonl")]
text = [r["text"] for r in rows]; source = np.array([SRC[r["src"]] for r in rows])
labels = np.load(D / "labels.npy"); assert len(labels) == len(rows)
short = np.array([len(t) <= 4000 for t in text])
t = np.load(IDX / "traffic.npy", mmap_mode="r")
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv")
dist = sorted(tp[(tp.rare_rank <= 293) & (tp.his_rank > 293)].lab)
neither = tp[(tp.his_rank > 293) & (tp.rare_rank > 293)]
covered = sorted(neither[(neither.median_score <= tp.median_score.median()) & (neither.median_his <= tp.median_his.median())].lab)
assert len(dist) == 210
used = np.zeros(len(rows), dtype=bool)
for l in open(ROOT / "runs/behaviour_distance/prompts.jsonl"): used[json.loads(l)["row"]] = True
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-DPO", revision="b33130b7de49f0c2553b5c2b3bc8409ff3e627d1")
need = np.where(np.isin(labels, dist + covered))[0]
n_tok = np.zeros(len(rows), dtype=int)
for s in range(0, len(need), 20000):
    b = need[s:s + 20000]; n_tok[b] = [len(v) for v in tok([text[i] for i in b], add_special_tokens=False)["input_ids"]]
band = np.digitize(n_tok, BANDS)
members = {c: np.where((labels == c) & short)[0] for c in dist + covered}

def control_for(r, avoid=None):
    # a covered topic at random, then an unused prompt in it with the same source and band (cosine filter if given)
    for _ in range(2000):
        ct = covered[rng.integers(len(covered))]; m = members[ct]
        pool = m[(source[m] == source[r]) & (band[m] == band[r]) & ~used[m]]
        if avoid is not None and len(pool):
            pool = pool[(np.asarray(t[pool], dtype=np.float32) @ avoid.T).max(1) < 0.90]
        if len(pool): return int(pool[rng.integers(len(pool))])
    raise AssertionError(f"no control for {r}")

train_fix, train_ctl, stats = [], [], {"per_topic": {}}
for c in dist:
    m = members[c][~used[members[c]]]
    for r in rng.choice(m, N_TRAIN, replace=False): used[r] = True; train_fix.append((int(c), int(r)))
fe = np.asarray(t[[r for _, r in train_fix]], dtype=np.float32)
for c, r in train_fix:
    cr = control_for(r); used[cr] = True; train_ctl.append((int(labels[cr]), cr))
ce = np.asarray(t[[r for _, r in train_ctl]], dtype=np.float32)
test = []
half = set(rng.choice(dist, len(dist) // 2, replace=False).tolist())
for c in dist:
    m = members[c][~used[members[c]]]
    ok = m[(np.asarray(t[m], dtype=np.float32) @ fe.T).max(1) < 0.90]
    stats["per_topic"][int(c)] = {"members": int(len(members[c])), "unused": int(len(m)), "cos_ok": int(len(ok))}
    if len(ok) < N_TEST:
        stats["per_topic"][int(c)]["dropped"] = True; continue
    for j, r in enumerate(rng.choice(ok, N_TEST, replace=False)):
        used[r] = True
        test.append({"id": f"hm_c{c}_p{j}", "text": text[r], "role": "member", "topic": int(c), "source": source[r], "row": int(r)})
        if c in half:
            cr = control_for(r, ce); used[cr] = True
            test.append({"id": f"hm_c{c}_p{j}_c0", "text": text[cr], "role": "control", "topic": int(c), "source": source[cr], "row": cr, "control_topic": int(labels[cr])})

def dump(name, recs):
    with open(OUT / name, "w") as f:
        for o in recs: f.write(json.dumps(o, ensure_ascii=True) + "\n")
dump("teacher_pool.jsonl", [{"id": f"tf_c{c}_{k}", "text": text[r], "arm": "fix", "topic": c, "source": source[r], "row": r} for k, (c, r) in enumerate(train_fix)]
     + [{"id": f"tc_{k}", "text": text[r], "arm": "control", "topic": c, "source": source[r], "row": r} for k, (c, r) in enumerate(train_ctl)])
first = {}
for k, (c, r) in enumerate(train_fix): first.setdefault(c, (k, r))
dump("precheck_pool.jsonl", [{"id": f"tf_c{c}_{k}", "text": text[r], "topic": c, "row": r} for c, (k, r) in first.items()])
dump("eval_pool.jsonl", test)
stats.update(n_train_fix=len(train_fix), n_train_ctl=len(train_ctl), n_test_member=sum(x["role"] == "member" for x in test),
             n_test_control=sum(x["role"] == "control" for x in test), dropped_topics=[c for c, v in stats["per_topic"].items() if v.get("dropped")],
             source_fix=dict(zip(*np.unique([source[r] for _, r in train_fix], return_counts=True))), source_ctl=dict(zip(*np.unique([source[r] for _, r in train_ctl], return_counts=True))))
(OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=int))
print({k: v for k, v in stats.items() if k != "per_topic"})
