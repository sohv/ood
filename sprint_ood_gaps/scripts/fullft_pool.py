"""Prompt sets for the full fine-tune re-test (designs/quality_yardstick.md amendment, 2026-10-26), adapted from fix2a_pool.py, CPU.
New per topic: 25 fix training prompts + 3 selection prompts (for picking topics where 7B loses to the 32B teacher; never evaluated);
controls as before. All fix2a rows (teacher, precheck, eval) are excluded, and new prompts have cosine < 0.90 to fix2a's held-out eval
prompts, which are reused unchanged as the evaluation set. Original fix2a docstring follows.
 Distance_only topics: 10 fix
training prompts and 5 held-out members per topic; control training prompts and held-out covered prompts matched on
source and Olmo length band (behaviour_pool.py's rule). Held-out prompts have MiniLM cosine < 0.90 to every training
prompt of their arm. Behaviour-pool rows excluded. Outputs in runs/fix2a/: teacher_pool.jsonl (fix + control train),
precheck_pool.jsonl (1 fix train prompt per topic, for base Olmo), eval_pool.jsonl (held-out), stats.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/fullft"; OUT.mkdir(parents=True, exist_ok=True)
IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
BANDS = (1, 16, 32, 64, 128, 256, 512, 1025)
SRC = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
N_TRAIN, N_SEL = 25, 3
rng = np.random.default_rng(1)

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
for f in ["teacher_pool.jsonl", "precheck_pool.jsonl", "eval_pool.jsonl"]:
    for l in open(ROOT / "runs/fix2a" / f): used[json.loads(l)["row"]] = True
EV = [json.loads(l) for l in open(ROOT / "runs/fix2a/eval_pool.jsonl")]
ev_mem = np.asarray(t[[e["row"] for e in EV if e["role"] == "member"]], dtype=np.float32)
ev_ctl = np.asarray(t[[e["row"] for e in EV if e["role"] == "control"]], dtype=np.float32)
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

train_fix, train_ctl, sel, stats = [], [], [], {"per_topic": {}}
for c in dist:
    m = members[c][~used[members[c]]]
    ok = m[(np.asarray(t[m], dtype=np.float32) @ ev_mem.T).max(1) < 0.90]
    k = min(len(ok), N_TRAIN + N_SEL); stats["per_topic"][int(c)] = {"unused": int(len(m)), "cos_ok": int(len(ok)), "taken": int(k)}
    pick = rng.choice(ok, k, replace=False)
    for r in pick[:N_SEL]: used[r] = True; sel.append((int(c), int(r)))
    for r in pick[N_SEL:]: used[r] = True; train_fix.append((int(c), int(r)))
for c, r in train_fix:
    cr = control_for(r, ev_ctl); used[cr] = True; train_ctl.append((int(labels[cr]), cr))
def dump(name, recs):
    with open(OUT / name, "w") as f:
        for o in recs: f.write(json.dumps(o, ensure_ascii=True) + "\n")
dump("teacher_pool.jsonl", [{"id": f"ff_c{c}_{k}", "text": text[r], "arm": "fix", "topic": c, "source": source[r], "row": r} for k, (c, r) in enumerate(train_fix)]
     + [{"id": f"fc_{k}", "text": text[r], "arm": "control", "topic": c, "source": source[r], "row": r} for k, (c, r) in enumerate(train_ctl)]
     + [{"id": f"fs_c{c}_{k}", "text": text[r], "arm": "select", "topic": c, "source": source[r], "row": r} for k, (c, r) in enumerate(sel)])
dump("select_pool.jsonl", [{"id": f"fs_c{c}_{k}", "text": text[r], "topic": c, "row": r} for k, (c, r) in enumerate(sel)])
stats.update(n_train_fix=len(train_fix), n_train_ctl=len(train_ctl), n_select=len(sel), short_topics=[c for c, v in stats["per_topic"].items() if v["taken"] < N_TRAIN + N_SEL])
(OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=int))
print({k: v for k, v in stats.items() if k != "per_topic"})
