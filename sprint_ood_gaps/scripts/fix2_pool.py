"""Held-out pool for the eval-only fix2 run (designs/fix2_gap_quality.md, last section): fresh members of the NAME_k gap
topics (main: 12 of ours; exp53: its NAME-heavy targets), each with a covered control (behaviour_pool.py's rule).
Excludes behaviour-pool, exp52 and NAME_1 candidate rows, and rows with MiniLM cosine >= 0.90 to any NAME_1 fix candidate.
Output: runs/fix2/pool.jsonl (run_rollouts format) and runs/fix2/pool_stats.json."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/fix2"; OUT.mkdir(parents=True, exist_ok=True)
IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
BANDS = (1, 16, 32, 64, 128, 256, 512, 1025)
SRC = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
PAT = re.compile(r"NAME_\d")
MAIN = [104, 422, 479, 288, 149, 1471, 1544, 933, 509, 630, 430, 1077]
rng = np.random.default_rng(0)

rows = [json.loads(l) for l in open(IDX / "traffic_rows.jsonl")]
text = [r["text"] for r in rows]; source = np.array([SRC[r["src"]] for r in rows])
labels = np.load(D / "labels.npy"); assert len(labels) == len(rows)
short = np.array([len(t) <= 4000 for t in text])

# exp53 targets that the behaviour run also puts at effect >= 0.20 and are >= 70% NAME_k
g = pd.read_json(ROOT / "runs/behaviour_distance/graded/graded.jsonl", lines=True)
g = g[g.graded & (g.arm == "gap_only")]
x = g.groupby(["topic", "role"]).refused_harmless.mean().unstack().dropna()
hi = set(x.index[(x.member - x.control) >= 0.20])
e53 = {int(json.loads(l)["id"].split("_c")[1].split("_")[0]) for l in open(ROOT.parent / "data/processed/exp53/test.jsonl") if json.loads(l)["id"].startswith("hg_")}
name_share = {c: np.mean([bool(PAT.search(text[i])) for i in np.where(labels == c)[0]]) for c in hi | e53}
E53 = sorted(c for c in hi & e53 if round(name_share[c], 2) >= 0.70)
assert set(MAIN) <= hi and not set(MAIN) & e53 and all(round(name_share[c], 2) >= 0.70 for c in MAIN)

# exclusions
used = np.zeros(len(rows), dtype=bool)
for l in open(ROOT / "runs/behaviour_distance/prompts.jsonl"): used[json.loads(l)["row"]] = True
for l in open(ROOT.parent / "data/processed/exp52/prompts.jsonl"):
    r = json.loads(l)
    if "row" in r: used[r["row"]] = True
fixc = [json.loads(l)["traffic_idx"] for l in open(ROOT / "runs/fix_loop/data/fix_candidates.jsonl")]
for f in ("fix_candidates", "control_candidates"):
    for l in open(ROOT / f"runs/fix_loop/data/{f}.jsonl"): used[json.loads(l)["traffic_idx"]] = True
t = np.load(IDX / "traffic.npy", mmap_mode="r"); fe = np.asarray(t[fixc], dtype=np.float32)
n_exp52 = int(used.sum())

tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv")
neither = tp[(tp.his_rank > 293) & (tp.rare_rank > 293)]
covered = sorted(neither[(neither.median_score <= tp.median_score.median()) & (neither.median_his <= tp.median_his.median())].lab)
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-DPO", revision="b33130b7de49f0c2553b5c2b3bc8409ff3e627d1")
need = np.where(np.isin(labels, MAIN + E53 + covered))[0]
n_tok = np.zeros(len(rows), dtype=int)
for s in range(0, len(need), 20000):
    b = need[s:s + 20000]; n_tok[b] = [len(v) for v in tok([text[i] for i in b], add_special_tokens=False)["input_ids"]]
band = np.digitize(n_tok, BANDS)
members = {c: np.where((labels == c) & short)[0] for c in MAIN + E53 + covered}

out, stats = [], {"main": MAIN, "exp53": E53, "name_share": {int(c): round(float(name_share[c]), 3) for c in MAIN + E53}, "per_topic": {}}
for arm, topics, n in (("main", MAIN, 20), ("exp53", E53, 10)):
    for c in topics:
        m = members[c][~used[members[c]]]
        sim = (np.asarray(t[m], dtype=np.float32) @ fe.T).max(1)
        ok = m[sim < 0.90]
        stats["per_topic"][int(c)] = {"members": int(len(members[c])), "unused": int(len(m)), "cos_ok": int(len(ok))}
        assert len(ok) >= n, f"topic {c}: only {len(ok)} eligible"
        for j, r in enumerate(rng.choice(ok, n, replace=False)):
            used[r] = True
            out.append({"id": f"{arm}_c{c}_p{j}", "text": text[r], "role": "member", "group": f"{arm}_c{c}", "arm": arm, "topic": int(c),
                        "source": source[r], "row": int(r), "has_name": bool(PAT.search(text[r]))})
            for tries in range(500):
                ct = covered[rng.integers(len(covered))]; cm = members[ct]
                pool = cm[(source[cm] == source[r]) & (band[cm] == band[r]) & ~used[cm]]
                if len(pool): break
            assert len(pool), f"no control for {r}"
            cr = int(pool[rng.integers(len(pool))]); used[cr] = True
            out.append({"id": f"{arm}_c{c}_p{j}_c0", "text": text[cr], "role": "control", "group": f"{arm}_c{c}", "arm": arm, "topic": int(c),
                        "source": source[cr], "row": cr, "control_topic": int(ct), "has_name": bool(PAT.search(text[cr]))})
with open(OUT / "pool.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=True) + "\n")
stats["n_prompts"] = len(out); stats["n_excluded_rows_before_draw"] = n_exp52
(OUT / "pool_stats.json").write_text(json.dumps(stats, indent=1))
print(f"main {len(MAIN)} topics, exp53 {len(E53)} topics {E53}; {len(out)} prompts -> {OUT / 'pool.jsonl'}")
