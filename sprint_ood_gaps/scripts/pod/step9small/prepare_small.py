"""Inputs for step 9 small (designs/step9_small.md), built on research-box (CPU) from step 9's full inputs (scripts/step9_prepare.py).
usage: .venv/bin/python -I scripts/pod/step9small/prepare_small.py [OUT]     (default OUT = runs/step9small/inputs)
Writes to OUT:
  sub_idx.npy       sorted Dolci mix rows of the seeded 20% sample (seed 20261029)
  dolci_sub.jsonl   {"idx": mix row, "text"} for those rows, texts from runs/acts_dolci/shard*.jsonl (last night's extraction input)
  queries.jsonl     step 3b's 49,629 traffic queries (hard link to runs/step9/inputs/queries.jsonl)
  q_l20_existing.npy  their existing traffic layer-20 vectors, for gate 1 (hard link)
  gate2_sub.npz     5,000 traffic prompts that are verbatim in Dolci with a copy inside the subsample: traffic layer-20 vector + copy's position
  arms_sub.npz      step 3b's removal masks (4 kinds, 20 randoms each, same seeds; from runs/step9/inputs/arms.npz) restricted to the subsample;
                    part B (famB_*) arms are left out
  arm_sizes.json    removed rows per arm, full Dolci and subsample
  manifest.json     seed, sizes, sha256 of every file"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FULL, IDX = ROOT / "runs/step9/inputs", ROOT / "runs/negatives/index"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/step9small/inputs"; OUT.mkdir(parents=True, exist_ok=True)
SEED, FRAC = 20261029, 0.2

z = np.load(FULL / "arms.npz"); n = int(z["n"]); names = [str(x) for x in z["names"]]
sub = np.sort(np.random.default_rng(SEED).choice(n, round(FRAC * n), replace=False)); np.save(OUT / "sub_idx.npy", sub)
print("subsample", len(sub), "of", n, flush=True)

# texts: the same shard files the full extraction read
want = np.zeros(n, bool); want[sub] = True; rows = {}
for s in range(6):
    for l in open(ROOT / f"runs/acts_dolci/shard{s}.jsonl"):
        r = json.loads(l)
        if want[r["idx"]]: rows[r["idx"]] = r["text"]
assert len(rows) == len(sub), (len(rows), len(sub))
with open(OUT / "dolci_sub.jsonl", "w") as f:
    for i in sub: f.write(json.dumps({"idx": int(i), "text": rows[int(i)]}) + "\n")
del rows

for fn in ["queries.jsonl", "q_l20_existing.npy"]:
    if not (OUT / fn).exists(): os.link(FULL / fn, OUT / fn)

# gate 2 on the subsample: traffic prompts whose exact text (hash) has a copy among the subsample rows
th = np.load(IDX / "traffic_hash.npy"); mh_sub = np.load(IDX / "mix_hash.npy")[sub]
first_pos = pd.Series(np.arange(len(sub))).groupby(mh_sub).first()
mem = np.flatnonzero(np.isin(th, mh_sub)); pick = np.sort(np.random.default_rng(0).choice(mem, 5000, replace=False))
pos = first_pos.loc[th[pick]].values
A20 = np.load(ROOT / "runs/acts_traffic/layer20.npy", mmap_mode="r")
np.savez(OUT / "gate2_sub.npz", traffic_idx=pick, sub_pos=pos, mix_idx=sub[pos], traffic_l20=np.asarray(A20[pick]))
print("gate 2 pairs", len(pick), "from", len(mem), "traffic prompts with a copy in the subsample", flush=True)

# removal masks restricted to the subsample
keep_names, bits, sizes = [], [], {}
for i, nm in enumerate(names):
    if nm.startswith("famB_"): continue
    m = np.unpackbits(z["bits"][i])[:n].astype(bool); ms = m[sub]
    keep_names.append(nm); bits.append(np.packbits(ms)); sizes[nm] = {"full": int(m.sum()), "sub": int(ms.sum())}
assert {k: sizes[k]["full"] for k in ["safety", "coding", "aya_only", "all_non_english"]} == \
    {"safety": 81727, "coding": 328601, "aya_only": 89119, "all_non_english": 145017}
assert len(keep_names) == 84, len(keep_names)
np.savez_compressed(OUT / "arms_sub.npz", names=np.array(keep_names), bits=np.stack(bits), n=len(sub))
json.dump(sizes, open(OUT / "arm_sizes.json", "w"), indent=1)
for k in ["safety", "coding", "aya_only", "all_non_english"]:
    r = [sizes[f"rand_{k}_{s}"]["sub"] for s in range(20)]
    print(k, "removed in subsample", sizes[k]["sub"], "| randoms", min(r), "-", max(r), flush=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
man = {"seed": SEED, "frac": FRAC, "n_dolci": n, "n_sub": int(len(sub)), "n_gate2": int(len(pick)),
       "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(OUT.iterdir()) if p.name != "manifest.json"}}
json.dump(man, open(OUT / "manifest.json", "w"), indent=1); print(json.dumps(man, indent=1))
