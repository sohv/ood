"""Inputs for the step 9 (revised) search pod, built on research-box (designs/one_sided_negatives.md step 9 revised).
Writes runs/step9/inputs/: queries.jsonl (49,629 traffic prompts of step 3b, for re-extraction), q_l20_existing.npy,
gate2.npz (5,000 traffic prompts verbatim in Dolci: their existing traffic layer-20 vectors and their Dolci row),
arms.npz (removal masks, packed bits: the 4 kinds of step 3b and 20 randoms each, same definitions and seeds; part B family arms),
partb.npz (step 1 query rows and labels per family)."""
import glob, json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/step9/inputs"; OUT.mkdir(parents=True, exist_ok=True)
q = pd.read_parquet(ROOT / "runs/removal_3b/queries.parquet")[["traffic_idx", "topic", "lang", "non_en"]]
texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
with open(OUT / "queries.jsonl", "w") as f:
    for i in q.traffic_idx: f.write(json.dumps({"idx": int(i), "text": texts[i]}) + "\n")
A20 = np.load(ROOT / "runs/acts_traffic/layer20.npy", mmap_mode="r")
np.save(OUT / "q_l20_existing.npy", np.asarray(A20[q.traffic_idx.values]))
# gate 2: traffic prompts that are verbatim Dolci members
th = np.load(IDX / "traffic_hash.npy"); mh = np.load(IDX / "mix_hash.npy")
first = pd.Series(np.arange(len(mh))).groupby(mh).first()
mem = np.flatnonzero(np.isin(th, mh)); pick = np.sort(np.random.default_rng(0).choice(mem, 5000, replace=False))
np.savez(OUT / "gate2.npz", traffic_idx=pick, mix_idx=first.loc[th[pick]].values, traffic_l20=np.asarray(A20[pick]))

# arms: identical to scripts/removal_3b.py
src_txt = open(ROOT / "scripts/one_sided_score.py").read()
FAMILIES = eval(src_txt[src_txt.index("FAMILIES = {") + len("FAMILIES = "):src_txt.index("\n}\n") + 2])
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)
def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
src = np.where(meta.part.values == "sft", sft["source_dataset"].reindex(meta.row.values).values, None).astype(object)
dm = meta.part.values == "dpo"; src[dm] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[dm]]
dlang = np.load(ROOT / "runs/removal_3b/lang_dolci.npy", allow_pickle=True); n = len(mh)
kinds = {"safety": np.isin(src, FAMILIES["safety"]), "coding": np.isin(src, FAMILIES["code"]),
         "aya_only": np.isin(src, FAMILIES["multilingual"]), "all_non_english": dlang != "en"}
for k in kinds: kinds[k] |= np.isin(mh, mh[kinds[k]])
arms = {}
for k, r in kinds.items():
    arms[k] = r
    for s in range(20):
        m = np.zeros(n, bool); m[np.random.default_rng(1000 * len(k) + s).choice(n, r.sum(), replace=False)] = True; arms[f"rand_{k}_{s}"] = m
# check against step 3b's saved query scores: the MiniLM safety score recomputed elsewhere is not needed; check mask sizes instead
assert {k: int(v.sum()) for k, v in kinds.items()} == {"safety": 81727, "coding": 328601, "aya_only": 89119, "all_non_english": 145017}, {k: int(v.sum()) for k, v in kinds.items()}
# part B: step 1's family arms and query rows (rows_*.parquet), reference = all minus family rows minus held-out negatives, with copies
fam_of = np.array([next((f for f, v in FAMILIES.items() if s in v), "") for s in src], dtype=object)
negs = np.unique(np.concatenate([pd.read_parquet(p).query("label == 0").mix_idx.values for p in glob.glob(str(ROOT / "runs/one_sided/rows_*.parquet"))]))
neg_hash = set(mh[negs].tolist()); partb = {}
for p in sorted(glob.glob(str(ROOT / "runs/one_sided/rows_*.parquet"))):
    f = Path(p).stem[5:]; r = pd.read_parquet(p)
    drop = set(mh[fam_of == f].tolist()) | neg_hash
    arms[f"famB_{f}"] = np.isin(mh, list(drop)); partb[f"{f}_rows"] = r.mix_idx.values; partb[f"{f}_label"] = r.label.values
np.savez_compressed(OUT / "arms.npz", names=np.array(list(arms)), bits=np.stack([np.packbits(v) for v in arms.values()]), n=n)
np.savez(OUT / "partb.npz", **partb)
print({k: int(v.sum()) for k, v in arms.items() if not k.startswith("rand_")}, len(arms), "arms")
