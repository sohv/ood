"""Stitch traffic.npy = [old pool embeddings (keep_old)] + [cpu partial new[:START]] + [gpu new[START:]].
Needs: split.json, traffic_rows.jsonl, traffic_hash.npy (from neg_traffic_split.py), traffic_new_partial.npy (cpu job,
>= START rows), runs/negatives/index/gpu_out/emb_*.npy (from colab). Rebuilds the keep_old selection exactly as neg_index does."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neg_index import ROOT, EXP09, EXP31, EMB_V2, normalise, load_jsonl, log

OUT = ROOT / "runs/negatives/index"
sp = json.load(open(OUT / "split.json")); START, n_old, n_new = sp["start"], sp["n_old"], sp["n_new"]
c9 = json.loads((EXP09 / "config.json").read_text()); CLONE = ROOT.parent
pool9 = [r["text"] for r in load_jsonl(CLONE / c9["wildchat_pool"])] + [r["text"] for r in load_jsonl(CLONE / c9["lmsys_pool"])]
ext = [r["text"] for r in load_jsonl(EXP31 / "traffic_ext.jsonl")]
e_old_all = np.vstack([np.load(EMB_V2 / "emb_traffic.npy").astype(np.float16), np.load(EMB_V2 / "emb_traffic_ext.npy")])
seen, keep_old = set(), []
for i, t in enumerate(pool9 + ext):
    nm = normalise(t)
    if nm and nm not in seen:
        seen.add(nm); keep_old.append(i)
assert len(keep_old) == n_old, (len(keep_old), n_old)
cpu = np.load(OUT / "traffic_new_partial.npy")[:START]
assert len(cpu) == START, f"cpu partial has {len(np.load(OUT / 'traffic_new_partial.npy'))} rows, need {START}"
gpu_files = sorted(glob.glob(str(OUT / "gpu_out/emb_*.npy")))
gpu = np.vstack([np.load(f) for f in gpu_files]) if gpu_files else np.zeros((0, 384), np.float16)
assert len(gpu) == n_new - START, (len(gpu), n_new - START)
t = np.vstack([e_old_all[keep_old], cpu.astype(np.float16), gpu.astype(np.float16)])
assert len(t) == 992933 == len(np.load(OUT / "traffic_hash.npy"))
np.save(OUT / "traffic.npy", t)
log("traffic.npy saved", t.shape, "| norms of 3 gpu rows:", np.linalg.norm(gpu[:3].astype(np.float32), axis=1) if len(gpu) else None)
