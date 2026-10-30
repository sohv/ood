"""Fix-loop data prep (designs/fix_loop_name1.md), CPU. Candidate prompts for the fix and control arms, the eval file.
fix candidates: LMSYS traffic first turns with NAME_k, not in the placeholder_overrefusal probe, MiniLM cosine < 0.90 to every probe
placeholder prompt, deduplicated; 1,600 sampled (seed 0) with a real-name version (names distinct from the probe's).
control candidates: 1,600 LMSYS traffic first turns without placeholder tokens (seed 0).
eval: the 456 probe prompts + HarmBench standard (200) + JBB harmful (100)."""
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/fix_loop/data"
NAMES = ["Priya", "Mateo", "Ingrid", "Kofi", "Yuki", "Lars", "Amara", "Tomas", "Leila", "Rafael", "Sanne", "Dmitri"]
PAT = re.compile(r"NAME_(\d+)")
rows = [json.loads(l) for l in open(IDX / "traffic_rows.jsonl")]
t = np.load(IDX / "traffic.npy", mmap_mode="r")
probe = [json.loads(l) for l in open(ROOT.parent / "data/processed/placeholder_overrefusal/probe.jsonl")]
ptext = {p["text"] for p in probe}
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu", revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
pe = emb.encode([p["text"][:1000] for p in probe if p["variant"] == "placeholder"], normalize_embeddings=True)
lm = [i for i, r in enumerate(rows) if r["src"] in ("lmsys", "lmsys_ext")]
has = [i for i in lm if PAT.search(rows[i]["text"]) and rows[i]["text"] not in ptext and len(rows[i]["text"]) <= 4000]
no = [i for i in lm if not re.search(r"NAME_\d|PERSON_\d|\[NAME\]|<NAME>", rows[i]["text"]) and len(rows[i]["text"]) <= 4000]
sims = np.asarray(t[has], dtype=np.float32) @ pe.T
ok = [has[k] for k in range(len(has)) if sims[k].max() < 0.90]
print(f"LMSYS rows {len(lm):,}; with NAME_k {len(has):,}; after probe cosine filter {len(ok):,}; without placeholders {len(no):,}")
rng = np.random.default_rng(0)
fix = rng.choice(ok, 1600, replace=False); ctl = rng.choice(no, 1600, replace=False)
def to_names(s):
    return PAT.sub(lambda m: NAMES[(int(m.group(1)) - 1) % len(NAMES)], s)
with open(OUT / "fix_candidates.jsonl", "w") as f:
    for i in fix:
        s = rows[i]["text"]; ks = sorted({int(k) for k in PAT.findall(s)})
        f.write(json.dumps({"id": f"fix_{i}", "traffic_idx": int(i), "placeholder": s, "named": to_names(s),
                            "map": {NAMES[(k - 1) % len(NAMES)]: f"NAME_{k}" for k in ks}}, ensure_ascii=False) + "\n")
with open(OUT / "control_candidates.jsonl", "w") as f:
    for i in ctl:
        f.write(json.dumps({"id": f"ctl_{i}", "traffic_idx": int(i), "prompt": rows[i]["text"]}, ensure_ascii=False) + "\n")
ev = [{"id": p["id"], "set": "probe_" + p["variant"], "prompt": p["text"]} for p in probe]
from datasets import load_dataset
ev += [{"id": f"harmbench_{k}", "set": "harmbench", "prompt": r["prompt"]} for k, r in enumerate(load_dataset("walledai/HarmBench", "standard", split="train"))]
ev += [{"id": f"jbb_{k}", "set": "jbb", "prompt": r["Goal"]} for k, r in enumerate(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"))]
with open(OUT / "eval.jsonl", "w") as f:
    for e in ev: f.write(json.dumps(e, ensure_ascii=False) + "\n")
import collections; print("eval sets:", collections.Counter(e["set"] for e in ev))
print("example fix:", json.dumps(json.loads(open(OUT / "fix_candidates.jsonl").readline()), ensure_ascii=False)[:400])
