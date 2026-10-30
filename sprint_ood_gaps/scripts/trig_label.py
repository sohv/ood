"""Trigger scan step 1 (designs/triggers_scan.md): stream Dolci SFT parquet files then DPO, keep the first occurrence of each
normalised first user turn, label its first assistant reply (DPO: chosen) with scripts/refusal.py. Writes
runs/triggers/dolci_labelled/part_XX.parquet with columns hash, part, source, refused, prompt (first 3000 chars), reply (first 400)."""
import glob, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from refusal import is_refusal
from neg_index import normalise, text_hash
OUT = ROOT / "runs/triggers/dolci_labelled"; OUT.mkdir(parents=True, exist_ok=True)
HUB = Path.home() / ".cache/huggingface/hub"
sft = sorted(glob.glob(str(HUB / "datasets--allenai--Dolci-Instruct-SFT/snapshots/*/data/*.parquet")))
dpo = sorted(glob.glob(str(HUB / "datasets--allenai--Dolci-Instruct-DPO/snapshots/*/data/*.parquet")))
print(len(sft), "sft files", len(dpo), "dpo files", flush=True)
seen, n_tot = set(), 0
def first(msgs, role): return next((m["content"] for m in msgs if m["role"] == role), None)
for k, (part, p) in enumerate([("sft", p) for p in sft] + [("dpo", p) for p in dpo]):
    d = pd.read_parquet(p)
    msgs = d["messages"] if part == "sft" else d["chosen"]
    src = d["source_dataset"] if "source_dataset" in d else (d["source"] if "source" in d else pd.Series(["dpo"] * len(d)))
    rows = []
    for m, s in zip(msgs, src):
        u = first(m, "user")
        if u is None: continue
        n_tot += 1; h = text_hash(normalise(u))
        if h in seen: continue
        seen.add(h); a = first(m, "assistant") or ""
        rows.append((h, part, s, is_refusal(a), u[:3000], a[:400]))
    pd.DataFrame(rows, columns=["hash", "part", "source", "refused", "prompt", "reply"]).to_parquet(OUT / f"part_{k:02d}.parquet")
    print(part, Path(p).name, len(d), "kept", len(rows), "unique so far", len(seen), flush=True); del d, msgs
print("total rows with a user turn", n_tot, "unique", len(seen))
