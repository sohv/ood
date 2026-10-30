"""One-minute pilot: MiniLM (paraphrase-multilingual-MiniLM-L12-v2) CPU throughput on 2,000 cached WildChat first turns,
to size a re-embedding of ~2.7M prompts. Writes runs/negatives/pilot/throughput.json."""
import time, json, torch
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT = Path(__file__).resolve().parents[1] / "runs/negatives/pilot"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
texts = []
for row in ds:
    conv = row.get("conversation") or []
    if conv and conv[0].get("role") == "user" and row.get("language") == "English":
        texts.append(conv[0]["content"][:2000])
    if len(texts) >= 2000: break
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
res = {}
for max_len in (128, 256):
    model.max_seq_length = max_len
    model.encode(texts[:64], batch_size=64)  # warm-up
    t0 = time.time()
    emb = model.encode(texts, batch_size=128, show_progress_bar=False)
    dt = time.time() - t0
    res[f"max_seq_{max_len}"] = {"n": len(texts), "seconds": round(dt, 1), "per_sec": round(len(texts) / dt, 1),
                                "hours_for_2.7M": round(2.7e6 / (len(texts) / dt) / 3600, 2), "dim": int(emb.shape[1])}
    print(res[f"max_seq_{max_len}"], flush=True)
(OUT / "throughput.json").write_text(json.dumps(res, indent=1))
