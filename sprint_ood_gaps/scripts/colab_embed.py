# Runs on the Colab GPU: embed gpu_texts_from_*.jsonl.gz with the same MiniLM revision as Sohan's build, in 50k chunks,
# each chunk saved as emb_<start>.npy (float16, unit norm) so partial results survive a vanishing runtime.
import glob, gzip, json, os, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer

src = sorted(glob.glob("gpu_texts_from_*.jsonl.gz"))[0]
texts = [json.loads(l)["text"] for l in gzip.open(src, "rt")]
print("texts", len(texts), flush=True)
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cuda", revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
os.makedirs("out", exist_ok=True)
t0 = time.time()
for s in range(0, len(texts), 50000):
    p = f"out/emb_{s:07d}.npy"
    if os.path.exists(p):
        continue
    e = m.encode(texts[s:s + 50000], batch_size=512, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    np.save(p + ".tmp.npy", e); os.replace(p + ".tmp.npy", p)
    print(f"{s + len(e)}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
open("out/DONE", "w").write(str(len(texts)))
print("done", flush=True)
