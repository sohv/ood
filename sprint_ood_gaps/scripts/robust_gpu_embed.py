# Runs on the GPU pod in /workspace/robust: embed traffic.txt.gz and mix.txt.gz with Qwen3-Embedding-0.6B (512 tokens,
# no instruction prefix, unit norm), 100k-row chunks saved as out/<name>_<start>.npy (float16) so partial results survive.
# Then the same one-sided search as scripts/colab_one_sided.py (float32, TF32 off, exact-hash copies excluded) -> qwen_one_sided.npz.
import glob, gzip, json, os, sys, time
import numpy as np, torch
from sentence_transformers import SentenceTransformer
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
MODEL, REV = "Qwen/Qwen3-Embedding-0.6B", os.environ.get("QREV")
CH = 100_000; t0 = time.time(); os.makedirs("out", exist_ok=True)
def log(*a): print(f"{time.time() - t0:7.0f}s", *a, flush=True)
probe = len(sys.argv) > 1 and sys.argv[1] == "probe"
# worker mode: "embed W NW" embeds chunks j with j % NW == W over the joint list (traffic then mix) and exits; "search" only searches
mode = sys.argv[1] if len(sys.argv) > 1 else "all"; W, NW = (int(sys.argv[2]), int(sys.argv[3])) if mode == "embed" else (0, 1)
if mode != "search":
  m = SentenceTransformer(MODEL, device="cuda", revision=REV, model_kwargs={"torch_dtype": torch.bfloat16}, tokenizer_kwargs={"padding_side": "left"})
  m.max_seq_length = 512
  log("model", MODEL, "revision", REV)
J = 0
for name in (("traffic", "mix") if mode != "search" else ()):
    texts = [json.loads(l) for l in gzip.open(f"{name}.txt.gz", "rt")]
    log(name, len(texts))
    if probe:
        s = time.time(); e = m.encode(texts[:20000], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
        log(f"probe {name}: 20000 rows {time.time() - s:.0f}s, norm {np.linalg.norm(e[:3].astype(np.float32), axis=1)}"); continue
    for s in range(0, len(texts), CH):
        p = f"out/{name}_{s:07d}.npy"; J += 1
        if (J - 1) % NW != W or os.path.exists(p): continue
        e = m.encode(texts[s:s + CH], batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
        np.save(p + ".tmp.npy", e); os.replace(p + ".tmp.npy", p); log(f"{name} {s + len(e)}/{len(texts)}")
if probe or mode == "embed": sys.exit()
if mode != "search": del m
torch.cuda.empty_cache()
t = np.vstack([np.load(p) for p in sorted(glob.glob("out/traffic_*.npy"))]); mix = np.vstack([np.load(p) for p in sorted(glob.glob("out/mix_*.npy"))])
th = np.load("traffic_hash.npy"); mh_np = np.load("mix_hash.npy"); assert len(t) == len(th) and len(mix) == len(mh_np), (t.shape, mix.shape)
np.save("traffic_qwen.npy", t); np.save("mix_qwen.npy", mix)
M = torch.from_numpy(mix.astype(np.float32)).cuda(); mh = torch.from_numpy(mh_np).cuda()
K, EXTRA, QC = 10, 10, 1024
top_s = np.empty((len(t), K), np.float32); top_i = np.empty((len(t), K), np.int64)
for s in range(0, len(t), QC):
    q = torch.from_numpy(t[s:s + QC].astype(np.float32)).cuda(); qh = torch.from_numpy(th[s:s + QC]).cuda()[:, None]
    sim, idx = (q @ M.T).topk(K + EXTRA, dim=1)
    sim = sim.masked_fill(mh[idx] == qh, -float("inf")); ss, order = sim.topk(K, dim=1)
    top_s[s:s + QC] = ss.cpu().numpy(); top_i[s:s + QC] = idx.gather(1, order).cpu().numpy()
    if (s // QC) % 200 == 0: log(f"search {s + len(q)}/{len(t)}")
np.savez_compressed("qwen_one_sided.npz", top_s=top_s, top_i=top_i); open("DONE", "w").write("ok"); log("done")
