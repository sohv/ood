# Runs on the Colab GPU. One-sided "rare gap" score for every traffic prompt: similarity to the nearest Dolci prompts,
# computed in float32 with TF32 off (the float16 path is what produced the tie artefact in exp37), exact copies
# (same normalised-text hash) excluded. Saves top-10 Dolci similarities and the nearest Dolci index per row.
import time, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
t0 = time.time()
m = torch.from_numpy(np.load("mix.npy").astype(np.float32)).cuda()
mh = torch.from_numpy(np.load("mix_hash.npy")).cuda()
t = np.load("traffic.npy"); th = np.load("traffic_hash.npy")
print("loaded", m.shape, t.shape, f"{time.time() - t0:.0f}s", flush=True)
K, EXTRA, CH = 10, 10, 4096
top_s = np.empty((len(t), K), np.float32); top_i = np.empty((len(t), K), np.int64)
for s in range(0, len(t), CH):
    q = torch.from_numpy(t[s:s + CH].astype(np.float32)).cuda(); qh = torch.from_numpy(th[s:s + CH]).cuda()[:, None]
    sim, idx = (q @ m.T).topk(K + EXTRA, dim=1)
    sim = sim.masked_fill(mh[idx] == qh, -float("inf"))
    ss, order = sim.topk(K, dim=1)
    top_s[s:s + CH] = ss.cpu().numpy(); top_i[s:s + CH] = idx.gather(1, order).cpu().numpy()
    if (s // CH) % 50 == 0:
        print(f"{s + len(q)}/{len(t)} {time.time() - t0:.0f}s", flush=True)
np.savez_compressed("one_sided.npz", top_s=top_s, top_i=top_i)
print("done", f"{time.time() - t0:.0f}s", flush=True)
