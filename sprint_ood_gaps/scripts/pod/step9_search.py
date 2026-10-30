"""Step 9 (revised) search, run on one GPU pod. Inputs in /workspace/in (from scripts/step9_prepare.py), Dolci features in
/workspace/dolci/shard*/acts_*.npz (layers 8/20/28), query features extracted here with the same extractor.
1) extract query features; 2) gate 1 (query layer 20 vs the existing traffic file) and gate 2 (traffic vs Dolci copies); 3) for each layer
and arm, exact masked kNN (k = 1, mean of 5) with std from the arm's own reference (primary) and from full Dolci (ablation).
Writes /workspace/out/{gates.json, scores_L<layer>.npz, partb_L<layer>.npz, DONE}."""
import glob, json, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from features_sohan import encode_chat, extract
MODEL, REV, LAYERS = "allenai/Olmo-3-7B-Instruct-DPO", "b33130b7", [8, 20, 28]
IN, OUT = "/workspace/in", "/workspace/out"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# 1) query features, same code path as the Dolci shards (acts_dolci.py)
rows = [json.loads(l) for l in open(f"{IN}/queries.jsonl")]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).cuda().eval()
enc = [encode_chat(tok, r["text"], max_length=512) for r in rows]
enc = [(ids, m if any(m) else [False] * (len(m) - 1) + [True]) for ids, m in enc]
qf = extract(model, enc, tok.pad_token_id or 0, max_tokens=16384, layers=LAYERS)["cmean"]  # [nq, 3, 4096] fp16
del model; torch.cuda.empty_cache(); np.save(f"{OUT}/q_feats.npy", qf); log("queries extracted", qf.shape)

# Dolci features, assembled per layer in mix-row order
files = sorted(glob.glob("/workspace/dolci/shard*/acts_*.npz")); n = int(np.load(f"{IN}/arms.npz")["n"])
idx_all = np.concatenate([np.load(f)["idx"] for f in files]); assert len(idx_all) == n and len(np.unique(idx_all)) == n, (len(idx_all), n)
def layer_matrix(j):
    X = torch.empty((n, 4096), dtype=torch.float16, device="cuda")
    for f in files:
        z = np.load(f); X[torch.from_numpy(z["idx"]).cuda()] = torch.from_numpy(z["cmean"][:, j]).cuda()
    return X

# 2) gates (layer 20)
def cos(a, b): a = torch.nn.functional.normalize(a.float(), dim=1); b = torch.nn.functional.normalize(b.float(), dim=1); return (a * b).sum(1)
g1 = cos(torch.from_numpy(qf[:, 1]).cuda(), torch.from_numpy(np.load(f"{IN}/q_l20_existing.npy")).cuda()).cpu().numpy()
X20 = layer_matrix(1); g = np.load(f"{IN}/gate2.npz")
g2 = cos(torch.from_numpy(g["traffic_l20"]).cuda(), X20[torch.from_numpy(g["mix_idx"]).cuda()]).cpu().numpy()
gates = {"gate1_median_cos": float(np.median(g1)), "gate1_p01": float(np.percentile(g1, 1)), "gate2_median_cos": float(np.median(g2)), "gate2_p01": float(np.percentile(g2, 1))}
json.dump(gates, open(f"{OUT}/gates.json", "w"), indent=1); log("gates", gates); del X20
if gates["gate1_median_cos"] < 0.999 or gates["gate2_median_cos"] < 0.999:
    open(f"{OUT}/GATE_FAILED", "w").close(); raise SystemExit("gate failed; see gates.json")

# 3) masked kNN
z = np.load(f"{IN}/arms.npz"); names = list(z["names"]); bits = z["bits"]
def mask(i): return torch.from_numpy(np.unpackbits(bits[i])[:n].astype(bool)).cuda()
pb = np.load(f"{IN}/partb.npz")
def normed(X, sd):
    # rows divided by the reference std then unit-normalised, in chunks to stay within GPU memory
    Xn = torch.empty_like(X)
    for c in range(0, len(X), 100_000): Xn[c:c + 100_000] = torch.nn.functional.normalize(X[c:c + 100_000].float() / sd, dim=1).half()
    return Xn
def knn(Xn, keep, sd, Q):
    # distance as Sohan's KNN, exact search over kept rows only; returns k = 1 and mean-of-5 distances
    out1, out5 = [], []
    for a in range(0, len(Q), 10_000):
        Qn = torch.nn.functional.normalize(Q[a:a + 10_000].float() / sd, dim=1).half()
        best = torch.full((len(Qn), 5), -2.0, device="cuda")
        for c in range(0, n, 200_000):
            sim = (Qn @ Xn[c:c + 200_000].T).float(); sim[:, ~keep[c:c + 200_000]] = -2.0
            best = torch.topk(torch.cat([best, sim], 1), 5, dim=1).values
        d = torch.sqrt(torch.clamp(2 - 2 * best, min=0)); out1.append(d[:, 0].cpu()); out5.append(d.mean(1).cpu())
    return torch.cat(out1).numpy(), torch.cat(out5).numpy()
for j, L in enumerate(LAYERS):
    X = layer_matrix(j); Q = torch.from_numpy(qf[:, j]).cuda()
    s1 = torch.zeros(4096, device="cuda"); s2 = torch.zeros(4096, device="cuda")
    for c in range(0, n, 100_000): xc = X[c:c + 100_000].float(); s1 += xc.sum(0); s2 += (xc ** 2).sum(0)
    def std_of(keep):
        r = X[~keep].float() if (~keep).any() else torch.zeros((1, 4096), device="cuda"); m = keep.sum()
        mu = (s1 - r.sum(0)) / m; return torch.sqrt(torch.clamp((s2 - (r ** 2).sum(0)) / m - mu ** 2, min=0)) + 1e-6
    full = torch.ones(n, dtype=torch.bool, device="cuda"); sd_full = std_of(full); Xn_full = normed(X, sd_full)
    res = {}
    res["full__k1"], res["full__m5"] = knn(Xn_full, full, sd_full, Q)
    for i, nm in enumerate(names):
        if nm.startswith("famB_"): continue
        keep = ~mask(i)
        res[f"{nm}__k1_fixedstd"], res[f"{nm}__m5_fixedstd"] = knn(Xn_full, keep, sd_full, Q)
        sd = std_of(keep); Xn = normed(X, sd); res[f"{nm}__k1"], res[f"{nm}__m5"] = knn(Xn, keep, sd, Q); del Xn
    np.savez(f"{OUT}/scores_L{L}.npz", **res); log("layer", L, "removal arms done")
    resb = {}
    for i, nm in enumerate(names):
        if not nm.startswith("famB_"): continue
        f = nm[5:]; keep = ~mask(i); rq = torch.from_numpy(pb[f"{f}_rows"]).cuda()
        sd = std_of(keep); Xn = normed(X, sd); resb[f"{f}__k1"], resb[f"{f}__m5"] = knn(Xn, keep, sd, X[rq]); del Xn
    np.savez(f"{OUT}/partb_L{L}.npz", **resb); log("layer", L, "part B done"); del X, Xn_full
open(f"{OUT}/DONE", "w").close()
