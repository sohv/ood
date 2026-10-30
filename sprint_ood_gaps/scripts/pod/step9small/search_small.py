"""Step 9 small search (designs/step9_small.md), run on the pod after extraction. Same method as scripts/pod/step9_search.py, with the
reference restricted to the 20% Dolci subsample. Changes from step9_search.py, nothing else:
  - Dolci rows = the subsample (sub_idx.npy); features read from feat_dolci/acts_*.npz and placed by searchsorted(sub_idx, idx);
  - query features read from feat_queries/acts_*.npz (extracted by the same acts_dolci.py in this pod) instead of extracted here;
  - removal masks = arms_sub.npz (step 3b's masks restricted to the subsample); part B (famB_*) not run;
  - gate 1 is checked earlier by gate1.py; gate 2 uses gate2_sub.npz (copies inside the subsample) and also reports the kNN distance of
    each traffic vector to the subsample reference against the median query distance (reported, not a stop condition);
  - device and block sizes configurable (DEV, QB, RB) for a CPU smoke test; scores saved as float32 with the query traffic_idx.
Writes out/gates.json (adds gate 2), out/scores_L{8,20,28}.npz (keys "<arm>__k1", "<arm>__m5", "<arm>__k1_fixedstd", "<arm>__m5_fixedstd",
"full__k1", "full__m5", "traffic_idx"), out/search_timing.json. Exit 1 if gate 2 fails (median cosine < 0.999)."""
import glob, json, os, sys, time
import numpy as np, torch
W = os.environ.get("WS", "/workspace"); OUT = f"{W}/out"
DEV = os.environ.get("DEV", "cuda"); DT = torch.float16 if DEV == "cuda" else torch.float32
QB, RB, H = int(os.environ.get("QB", 10_000)), int(os.environ.get("RB", 200_000)), int(os.environ.get("HID", 4096))
LAYERS = [8, 20, 28]
T0 = time.time(); timing = {}
def log(*a): print(time.strftime("%H:%M:%S"), f"+{time.time() - T0:.0f}s", *a, flush=True)

sub = np.load(f"{W}/sub_idx.npy"); n = len(sub)
dfiles = sorted(glob.glob(f"{W}/feat_dolci/acts_*.npz")); qfiles = sorted(glob.glob(f"{W}/feat_queries/acts_*.npz"))
didx = np.concatenate([np.load(f)["idx"] for f in dfiles]); assert len(didx) == n and np.array_equal(np.sort(didx), sub), (len(didx), n)
qz = [np.load(f) for f in qfiles]; q_tidx = np.concatenate([z["idx"] for z in qz]); qf = np.concatenate([z["cmean"] for z in qz])  # [nq, 3, H]
assert np.array_equal(q_tidx, np.array([json.loads(l)["idx"] for l in open(f"{W}/queries.jsonl")]))
del qz; log("queries", qf.shape, "dolci rows", n)

def layer_matrix(j):
    X = torch.empty((n, H), dtype=DT, device=DEV)
    for f in dfiles:
        z = np.load(f); pos = torch.from_numpy(np.searchsorted(sub, z["idx"])).to(DEV)
        X[pos] = torch.from_numpy(z["cmean"][:, j]).to(DEV, DT)
    return X

# gate 2 (layer 20): traffic vectors of prompts verbatim in Dolci vs this pod's vector of their Dolci copy
def cos(a, b): a = torch.nn.functional.normalize(a.float(), dim=1); b = torch.nn.functional.normalize(b.float(), dim=1); return (a * b).sum(1)
gates = json.load(open(f"{OUT}/gates.json")) if os.path.exists(f"{OUT}/gates.json") else {}
X20 = layer_matrix(1); g = np.load(f"{W}/gate2_sub.npz")
T20 = torch.from_numpy(g["traffic_l20"]).to(DEV, DT)
g2 = cos(T20, X20[torch.from_numpy(g["sub_pos"]).to(DEV)]).cpu().numpy()
gates.update({"gate2_n": int(len(g2)), "gate2_median_cos": float(np.median(g2)), "gate2_p01": float(np.percentile(g2, 1)), "gate2_min": float(g2.min())})
json.dump(gates, open(f"{OUT}/gates.json", "w"), indent=1); log("gate 2", gates)
if gates["gate2_median_cos"] < 0.999:
    open(f"{OUT}/GATE_FAILED", "w").close(); sys.exit("gate 2 failed; see gates.json")

# masked kNN, as step9_search.py
z = np.load(f"{W}/arms_sub.npz"); names = [str(x) for x in z["names"]]; bits = z["bits"]; assert int(z["n"]) == n
def mask(i): return torch.from_numpy(np.unpackbits(bits[i])[:n].astype(bool)).to(DEV)
def normed(X, sd):
    # rows divided by the reference std then unit-normalised, in chunks to stay within GPU memory
    Xn = torch.empty_like(X)
    for c in range(0, len(X), 100_000): Xn[c:c + 100_000] = torch.nn.functional.normalize(X[c:c + 100_000].float() / sd, dim=1).to(DT)
    return Xn
def knn(Xn, keep, sd, Q):
    # distance as Sohan's KNN, exact search over kept rows only; returns k = 1 and mean-of-5 distances
    out1, out5 = [], []
    for a in range(0, len(Q), QB):
        Qn = torch.nn.functional.normalize(Q[a:a + QB].float() / sd, dim=1).to(DT)
        best = torch.full((len(Qn), 5), -2.0, device=DEV)
        for c in range(0, n, RB):
            sim = (Qn @ Xn[c:c + RB].T).float(); sim[:, ~keep[c:c + RB]] = -2.0
            best = torch.topk(torch.cat([best, sim], 1), 5, dim=1).values
        d = torch.sqrt(torch.clamp(2 - 2 * best, min=0)); out1.append(d[:, 0].cpu()); out5.append(d.mean(1).cpu())
    return torch.cat(out1).numpy().astype(np.float32), torch.cat(out5).numpy().astype(np.float32)

for j, L in enumerate(LAYERS):
    t = time.time()
    X = X20 if L == 20 else layer_matrix(j); Q = torch.from_numpy(qf[:, j]).to(DEV, DT)
    s1 = torch.zeros(H, device=DEV); s2 = torch.zeros(H, device=DEV)
    for c in range(0, n, 100_000): xc = X[c:c + 100_000].float(); s1 += xc.sum(0); s2 += (xc ** 2).sum(0)
    def std_of(keep):
        r = X[~keep].float() if (~keep).any() else torch.zeros((1, H), device=DEV); m = keep.sum()
        mu = (s1 - r.sum(0)) / m; return torch.sqrt(torch.clamp((s2 - (r ** 2).sum(0)) / m - mu ** 2, min=0)) + 1e-6
    full = torch.ones(n, dtype=torch.bool, device=DEV); sd_full = std_of(full); Xn_full = normed(X, sd_full)
    res = {"traffic_idx": q_tidx}
    res["full__k1"], res["full__m5"] = knn(Xn_full, full, sd_full, Q)
    if L == 20:  # gate 2, second part: distance of the gate traffic vectors to the reference (their copy is in it) vs the median query
        gk1, _ = knn(Xn_full, full, sd_full, T20)
        gates.update({"gate2_median_k1": float(np.median(gk1)), "query_median_k1_L20": float(np.median(res["full__k1"])),
                      "gate2_k1_over_query_k1": float(np.median(gk1) / np.median(res["full__k1"]))})
        json.dump(gates, open(f"{OUT}/gates.json", "w"), indent=1); log("gates", gates)
    for i, nm in enumerate(names):
        keep = ~mask(i)
        res[f"{nm}__k1_fixedstd"], res[f"{nm}__m5_fixedstd"] = knn(Xn_full, keep, sd_full, Q)
        sd = std_of(keep); Xn = normed(X, sd); res[f"{nm}__k1"], res[f"{nm}__m5"] = knn(Xn, keep, sd, Q); del Xn
        if i % 20 == 0: log("layer", L, "arm", i, nm)
    np.savez(f"{OUT}/scores_L{L}.npz", **res); timing[f"L{L}_s"] = round(time.time() - t, 1); log("layer", L, "done")
    del X, Xn_full
    if L == 20: del X20
    if DEV == "cuda": torch.cuda.empty_cache()
json.dump(timing, open(f"{OUT}/search_timing.json", "w"), indent=1)
