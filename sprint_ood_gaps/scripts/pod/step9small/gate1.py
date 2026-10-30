"""Gate 1 of step 9 (designs/one_sided_negatives.md, step 9 revised), run on the pod right after the queries are extracted, before the
Dolci extraction: the 49,629 query layer-20 vectors extracted here must match the existing traffic layer-20 file (runs/acts_traffic/layer20.npy,
extracted on 28 Oct with the same code) at median cosine >= 0.999. Also checks the order of rows and NaN/inf. Writes out/gates.json; exit 1 on fail."""
import glob, json, os, sys
import numpy as np
W = os.environ.get("WS", "/workspace")
files = sorted(glob.glob(f"{W}/feat_queries/acts_*.npz")); zs = [np.load(f) for f in files]
idx = np.concatenate([z["idx"] for z in zs]); q20 = np.concatenate([z["cmean"][:, 1] for z in zs]).astype(np.float32)  # layers [8, 20, 28]
want = np.array([json.loads(l)["idx"] for l in open(f"{W}/queries.jsonl")])
assert np.array_equal(idx, want), "query order differs from queries.jsonl"
E = np.load(f"{W}/q_l20_existing.npy").astype(np.float32)
cos = (q20 * E).sum(1) / (np.linalg.norm(q20, axis=1) * np.linalg.norm(E, axis=1) + 1e-12)
allf = np.concatenate([z["cmean"].reshape(len(z["cmean"]), -1) for z in zs])
g = {"gate1_n": int(len(cos)), "gate1_median_cos": float(np.median(cos)), "gate1_p01": float(np.percentile(cos, 1)), "gate1_min": float(cos.min()),
     "gate1_share_below_0.99": float((cos < 0.99).mean()), "query_nonfinite_rows": int((~np.isfinite(allf)).any(1).sum()),
     "query_n_content_tokens_median": float(np.median(np.concatenate([z["n_content_tokens"] for z in zs]))),
     "query_tokens_total": int(sum(z["n_tokens"].sum() for z in zs))}
json.dump(g, open(f"{W}/out/gates.json", "w"), indent=1); print(g, flush=True)
sys.exit(0 if g["gate1_median_cos"] >= 0.999 and g["query_nonfinite_rows"] == 0 else 1)
