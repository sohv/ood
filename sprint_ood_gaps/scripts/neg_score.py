"""exp37's k=50 neighbour ratio on CPU with faiss, same arithmetic as ../src/metrics/gap_pipeline.py:density_ratios_batched.

  ratio = ((n_traffic + 1) / n_t) / ((n_mix + 1) / n_m) over the 50 nearest valid neighbours in traffic + mix,
  where neighbours that are the query itself (traffic index q_self) or exact copies (equal blake2b hash of the
  normalised text) are left out; k + extra = 70 candidates are taken from each side before masking, as he does.

Modes:
  --validate N   score N traffic rows (seed 0) with q_self = their index and compare with his exp37 v2 ratio.npy
  --query FILE   score a jsonl of {"text": ...} rows (optionally "emb" precomputed) and write per-row records
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "runs/negatives/index"
K, EXTRA = 50, 20


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(norm: str) -> int:
    return int.from_bytes(hashlib.blake2b(norm.encode(), digest_size=8).digest(), "little") >> 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Scorer:
    def __init__(self, threads: int = 8):
        faiss.omp_set_num_threads(threads)
        self.t = np.load(IDX / "traffic.npy").astype(np.float32)
        self.m = np.load(IDX / "mix.npy").astype(np.float32)
        self.t_hash = np.load(IDX / "traffic_hash.npy")
        self.m_hash = np.load(IDX / "mix_hash.npy")
        self.n_t, self.n_m = len(self.t), len(self.m)
        log(f"index: traffic {self.n_t}, mix {self.n_m}, dim {self.t.shape[1]}")
        self.it = faiss.IndexFlatIP(self.t.shape[1]); self.it.add(self.t)
        self.im = faiss.IndexFlatIP(self.m.shape[1]); self.im.add(self.m)
        log("faiss flat indexes built")

    def score(self, q: np.ndarray, q_hash: np.ndarray, q_self: np.ndarray, chunk: int = 2048):
        """Returns ratio, n_traffic, n_mix, nearest_mix_sim, nearest_mix_idx (after masking), as his code defines them."""
        q = q.astype(np.float32)
        n = len(q)
        ratio = np.empty(n); n_tr = np.empty(n, dtype=np.int32); n_mx = np.empty(n, dtype=np.int32)
        ratio32 = np.empty(n); n_tr32 = np.empty(n, dtype=np.int32)
        near = np.empty(n); near_i = np.empty(n, dtype=np.int64)
        near_t = np.empty(n); near_ti = np.empty(n, dtype=np.int64)
        for s in range(0, n, chunk):
            qq = q[s:s + chunk]; qh = q_hash[s:s + chunk][:, None]; qs = q_self[s:s + chunk][:, None]
            st, it = self.it.search(qq, K + EXTRA)
            sm, im = self.im.search(qq, K + EXTRA)
            st = np.where((it == qs) | (self.t_hash[it] == qh), -np.inf, st)
            sm = np.where(self.m_hash[im] == qh, -np.inf, sm)
            near[s:s + chunk] = sm.max(1)
            near_i[s:s + chunk] = im[np.arange(len(sm)), sm.argmax(1)]
            near_t[s:s + chunk] = st.max(1)
            near_ti[s:s + chunk] = it[np.arange(len(st)), st.argmax(1)]
            side = np.concatenate([np.zeros_like(st), np.ones_like(sm)], axis=1)
            for prec in ("fp16", "fp32"):
                # fp16: his GPU path rounds sims to float16 before the union top-k, traffic first on ties (validated
                # 19,981 of 20,000 rows against his ratio.npy); fp32: full precision
                a, b = (st.astype(np.float16).astype(np.float32), sm.astype(np.float16).astype(np.float32)) if prec == "fp16" else (st, sm)
                sims = np.concatenate([a, b], axis=1)
                top_i = np.argsort(-sims, axis=1, kind="stable")[:, :K]
                valid = np.isfinite(np.take_along_axis(sims, top_i, axis=1))
                nm = (np.take_along_axis(side, top_i, axis=1) * valid).sum(1)
                nt = valid.sum(1) - nm
                r = ((nt + 1) / self.n_t) / ((nm + 1) / self.n_m)
                if prec == "fp16":
                    ratio[s:s + chunk] = r; n_tr[s:s + chunk] = nt; n_mx[s:s + chunk] = nm
                else:
                    ratio32[s:s + chunk] = r; n_tr32[s:s + chunk] = nt
        self.last_fp32 = (ratio32, n_tr32)
        return ratio, n_tr, n_mx, near, near_i, near_t, near_ti


def validate(n: int, seed: int = 0):
    his = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/ratio.npy")
    sc = Scorer()
    assert len(his) == sc.n_t, f"his ratio.npy has {len(his)} rows, our traffic has {sc.n_t}"
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(sc.n_t, size=n, replace=False))
    t0 = time.time()
    ratio, nt, nm, near, _, _, _ = sc.score(sc.t[idx], sc.t_hash[idx], idx)
    log(f"scored {n} rows in {time.time() - t0:.1f}s")
    d = np.abs(ratio - his[idx]); rel = d / np.maximum(his[idx], 1e-9)
    out = {"n": n, "max_abs_diff": float(d.max()), "share_rel_diff_lt_1e-3": float((rel < 1e-3).mean()),
           "share_exact_1e-6": float((d < 1e-6).mean()), "spearman": float(np.corrcoef(np.argsort(np.argsort(ratio)), np.argsort(np.argsort(his[idx])))[0, 1]),
           "his_median": float(np.median(his[idx])), "our_median": float(np.median(ratio)), "worst_rows": [int(i) for i in idx[np.argsort(-d)[:10]]],
           "worst_pairs": [(float(a), float(b)) for a, b in zip(ratio[np.argsort(-d)[:10]], his[idx][np.argsort(-d)[:10]])]}
    (IDX / "validation.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out, indent=1))


def run_query(path: str, out_path: str, sc=None):
    rows = [json.loads(l) for l in open(path)]
    sc = sc or Scorer()
    if "emb" in rows[0]:
        q = np.array([r["emb"] for r in rows], dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu",
                                  revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
        q = emb.encode([r["text"][:1000] for r in rows], batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    # covered anchors carry mix_idx: use the stored hash of the original (untruncated) Dolci text, so the row's own copy
    # and its exact duplicates are masked (fixes the 4,000-char truncation bug found in review, 2026-10-23)
    qh = np.array([int(sc.m_hash[r["mix_idx"]]) if r.get("mix_idx") is not None else text_hash(normalise(r["text"])) for r in rows], dtype=np.int64)
    qs = np.array([r.get("traffic_idx", -1) for r in rows], dtype=np.int64)
    ratio, nt, nm, near, near_i, near_t, near_ti = sc.score(q, qh, qs)
    r32, n32 = sc.last_fp32
    mix_hash_set = set(sc.m_hash.tolist()); traffic_hash_set = set(sc.t_hash.tolist())
    with open(out_path, "w") as f:
        for len_done, (r, a, b, c, d, e, g, h) in enumerate(zip(rows, ratio, nt, nm, near, near_i, near_t, near_ti)):
            r = {k: v for k, v in r.items() if k != "emb"}
            hsh = int(sc.m_hash[r["mix_idx"]]) if r.get("mix_idx") is not None else text_hash(normalise(r["text"]))
            r.update({"ratio": float(a), "n_traffic": int(b), "n_mix": int(c), "nearest_mix_sim": float(d), "nearest_mix_idx": int(e),
                      "ratio_fp32": float(r32[len_done]), "n_traffic_fp32": int(n32[len_done]),
                      "nearest_traffic_sim": float(g), "nearest_traffic_idx": int(h),
                      "exact_in_mix": hsh in mix_hash_set, "exact_in_traffic": hsh in traffic_hash_set})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("wrote", out_path, len(rows), "rows; median ratio", float(np.median(ratio)))


def run_many(paths, out_dir, sc=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    global Scorer_singleton
    for p in paths:
        out = out_dir / Path(p).name
        if out.exists():
            log("exists", out.name); continue
        run_query(p, str(out), sc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--query"); ap.add_argument("--out")
    ap.add_argument("--many", nargs="*"); ap.add_argument("--out_dir", default=str(ROOT / "runs/negatives/scores"))
    a = ap.parse_args()
    if a.many:
        run_many(a.many, a.out_dir, Scorer())
    if a.validate: validate(a.validate)
    if a.query: run_query(a.query, a.out)
