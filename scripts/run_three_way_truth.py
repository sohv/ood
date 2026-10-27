# exp68: makes 10 topics deployment-only and 10 training-only by deleting their look-alikes, keeps 10 shared, and scores how well two-sided MiniLM and tf-idf kNN tell the three apart.
# uv run -m scripts.run_three_way_truth --run_dir results/raw/261027_exp68_three_way_truth_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --name_file results/raw/261022_exp41_gap_loop_v1/reviewed_topics.jsonl --seed 0

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import simple_parsing
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import centroid
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
KINDS = ("deployment_only", "training_only", "shared")


@dataclass
class ThreeWayConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    seed: int = 0
    n_per_kind: int = 10
    min_size: int = 200
    max_size: int = 3000
    ratio_band: tuple[float, float] = (0.5, 2.0)
    template_chars: int = 60
    template_share: float = 0.3
    k: int = 50
    extra: int = 20
    ref_frac: float = 0.2
    row_cap: int = 300
    n_line_rows: int = 20000
    n_subsets: int = 1000
    subset_size: int = 500
    line_factor: float = 3.0
    min_rows: int = 50
    min_valid: int = 8
    works: int = 24
    fails: int = 15
    better_by: int = 4
    tfidf_features: int = 200_000
    name_file: str = ""


def knn_counts_dense(q, q_hash, ref, ref_hash, ref_side, k, extra, chunk=1024) -> tuple[np.ndarray, np.ndarray]:
    # traffic and mix counts among the k nearest reference rows, exact copies (and so self) left out
    n_t, n_m = np.zeros(len(q)), np.zeros(len(q))
    for s in range(0, len(q), chunk):
        qq = torch.from_numpy(q[s : s + chunk]).to(ref.device, ref.dtype)
        sims, idx = (qq @ ref.T).float().topk(k + extra, dim=1)
        qh = torch.from_numpy(q_hash[s : s + chunk]).to(ref.device)[:, None]
        sims = sims.masked_fill(ref_hash[idx] == qh, -torch.inf)
        top_s, top_i = sims.topk(k, dim=1)
        side = ref_side[idx.gather(1, top_i)] * torch.isfinite(top_s)
        valid = torch.isfinite(top_s).sum(1)
        n_m[s : s + chunk] = side.sum(1).cpu().numpy()
        n_t[s : s + chunk] = (valid - side.sum(1)).cpu().numpy()
    return n_t, n_m


def knn_counts_sparse(q, q_hash, ref, ref_hash, ref_side, k, extra, chunk=256) -> tuple[np.ndarray, np.ndarray]:
    n_t, n_m = np.zeros(q.shape[0]), np.zeros(q.shape[0])
    ref_t = ref.T.tocsr()
    for s in range(0, q.shape[0], chunk):
        sims = (q[s : s + chunk] @ ref_t).toarray()
        idx = np.argpartition(-sims, k + extra, axis=1)[:, : k + extra]
        top = np.take_along_axis(sims, idx, 1)
        top[ref_hash[idx] == q_hash[s : s + chunk, None]] = -np.inf
        order = np.argsort(-top, axis=1)[:, :k]
        keep_i = np.take_along_axis(idx, order, 1)
        is_valid = np.isfinite(np.take_along_axis(top, order, 1))
        mix = (ref_side[keep_i] * is_valid).sum(1)
        n_m[s : s + chunk] = mix
        n_t[s : s + chunk] = is_valid.sum(1) - mix
    return n_t, n_m


def main():
    config = simple_parsing.parse(ThreeWayConfig)
    assert config.run_dir and config.exp37_dir and config.name_file, "run_dir, exp37_dir and name_file required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    ratio37 = np.load(Path(config.exp37_dir) / "ratio.npy")
    m_np, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    t_text = [r["text"] for r in rows]
    LOGGER.info(f"traffic {len(rows)}, mix {len(m_np)}")

    clusters = load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}
    sig = {}
    for c in order:
        _, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = count >= config.template_share * len(members[c])
    pool_median = float(np.median(ratio37))
    lo, hi = (b * pool_median for b in config.ratio_band)
    candidates = [
        c
        for c in order
        if not sig[c]
        and config.min_size <= len(members[c]) <= config.max_size
        and lo <= float(np.median(ratio37[members[c]])) <= hi
    ]
    drawn = rng.choice(candidates, size=3 * config.n_per_kind, replace=False).tolist()
    kind = {int(c): KINDS[i // config.n_per_kind] for i, c in enumerate(drawn)}
    units = [int(c) for c in drawn]

    m_gpu = torch.from_numpy(m_np).cuda()
    t_gpu = torch.from_numpy(t_np).cuda()
    cmat = torch.from_numpy(np.stack([centroid(t_np[members[c]].astype(np.float32)) for c in order])).cuda().half()
    nearest = torch.cat([(m_gpu[s : s + 65536] @ cmat.T).argmax(1) for s in range(0, len(m_np), 65536)]).cpu().numpy()
    order_arr = np.array(order)
    m_topic = order_arr[nearest]
    del cmat

    def knn_rows(q: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
        return np.unique(
            torch.cat(
                [(q[s : s + 1024] @ ref.T).topk(config.k, dim=1).indices.flatten() for s in range(0, len(q), 1024)]
            )
            .cpu()
            .numpy()
        )

    keep_t, keep_m = np.ones(len(rows), dtype=bool), np.ones(len(m_np), dtype=bool)
    deleted = {}
    for c in units:
        if kind[c] == "deployment_only":
            hit = np.union1d(np.where(m_topic == c)[0], knn_rows(t_gpu[members[c]], m_gpu))
            deleted[c] = int(keep_m[hit].sum())
            keep_m[hit] = False
        elif kind[c] == "training_only":
            own_mix = np.where(m_topic == c)[0]
            hit = np.union1d(members[c], knn_rows(m_gpu[own_mix], t_gpu) if len(own_mix) else np.array([], int))
            deleted[c] = int(keep_t[hit].sum())
            keep_t[hit] = False
    del m_gpu, t_gpu
    torch.cuda.empty_cache()

    unit_rows = {}
    for c in units:
        tr = members[c][keep_t[members[c]]]
        mr = np.where((m_topic == c) & keep_m)[0]
        unit_rows[c] = (tr, mr)
    validity = {
        c: {
            "traffic_left": len(unit_rows[c][0]),
            "mix_left": len(unit_rows[c][1]),
            "deleted": deleted.get(c, 0),
        }
        for c in units
    }
    ok_dep = sum(
        validity[c]["traffic_left"] >= config.min_rows and validity[c]["mix_left"] == 0
        for c in units
        if kind[c] == "deployment_only"
    )
    ok_tr = sum(
        validity[c]["mix_left"] >= config.min_rows and validity[c]["traffic_left"] == 0
        for c in units
        if kind[c] == "training_only"
    )
    is_valid = ok_dep >= config.min_valid and ok_tr >= config.min_valid
    LOGGER.info(f"validity: deployment {ok_dep}, training {ok_tr}, valid {is_valid}")

    kt, km = np.where(keep_t)[0], np.where(keep_m)[0]
    ref_t = np.sort(rng.choice(kt, size=int(config.ref_frac * len(kt)), replace=False))
    ref_m = np.sort(rng.choice(km, size=int(config.ref_frac * len(km)), replace=False))
    queries = []
    for c in units:
        tr, mr = unit_rows[c]
        tr = rng.choice(tr, size=min(config.row_cap, len(tr)), replace=False) if len(tr) else tr
        mr = rng.choice(mr, size=min(config.row_cap, len(mr)), replace=False) if len(mr) else mr
        queries += [(c, "traffic", int(i)) for i in tr] + [(c, "mix", int(i)) for i in mr]
    # line rows: a random sample of the sampled reference, scored the same way
    line_t = rng.choice(ref_t, size=config.n_line_rows // 2, replace=False)
    line_m = rng.choice(ref_m, size=config.n_line_rows // 2, replace=False)
    queries += [(-1, "traffic", int(i)) for i in line_t] + [(-1, "mix", int(i)) for i in line_m]
    q_side = np.array([s == "mix" for _, s, _ in queries])
    q_idx = np.array([i for _, _, i in queries])
    q_unit = np.array([u for u, _, _ in queries])
    q_hash = np.where(q_side, m_hash[np.where(q_side, q_idx, 0)], t_hash[np.where(q_side, 0, q_idx)])
    LOGGER.info(f"reference traffic {len(ref_t)} mix {len(ref_m)}, queries {len(queries)}")

    def dense_counts(rt: np.ndarray, rm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ref = torch.from_numpy(np.concatenate([t_np[rt], m_np[rm]])).cuda()
        ref_hash = torch.from_numpy(np.concatenate([t_hash[rt], m_hash[rm]])).cuda()
        ref_side = torch.from_numpy(np.r_[np.zeros(len(rt)), np.ones(len(rm))]).cuda()
        q = np.where(q_side[:, None], m_np[np.where(q_side, q_idx, 0)], t_np[np.where(q_side, 0, q_idx)])
        out = knn_counts_dense(q, q_hash, ref, ref_hash, ref_side, config.k, config.extra)
        del ref
        torch.cuda.empty_cache()
        return out

    counts = {"minilm": dense_counts(ref_t, ref_m)}
    ref_text = [t_text[i] for i in ref_t] + [m_text[i] for i in ref_m]
    vec = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=config.tfidf_features, dtype=np.float32
    )
    ref_x = vec.fit_transform(ref_text)
    q_x = vec.transform([m_text[i] if s else t_text[i] for s, i in zip(q_side, q_idx, strict=True)])
    LOGGER.info(f"tfidf vocab {len(vec.vocabulary_)}, ref {ref_x.shape}")
    counts["tfidf"] = knn_counts_sparse(
        sp.csr_matrix(q_x),
        q_hash,
        ref_x,
        np.concatenate([t_hash[ref_t], m_hash[ref_m]]),
        np.r_[np.zeros(len(ref_t)), np.ones(len(ref_m))],
        config.k,
        config.extra,
    )
    counts["minilm_full_ref"] = dense_counts(kt, km)
    n_ref = {
        "minilm": (len(ref_t), len(ref_m)),
        "tfidf": (len(ref_t), len(ref_m)),
        "minilm_full_ref": (len(kt), len(km)),
    }

    def classify(det: str) -> tuple[dict, dict]:
        n_t, n_m = counts[det]
        nt, nm = n_ref[det]
        score = np.log(((n_t + 1) / nt) / ((n_m + 1) / nm))
        lines = score[q_unit == -1]
        m0 = float(
            np.median(
                [np.median(rng.choice(lines, config.subset_size, replace=False)) for _ in range(config.n_subsets)]
            )
        )
        up, down = m0 + np.log(config.line_factor), m0 - np.log(config.line_factor)
        out = {}
        for c in units:
            s = float(np.median(score[q_unit == c]))
            call = "deployment_only" if s >= up else "training_only" if s <= down else "shared"
            out[c] = {"score": round(s, 4), "call": call}
        return out, {"m0": round(m0, 4), "up": round(float(up), 4), "down": round(float(down), 4)}

    def one_sided() -> tuple[dict, dict]:
        # exp41's traffic-side score on the full reference: median ratio over traffic rows only
        n_t, n_m = counts["minilm_full_ref"]
        nt, nm = n_ref["minilm_full_ref"]
        ratio = ((n_t + 1) / nt) / ((n_m + 1) / nm)
        lines = ratio[(q_unit == -1) & ~q_side]
        line = config.line_factor * float(
            np.median(
                [np.median(rng.choice(lines, config.subset_size, replace=False)) for _ in range(config.n_subsets)]
            )
        )
        out = {}
        for c in units:
            sel = (q_unit == c) & ~q_side
            s = float(np.median(ratio[sel])) if sel.any() else None
            out[c] = {
                "score": None if s is None else round(s, 4),
                "call": "deployment_only" if s is not None and s >= line else "shared",
            }
        return out, {"line": round(line, 4)}

    results = {d: classify(d) for d in ("minilm", "tfidf", "minilm_full_ref")}
    results["one_sided_traffic"] = one_sided()

    def summary(calls: dict) -> dict:
        correct = sum(calls[c]["call"] == kind[c] for c in units)
        confusion = {t: {p: sum(kind[c] == t and calls[c]["call"] == p for c in units) for p in KINDS} for t in KINDS}
        return {"correct": correct, "of": len(units), "confusion": confusion}

    summaries = {d: summary(r[0]) for d, r in results.items()}

    def judge(n: int) -> str:
        return "works" if n >= config.works else "fails" if n <= config.fails else "ambiguous"

    diff = summaries["minilm"]["correct"] - summaries["tfidf"]["correct"]
    names = {int(x.get("topic", x.get("cluster"))): x["name"] for x in load_jsonl(Path(config.name_file))}
    decision = {
        "valid": is_valid,
        "validity_counts": {"deployment_only_ok": ok_dep, "training_only_ok": ok_tr},
        "verdict_per_detector": {d: judge(summaries[d]["correct"]) for d in ("minilm", "tfidf")},
        "comparison": "minilm_better"
        if diff >= config.better_by
        else "tfidf_better"
        if -diff >= config.better_by
        else "tie",
        "summaries": summaries,
        "lines": {d: r[1] for d, r in results.items()},
        "units": [
            {
                "id": f"c{c}",
                "topic": c,
                "kind": kind[c],
                "name": names.get(c),
                "size": len(members[c]),
                **validity[c],
                **{f"{d}_score": results[d][0][c]["score"] for d in results},
                **{f"{d}_call": results[d][0][c]["call"] for d in results},
            }
            for c in units
        ],
        "reference": {"traffic": len(ref_t), "mix": len(ref_m), "traffic_kept": len(kt), "mix_kept": len(km)},
        "n_queries": int((q_unit != -1).sum()),
    }
    write_json(decision, run / "decision.json")
    print(json.dumps({k: decision[k] for k in ("valid", "verdict_per_detector", "comparison")}, indent=1))
    print({d: s["correct"] for d, s in summaries.items()})
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
