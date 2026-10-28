# exp72: plants exp44 v2's 10 topics by deleting their tf-idf look-alikes from the mix, then counts how many the neighbour count, the tf-idf word score and the flash rater put in their top 293.
# uv run -m scripts.run_planted_words --run_dir results/raw/261027_exp72_planted_words_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp44_dir results/raw/261022_exp44_planted_gaps_v2 --exp56_dir results/raw/261026_exp56_autorater_v1 --seed 0

import asyncio
import json
import logging
import os
import resource
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import simple_parsing
import torch
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.run_autorater_baseline import PROMPT, REF_DESC, fmt_pairs, parse
from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import name_clusters
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json, write_jsonl
from src.metrics.two_sided_knn import sparse_nearest, word_logits
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SCORES = ("neighbour_count", "words", "rater")


@dataclass
class PlantedWordsConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp44_dir: str = ""
    exp56_dir: str = ""
    model_id: str = "google/gemini-3.8-flash"
    seed: int = 0
    k: int = 50
    chars: int = 1000
    max_features: int = 200_000
    min_df: int = 5
    n_validity_sample: int = 5000
    validity_pct: float = 25.0
    min_valid: int = 8
    n_shown: int = 10
    max_chars: int = 300
    max_concurrent: int = 16
    top_n: int = 293
    finds: int = 6
    fails: int = 2
    better_by: int = 3
    knn_chunk: int = 64
    knn_threads: int = 2
    gpu_poll_s: int = 300
    # auto: cuda fp16 if the shared gpu is free at that moment, else cpu fp32; wait_gpu polls until it is free
    count_device: str = "wait_gpu"
    gpu_free_mib: int = 45000
    gpu_done_file: str = "/root/exp66r2.out"
    # smoke runs: first n planted topics, first n ranked topics for the rater, count only for planted plus first n rows
    n_planted: int | None = None
    rater_limit: int | None = None
    count_limit: int | None = None


def nearest_parallel(q: sp.csr_matrix, ref: sp.csr_matrix, k: int, chunk: int, threads: int) -> tuple:
    # sparse_nearest over row blocks in threads; the transpose is built once and passed in csr-transposed form
    ref_tt = ref.T.tocsr().T
    blocks = [(s, min(s + chunk * 8, q.shape[0])) for s in range(0, q.shape[0], chunk * 8)]
    with ThreadPoolExecutor(threads) as ex:
        out = list(ex.map(lambda b: sparse_nearest(q[b[0] : b[1]], ref_tt, k, chunk), blocks))
    return np.vstack([o[0] for o in out]), np.vstack([o[1] for o in out])


def gpu_is_free(config: PlantedWordsConfig) -> bool:
    done = Path(config.gpu_done_file).exists() and "ALL DONE" in Path(config.gpu_done_file).read_text()
    used = int(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()[0]
    )
    vllm = subprocess.run(["pgrep", "-f", "run_fix_rollouts"], check=False, capture_output=True).returncode == 0
    LOGGER.info(f"gpu check: {done=} {used=} {vllm=}")
    return done or (used < config.gpu_free_mib and not vllm)


def main():
    config = simple_parsing.parse(PlantedWordsConfig)
    assert config.run_dir and config.exp37_dir and config.exp44_dir and config.exp56_dir, "dirs required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    shutil.copytree(Path(config.exp56_dir) / "rater_cache", run / "rater_cache")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    m_np, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    t_text = [r["text"] for r in rows]
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    fold_t, fold_m = (t_hash & 1).astype(np.int64), (m_hash & 1).astype(np.int64)
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    ranked = sorted((c for c in clusters if clusters[c]["rank"]), key=lambda c: clusters[c]["rank"])
    members = {c: np.where(labels == c)[0] for c in ranked}
    planted = sorted(
        p["cluster"] for p in json.loads((Path(config.exp44_dir) / "decision.json").read_text())["planted"]
    )
    planted = planted[: config.n_planted]
    LOGGER.info(f"{len(rows)} traffic, {len(m_text)} mix, {len(ranked)} ranked topics, planted {planted}")

    # word-space deletion: every mix row among the k tf-idf-nearest mix rows of any planted prompt
    vec = TfidfVectorizer(
        sublinear_tf=True, min_df=config.min_df, max_features=config.max_features, ngram_range=(1, 2), dtype=np.float32
    )
    x = vec.fit_transform([t[: config.chars] for t in t_text] + [t[: config.chars] for t in m_text]).tocsr()
    x_t, x_m = x[: len(rows)], x[len(rows) :]
    LOGGER.info(f"tf-idf fitted: {x.shape}, {x.nnz} nonzeros")
    p_idx = np.concatenate([members[c] for c in planted])
    p_topic = np.concatenate([np.full(len(members[c]), c) for c in planted])
    t0 = time.time()
    nn_idx, nn_sim = nearest_parallel(x_t[p_idx], x_m, config.k, config.knn_chunk, config.knn_threads)
    LOGGER.info(f"planted kNN of {len(p_idx)} prompts took {time.time() - t0:.0f}s")
    keep = np.ones(len(m_text), dtype=bool)
    removed = {}
    for c in planted:
        hit = np.unique(nn_idx[p_topic == c])
        removed[str(c)] = int(keep[hit].sum())
        keep[hit] = False
    n_deleted = int((~keep).sum())
    kept = np.where(keep)[0]
    LOGGER.info(f"deleted {n_deleted} mix rows: {removed}")

    # validity: planted medians of nearest ablated-mix cosine against the pool's 25th percentile from a 5,000 sample
    sample = np.random.default_rng(config.seed).choice(len(rows), size=config.n_validity_sample, replace=False)
    q_idx = np.concatenate([p_idx, sample])
    _, near_sim = nearest_parallel(x_t[q_idx], x_m[kept], 1, config.knn_chunk, config.knn_threads)
    near_p, near_s = near_sim[: len(p_idx), 0], near_sim[len(p_idx) :, 0]
    pool_pct = float(np.percentile(near_s, config.validity_pct))
    planted_med = {str(c): float(np.median(near_p[p_topic == c])) for c in planted}
    planted_med_before = {str(c): float(np.median(nn_sim[p_topic == c, 0])) for c in planted}
    n_below = sum(v < pool_pct for v in planted_med.values())
    is_valid = n_below >= min(config.min_valid, len(planted))
    LOGGER.info(f"validity: pool p25 {pool_pct:.4f}, planted medians {planted_med}, {n_below} below, {is_valid=}")
    del x, x_t, x_m

    # score 2: exp47's tf-idf word score on the ablated mix
    w_abl = word_logits(
        t_text,
        [m_text[i] for i in kept],
        fold_t,
        fold_m[keep],
        config.chars,
        config.min_df,
        config.max_features,
        config.seed,
    )
    word_score = {c: float(np.median(w_abl[members[c]])) for c in ranked}
    word_rank = sorted(ranked, key=lambda c: -word_score[c])
    LOGGER.info("word score on the ablated mix done")

    # score 3: exp56's flash variant b, each shown prompt paired with its minilm-nearest ablated-mix prompt
    rng = np.random.default_rng(config.seed)
    rater_topics = ranked[: config.rater_limit]
    shown = {c: rng.choice(members[c], size=min(config.n_shown, len(members[c])), replace=False) for c in rater_topics}
    m_abl32 = torch.from_numpy(m_np[keep].astype(np.float32))
    prompts = []
    for c in rater_topics:
        best = (torch.from_numpy(t_np[shown[c]].astype(np.float32)) @ m_abl32.T).argmax(1).numpy()
        topic, ref = fmt_pairs([rows[i]["text"] for i in shown[c]], [m_text[kept[j]] for j in best], config.max_chars)
        prompts.append(PROMPT.format(ref_desc=REF_DESC["B"], topic=topic, ref=ref))
    del m_abl32
    n_cached = len(list((run / "rater_cache").glob("*.json")))
    LOGGER.info(f"{len(prompts)} rater prompts built, cache holds {n_cached} exp56 replies")
    replies = asyncio.run(
        name_clusters(client, config.model_id, prompts, str(run / "rater_cache"), config.max_concurrent)
    )
    parsed = [parse(r) for r in replies]
    cov = np.array([p[0] if p[0] is not None else np.nan for p in parsed])
    rater_failed = int(np.isnan(cov).sum())
    null_replies = int(sum(r is None for r in replies))
    # failed calls get the median coverage, as exp56 did
    cov = np.where(np.isnan(cov), np.nanmedian(cov), cov)
    gap = 100 - cov + np.random.default_rng(config.seed).random(len(cov)) * 1e-6
    rater_rank = [rater_topics[i] for i in np.argsort(-gap, kind="stable")]
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "rank37": clusters[c]["rank"],
                "gap_score": round(float(gap[i]), 3),
                "label": parsed[i][1],
                "reply": replies[i],
            }
            for i, c in enumerate(rater_topics)
        ],
        run / "scores_rater.jsonl",
    )
    LOGGER.info(f"rater done: {rater_failed} failed of {len(prompts)}, {null_replies} null replies")

    # score 1: the minilm neighbour count, on the gpu if it is free, else on cpu
    if config.count_device == "wait_gpu":
        while not gpu_is_free(config):
            time.sleep(config.gpu_poll_s)
    device = "cuda" if config.count_device == "wait_gpu" or gpu_is_free(config) else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    q_rows = np.arange(len(rows)) if config.count_limit is None else np.union1d(p_idx, np.arange(config.count_limit))
    LOGGER.info(f"neighbour count on {device} for {len(q_rows)} queries")
    t0 = time.time()
    ratio, near = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    ratio[q_rows], near[q_rows] = density_ratios_batched(
        t_np[q_rows],
        t_hash[q_rows],
        q_rows,
        torch.from_numpy(t_np).to(device, dtype),
        torch.from_numpy(t_hash).to(device),
        torch.from_numpy(m_np[keep]).to(device, dtype),
        torch.from_numpy(m_hash[keep]).to(device),
        k=config.k,
        chunk=1024 if device == "cuda" else 128,
        return_nearest_mix=True,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    count_s = time.time() - t0
    knn_score = {
        c: float(np.nanmedian(ratio[members[c]])) if np.isfinite(ratio[members[c]]).any() else -np.inf for c in ranked
    }
    knn_near = {
        c: float(np.nanmedian(near[members[c]])) if np.isfinite(near[members[c]]).any() else np.inf for c in ranked
    }
    knn_rank = sorted(ranked, key=lambda c: (-knn_score[c], knn_near[c]))
    LOGGER.info("neighbour count on the ablated mix done")

    write_jsonl(
        [
            {
                "id": f"c{c}",
                "rank37": clusters[c]["rank"],
                "neighbour_ratio": knn_score[c],
                "neighbour_near": knn_near[c],
                "word_logit": word_score[c],
            }
            for c in ranked
        ],
        run / "scores_count_words.jsonl",
    )
    ranks = {"neighbour_count": knn_rank, "words": word_rank, "rater": rater_rank}
    pos = {a: {str(c): (r.index(c) + 1 if c in r else None) for c in planted} for a, r in ranks.items()}
    in_top = {a: sum(p is not None and p <= config.top_n for p in ps.values()) for a, ps in pos.items()}
    verdict = {
        a: "finds" if n >= config.finds else "fails" if n <= config.fails else "ambiguous" for a, n in in_top.items()
    }
    pairs = {}
    for i, a in enumerate(SCORES):
        for b in SCORES[i + 1 :]:
            lead = in_top[a] - in_top[b]
            pairs[f"{a}_vs_{b}"] = {
                "lead": lead,
                "result": f"{a}_beats"
                if lead >= config.better_by
                else f"{b}_beats"
                if -lead >= config.better_by
                else "tie",
            }

    decision = {
        "git_hash": get_git_hash(),
        "valid": bool(is_valid),
        "validity": {
            "pool_nearest_mix_cosine_p25": pool_pct,
            "pool_sample_size": config.n_validity_sample,
            "deviation": "the pool percentile is estimated from a 5,000-prompt traffic sample (seed 0), not all 992,933",
            "planted_median_nearest_ablated_mix_cosine": planted_med,
            "planted_median_nearest_full_mix_cosine": planted_med_before,
            "planted_below_p25": n_below,
            "needed": config.min_valid,
        },
        "planted": planted,
        "top_n": config.top_n,
        "planted_in_top": in_top,
        "verdict_per_score": verdict,
        "comparisons": pairs,
        "planted_positions": pos,
        "planted_scores": {
            str(c): {
                "neighbour_ratio": knn_score[c],
                "word_logit": word_score[c],
                "rater_gap": float(gap[rater_topics.index(c)]) if c in rater_topics else None,
                "size": len(members[c]),
            }
            for c in planted
        },
        "mix_rows_deleted": n_deleted,
        "mix_rows_deleted_per_topic": removed,
        "mix_rows_kept": int(keep.sum()),
        "rater": {
            "n_calls": len(prompts),
            "failed_calls": rater_failed,
            "null_replies": null_replies,
            "exp56_cache_files_copied": n_cached,
            "label_counts": {L: int(sum(p[1] == L for p in parsed)) for L in ("COVERED", "PARTLY", "MISSING")},
        },
        "null_counts": {
            "rater_unparsed": rater_failed,
            "rater_null_replies": null_replies,
            "neighbour_ratio_nan": int(np.isnan(ratio[q_rows]).sum()),
            "word_logit_nan": int(np.isnan(w_abl).sum()),
            "planted_positions_missing": {a: sum(p is None for p in ps.values()) for a, ps in pos.items()},
        },
        "neighbour_count_device": device,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "neighbour_count_seconds": count_s,
        "neighbour_count_queries": len(q_rows),
        "sanity": {
            "ranked_topics": len(ranked),
            "rater_topics": len(rater_topics),
            "traffic": len(rows),
            "mix": len(m_text),
        },
    }
    write_json(decision, run / "decision.json")
    print(f"valid {is_valid}: {n_below} of {len(planted)} planted below pool p25 {pool_pct:.4f}")
    print(f"planted in top {config.top_n}: {in_top}; verdicts {verdict}")
    print(f"comparisons {pairs}")
    print(f"positions {pos}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
