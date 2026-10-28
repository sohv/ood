# exp78: scores exp56's flash variant-b rater one request per call on exp01's legal, medical and wildchat pools and compares its auroc to tf-idf and activations.
# uv run -m scripts.run_rater_single_auroc --output_dir results/raw/261027_exp78_rater_auroc_v1 --exp01_dir results/raw/261018_exp01_repro_v1 --pools_dir data/processed/exp01 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --model_id google/gemini-3.8-flash --seed 0

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import simple_parsing
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from scripts.run_autorater_baseline import LABELS, fmt_pairs, parse
from src.data.gap_pool import load_mix
from src.data.io import load_jsonl
from src.metrics.detectors import auroc, stratified_auroc
from src.metrics.discovery import name_clusters
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
POOLS = ("legal", "medical", "wildchat_en_test")
OOD = ("legal", "medical")
REF = "wildchat_en_test"
WORDS = {"legal": 0.952, "medical": 0.856}
ACTIVATIONS = {"legal": 0.954, "medical": 0.897}
PROMPT = """You are auditing the post-training data of a chat model. Below is a request a real user sent to a chatbot, and {ref_desc}.

Judge how well training data like the example covers requests like this one: the same kind of task, subject, format or language, not the exact words.

User request:
{topic}

Training example:
{ref}

Reply with one JSON object and nothing else: {{"coverage": <0 to 100, where 0 means nothing like this request appears in training data like the example and 100 means it is fully covered>, "label": "COVERED" | "PARTLY" | "MISSING", "reason": "<one sentence>"}}"""
REF_DESC = "the single most similar example found in the post-training data"


@dataclass
class RaterAurocConfig:
    output_dir: str = ""
    exp01_dir: str = ""
    pools_dir: str = ""
    exp37_dir: str = ""
    model_id: str = "google/gemini-3.8-flash"
    seed: int = 0
    max_chars: int = 300
    max_concurrent: int = 16
    n_examples: int = 5
    mix_cache: str = "cache/exp78_79_mix"
    num_tasks: int | None = None


def cached_mix(c37: dict, cache_dir: str) -> np.ndarray:
    # load_mix runs once and is saved, so later runs memory-map the embeddings and read only the texts they need
    path = Path(cache_dir)
    if not (path / "emb.npy").exists():
        m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": m_text}).to_parquet(path / "texts.parquet")
        np.save(path / "emb.npy", m_np)
        write_json(
            {"git_hash": get_git_hash(), "sft_path": c37["sft_path"], "emb_v1": c37["emb_v1"]}, path / "meta.json"
        )
        del m_np, m_text
    return np.load(path / "emb.npy", mmap_mode="r")


def mix_texts(cache_dir: str, idx: np.ndarray) -> list[str]:
    return pq.read_table(Path(cache_dir) / "texts.parquet", columns=["text"]).column("text").take(idx).to_pylist()


def nearest_mix(q: np.ndarray, m: np.ndarray, q_chunk: int = 1024, m_chunk: int = 200_000) -> np.ndarray:
    # argmax cosine over the mix on cpu, both sides chunked so no block exceeds about 1 gb
    best_sim = np.full(len(q), -np.inf, dtype=np.float32)
    best_idx = np.zeros(len(q), dtype=np.int64)
    qt = torch.from_numpy(q.astype(np.float32))
    for ms in range(0, len(m), m_chunk):
        mt = torch.from_numpy(m[ms : ms + m_chunk].astype(np.float32))
        for qs in range(0, len(q), q_chunk):
            sim, idx = (qt[qs : qs + q_chunk] @ mt.T).max(1)
            sim, idx = sim.numpy(), idx.numpy() + ms
            better = sim > best_sim[qs : qs + q_chunk]
            best_sim[qs : qs + q_chunk][better] = sim[better]
            best_idx[qs : qs + q_chunk][better] = idx[better]
    return best_idx


def rate(
    texts: list[str], exp37_dir: str, mix_cache: str, model_id: str, max_chars: int, max_concurrent: int, run: Path
) -> dict:
    # one rater call per text, shown with its minilm-nearest dolci sft+dpo prompt
    c37 = json.loads((Path(exp37_dir) / "config.json").read_text())
    c9 = json.loads((Path(c37["exp09_dir"]) / "config.json").read_text())
    embedder = SentenceTransformer(c9["embed_model"], device="cpu", revision=c9["embed_model_revision"])
    q = embedder.encode(
        [t[: c37["embed_chars"]] for t in texts], batch_size=256, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float16)
    m_np = cached_mix(c37, mix_cache)
    LOGGER.info(f"mix {m_np.shape} {m_np.dtype}, queries {q.shape}")
    nn = nearest_mix(q, m_np)
    del m_np
    near = mix_texts(mix_cache, nn)
    prompts = [
        PROMPT.format(ref_desc=REF_DESC, topic=topic, ref=ref)
        for topic, ref in (fmt_pairs([t], [r], max_chars) for t, r in zip(texts, near))
    ]
    LOGGER.info(f"{len(prompts)} prompts built, calling {model_id}")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(name_clusters(client, model_id, prompts, str(run / "rater_cache"), max_concurrent))
    parsed = [parse(r) for r in replies]
    return {
        "coverage": np.array([p[0] if p[0] is not None else np.nan for p in parsed]),
        "label": [p[1] for p in parsed],
        "replies": replies,
        "prompts": prompts,
        "nearest_idx": nn,
        "embed_model": c9["embed_model"],
        "embed_model_revision": c9["embed_model_revision"],
    }


def gap_scores(coverage: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    # failed calls take the median coverage; seeded jitter breaks ties
    failed = int(np.isnan(coverage).sum())
    cov = np.where(np.isnan(coverage), np.nanmedian(coverage), coverage)
    return 100 - cov + np.random.default_rng(seed).random(len(cov)) * 1e-6, failed


def main():
    config = simple_parsing.parse(RaterAurocConfig)
    missing = [k for k in ("output_dir", "exp01_dir", "pools_dir", "exp37_dir") if getattr(config, k) == ""]
    assert not missing, f"required arguments not set: {missing}"
    run = Path(config.output_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    torch.set_num_threads(16)
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "prompt": PROMPT, "ref_desc": REF_DESC}, run / "config.json"
    )

    e01 = {r["id"]: r for r in load_jsonl(Path(config.exp01_dir) / "scores_decision_cell.jsonl")}
    rows = []
    for pool in POOLS:
        pool_rows = load_jsonl(Path(config.pools_dir) / f"{pool}.jsonl")[: config.num_tasks]
        for r in pool_rows:
            assert e01[r["id"]]["pool"] == pool, r["id"]
            rows.append({**r, "pool": pool, "len": e01[r["id"]]["n_content_tokens"]})
    LOGGER.info(f"{len(rows)} rows: " + ", ".join(f"{p} {sum(r['pool'] == p for r in rows)}" for p in POOLS))

    out = rate(
        [r["text"] for r in rows],
        config.exp37_dir,
        config.mix_cache,
        config.model_id,
        config.max_chars,
        config.max_concurrent,
        run,
    )
    gap, n_failed = gap_scores(out["coverage"], config.seed)
    pool_of = np.array([r["pool"] for r in rows])
    lengths = np.array([r["len"] for r in rows])
    write_jsonl(
        [
            {
                "id": r["id"],
                "pool": r["pool"],
                "n_content_tokens": int(r["len"]),
                "coverage": None if np.isnan(out["coverage"][i]) else float(out["coverage"][i]),
                "label": out["label"][i],
                "gap_score": float(gap[i]),
            }
            for i, r in enumerate(rows)
        ],
        run / "scores.jsonl",
    )
    write_json(
        {
            p: [
                {
                    "id": rows[i]["id"],
                    "prompt": out["prompts"][i],
                    "reply": out["replies"][i],
                }
                for i in np.where(pool_of == p)[0][: config.n_examples]
            ]
            for p in POOLS
        },
        run / "raw_replies.json",
    )

    # the same comparison recomputed from exp01's saved scores, as a reproduction check of the reference numbers
    def reference(ood: str, key: str) -> dict:
        o = np.array([r[key] for r in e01.values() if r["pool"] == ood])
        f = np.array([r[key] for r in e01.values() if r["pool"] == REF])
        lo = np.array([r["n_content_tokens"] for r in e01.values() if r["pool"] == ood])
        lf = np.array([r["n_content_tokens"] for r in e01.values() if r["pool"] == REF])
        return {"raw": auroc(o, f), "stratified": stratified_auroc(o, f, lo, lf)["auroc"]}

    results = {}
    for ood in OOD:
        o, f = gap[pool_of == ood], gap[pool_of == REF]
        strat = stratified_auroc(o, f, lengths[pool_of == ood], lengths[pool_of == REF])
        results[ood] = {
            "rater_raw": auroc(o, f),
            "rater_stratified": strat["auroc"],
            "strat_n_bands": strat["n_bands"],
            "strat_coverage": strat["coverage"],
            "strat_bands": strat["bands"],
            "words_stratified_preregistered": WORDS[ood],
            "activations_stratified_preregistered": ACTIVATIONS[ood],
            "tfidf_recomputed": reference(ood, "tfidf"),
            "dmaha_full_recomputed": reference(ood, "dmaha_full"),
        }
    pool_stats = {
        p: {
            "n": int((pool_of == p).sum()),
            "failed_calls": int(np.isnan(out["coverage"][pool_of == p]).sum()),
            "median_coverage": float(np.nanmedian(out["coverage"][pool_of == p])),
            "label_counts": {L: int(sum(out["label"][i] == L for i in np.where(pool_of == p)[0])) for L in LABELS},
        }
        for p in POOLS
    }
    strat_vals = {p: results[p]["rater_stratified"] for p in OOD}
    if any(v is None for v in strat_vals.values()):
        verdict = "not_measurable"
    elif all(strat_vals[p] >= WORDS[p] + 0.05 for p in OOD):
        verdict = "rater_beats_words"
    elif all(WORDS[p] >= strat_vals[p] + 0.05 for p in OOD):
        verdict = "words_beat_rater"
    else:
        verdict = "tie"
    decision = {
        "git_hash": get_git_hash(),
        "verdict": verdict,
        "bar": "rater beats words if its stratified auroc is >= 0.05 above tf-idf on both pools; words beat rater if >= 0.05 above on both; otherwise tie",
        "results": results,
        "pools": pool_stats,
        "sanity": {
            "rows_written": len(rows),
            "rows_expected": len(POOLS) * (config.num_tasks or 500),
            "failed_calls": n_failed,
            "embed_model": out["embed_model"],
            "embed_model_revision": out["embed_model_revision"],
        },
    }
    write_json(decision, run / "decision.json")
    print(f"verdict {verdict}")
    for p in OOD:
        r = results[p]
        print(
            f"{p} vs wildchat: rater raw {r['rater_raw']:.3f} stratified {r['rater_stratified']}; "
            f"words {WORDS[p]} activations {ACTIVATIONS[p]}; recomputed tf-idf {r['tfidf_recomputed']}"
        )
    print(f"failed calls {n_failed}; pools {pool_stats}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
