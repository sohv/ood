# exp09: iterative gap discovery in wildchat + lmsys with density-ratio and two-sided knn rankings, blind cluster naming.
# uv run --no-sync -m scripts.run_discovery --run_dir results/raw/261018_exp09_discovery_v1 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import asyncio
import collections
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from huggingface_hub import HfApi
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from src.data.io import load_jsonl
from src.data.needles import wildchat_language_map
from src.data.pools import load_dolci_dpo, normalise
from src.interp.features import LAYERS
from src.metrics.density_ratio import cross_fit_dense, length_controlled
from src.metrics.detectors import KNN
from src.metrics.discovery import centroid, cluster, match_gap, name_clusters, naming_prompt, parse_naming
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    run_dir: str = ""
    exp01_dir: str = "results/raw/261018_exp01_repro_v1"
    exp01_pools: str = "data/processed/exp01"
    wildchat_features: str = "results/raw/261018_exp06_density_ratio_v1/features"
    wildchat_pool: str = "data/processed/exp06/deploy.jsonl"
    lmsys_pool: str = "data/processed/exp09/lmsys.jsonl"
    seed: int = 0
    rounds: int = 5
    top_k: int = 500
    n_clusters: int = 10
    n_shown: int = 25
    min_size: int = 15
    same_gap_cos: float = 0.85
    c_dense: float = 0.1
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    namer: str = "anthropic/claude-sonnet-4.6"


def main():
    config = simple_parsing.parse(DiscoveryConfig)
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config), "embed_model_revision": HfApi().model_info(config.embed_model).sha,
                "olmo_layer": 16, "pooling": "cmean"}, run / "detector_config.json")
    li = LAYERS.index(16)

    wc = load_jsonl(config.wildchat_pool)
    wf = np.load(config.wildchat_features)
    lm = load_jsonl(config.lmsys_pool)
    lf = np.load(run / "features" / "lmsys.npz")
    assert list(wf["ids"]) == [r["id"] for r in wc] and list(lf["ids"]) == [r["id"] for r in lm], "feature rows misaligned"
    lang_map = wildchat_language_map([1, 2, 3])
    texts = [r["text"] for r in wc] + [r["text"] for r in lm]
    source = np.array(["wildchat"] * len(wc) + ["lmsys"] * len(lm))
    language = np.array([lang_map.get(normalise(r["text"]), "unknown") for r in wc] + [r["language"] for r in lm])
    x = np.vstack([wf["cmean"][:, li], lf["cmean"][:, li]]).astype(np.float32)
    lengths = np.concatenate([wf["n_content_tokens"], lf["n_content_tokens"]])
    LOGGER.info(f"pool {len(texts)} rows; wildchat language lookup misses {np.mean(language[: len(wc)] == 'unknown'):.3f}")

    e1 = Path(config.exp01_dir) / "features"
    mix = np.vstack([np.load(e1 / f"{n}.npz")["cmean"][:, li] for n in ("idfit", "idmix")]).astype(np.float32)
    dolma = np.load(e1 / "dolma_full.npz")["cmean"][:, li].astype(np.float32)
    scale = np.load(e1 / "idfit.npz")["cmean"][:, li].astype(np.float32).std(0) + 1e-6
    knn_dolma = KNN(dolma, scale)(x)

    embedder = SentenceTransformer(config.embed_model, device="cuda")
    emb = embedder.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    dolci = load_dolci_dpo()["first_user"].drop_duplicates().tolist()
    ref = torch.from_numpy(embedder.encode(dolci, batch_size=512, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)).cuda()
    coverage = np.concatenate([(torch.from_numpy(emb[i : i + 4096]).cuda() @ ref.T).max(dim=1).values.cpu().numpy() for i in range(0, len(emb), 4096)])
    cov_threshold = float(np.percentile(coverage, 25))
    write_jsonl([{"id": f"pool_{i:05d}", "source": str(source[i]), "language": str(language[i]), "n_tokens": int(lengths[i]), "coverage": float(coverage[i])}
                 for i in range(len(texts))], run / "pool_rows.jsonl")
    LOGGER.info(f"coverage 25th percentile threshold {cov_threshold:.3f}, median {np.median(coverage):.3f}")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    pool_median_len = float(np.median(lengths))

    def describe(members: np.ndarray, arm: str, kind: str, rnd: int, cid: int, naming: dict) -> dict:
        lang = collections.Counter(language[members]).most_common(1)[0]
        c = centroid(emb[members])
        is_confirmed = bool(naming["coherent"]) and len(members) >= config.min_size and float(np.median(coverage[members])) < cov_threshold
        return {"id": f"{arm}_{kind}_r{rnd}_c{cid}", "arm": arm, "kind": kind, "round": rnd, "cluster": cid, "size": int(len(members)),
                **naming, "median_coverage": float(np.median(coverage[members])), "confirmed": is_confirmed,
                "median_len_ratio": float(np.median(lengths[members]) / pool_median_len),
                "top_language": lang[0], "top_language_share": lang[1] / len(members), "lmsys_share": float(np.mean(source[members] == "lmsys")),
                "examples": [" ".join(texts[i].split())[:200] for i in members[:5]], "_centroid": c, "_members": members}

    all_clusters, summary = [], {}
    for arm in ("dr", "dknn"):
        remaining = np.ones(len(texts), dtype=bool)
        moved = np.zeros(len(texts), dtype=bool)
        gaps: dict[str, list[np.ndarray]] = {"top": [], "random": []}
        for rnd in range(1, config.rounds + 1):
            idx = np.where(remaining)[0]
            if arm == "dr":
                score = cross_fit_dense(x[idx], np.vstack([mix, x[moved]]), config.c_dense, 5, config.seed + rnd)
            else:
                score = KNN(np.vstack([mix, x[moved]]), scale)(x[idx]) - knn_dolma[idx]
            lc = length_controlled(score, lengths[idx])
            top = idx[np.argsort(-(lc + np.random.default_rng(rnd).random(len(lc)) * 1e-9))[: config.top_k]]
            rand = np.random.default_rng(10_000 + rnd).choice(idx, size=config.top_k, replace=False)
            groups = []
            for kind, rows in (("top", top), ("random", rand)):
                labels = cluster(emb[rows], config.n_clusters, config.seed)
                groups += [(kind, cid, rows[labels == cid]) for cid in range(config.n_clusters)]
            prompts = [naming_prompt([texts[i] for i in members], config.n_shown, config.seed) for _, _, members in groups]
            replies = asyncio.run(name_clusters(client, config.namer, prompts, str(run / "naming_cache")))
            n_failed = sum(r is None for r in replies)
            round_rows = [describe(members, arm, kind, rnd, cid, parse_naming(reply)) for (kind, cid, members), reply in zip(groups, replies)]
            for row in round_rows:
                if row["confirmed"]:
                    j = match_gap(row["_centroid"], gaps[row["kind"]], config.same_gap_cos)
                    row["is_new"] = j is None
                    row["repeats_gap"] = j
                    if j is None:
                        gaps[row["kind"]].append(row["_centroid"])
                    if row["kind"] == "top":
                        remaining[row["_members"]] = False
                        moved[row["_members"]] = True
            summary[f"{arm}_r{rnd}"] = {
                "naming_failures": n_failed,
                "top_confirmed": sum(r["confirmed"] for r in round_rows if r["kind"] == "top"),
                "top_new": sum(r.get("is_new", False) for r in round_rows if r["kind"] == "top"),
                "random_confirmed": sum(r["confirmed"] for r in round_rows if r["kind"] == "random"),
                "random_new": sum(r.get("is_new", False) for r in round_rows if r["kind"] == "random"),
                "top_lmsys_share": float(np.mean(source[top] == "lmsys")), "moved_total": int(moved.sum()),
            }
            all_clusters += round_rows
            LOGGER.info(f"{arm} round {rnd}: {summary[f'{arm}_r{rnd}']}")
        summary[f"{arm}_distinct"] = {"top": len(gaps["top"]), "random": len(gaps["random"])}
        summary[f"_{arm}_centroids"] = gaps["top"]

    cross = []
    for i, a in enumerate(summary.pop("_dr_centroids")):
        cross.append(match_gap(a, summary["_dknn_centroids"], config.same_gap_cos))
    summary.pop("_dknn_centroids")
    summary["dr_gaps_also_found_by_dknn"] = int(sum(c is not None for c in cross))

    def verdict(arm: str) -> str:
        top, rand = summary[f"{arm}_distinct"]["top"], summary[f"{arm}_distinct"]["random"]
        if top >= 5 and top >= 2 * rand:
            return "useful_discovery"
        return "no_better_than_random" if top <= rand + 1 else "ambiguous"

    decision = {arm: {**summary[f"{arm}_distinct"], "verdict": verdict(arm)} for arm in ("dr", "dknn")}
    write_jsonl([{k: v for k, v in r.items() if not k.startswith("_")} | {"id": r["id"]} for r in all_clusters], run / "clusters.jsonl")
    write_json(summary, run / "summary.json")
    write_json(decision, run / "decision.json")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
