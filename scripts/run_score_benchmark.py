# alt_gap_scores: compares four two-sided gap scores (minilm neighbour count, dsir, fine-tuned classifier, bge-m3 neighbour count) on planted gaps and user labels.
# uv run -m scripts.run_score_benchmark --run_dir results/raw/261022_alt_gap_scores_score_benchmark_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --planted_ablation_dir results/raw/261022_planted_ablation_planted_gaps_v2 --label_dirs results/raw/261022_deep_gap_loop_topic_labels_v1/labels results/raw/261022_classifier_gap_loop_topic_labels_v1/labels --seed 0

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.gap_scores import dsir_scores, finetune_logits
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
MINILM = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
BGE = ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")


@dataclass
class ScoreBenchConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    planted_ablation_dir: str = ""
    label_dirs: list[str] = field(default_factory=list)
    cache_dir: str = "/workspace/ood_cache"
    # local copy of bge-m3 at BGE[1] (weights checked against the hub sha256); the shared hf cache wrote empty files
    bge_path: str = "/root/models/bge-m3"
    seed: int = 0
    k: int = 50
    top_n: int = 293
    bge_max_len: int = 256
    planted_margin: int = 3
    auc_margin: float = 0.10
    planted_slack: int = 2


def main():
    config = simple_parsing.parse(ScoreBenchConfig)
    assert (
        config.run_dir and config.pipeline_million_dir and config.planted_ablation_dir and config.label_dirs
    ), "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), "minilm": MINILM, "bge": BGE, **asdict(config)}, run / "config.json")

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_mini, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    m_mini, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    t_text = [r["text"] for r in rows]
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    fold_t, fold_m = (t_hash & 1).astype(np.int64), (m_hash & 1).astype(np.int64)
    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}

    # bge-m3 embeddings, cached because they take most of the run
    cache = Path(config.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    bge = {}
    model = None
    for name, texts in (("traffic", t_text), ("mix", m_text)):
        path = cache / f"bge_m3_{BGE[1][:8]}_{config.bge_max_len}_{name}_{len(texts)}.npy"
        if not path.exists():
            # one load for both sets, so a flaky cache file is read once
            if model is None:
                model = SentenceTransformer(config.bge_path, device="cuda", model_kwargs={"torch_dtype": torch.float16})
                model.max_seq_length = config.bge_max_len
            np.save(path, model.encode(texts, batch_size=256, normalize_embeddings=True).astype(np.float16))
        bge[name] = np.load(path)
    del model
    torch.cuda.empty_cache()
    LOGGER.info("bge-m3 embeddings ready")

    planted = [
        p["cluster"] for p in json.loads((Path(config.planted_ablation_dir) / "decision.json").read_text())["planted"]
    ]
    keep = np.ones(len(m_text), dtype=bool)
    for t_np, m_np in ((t_mini, m_mini), (bge["traffic"], bge["mix"])):
        m_gpu = torch.from_numpy(m_np).cuda()
        for c in planted:
            q = torch.from_numpy(t_np[members[c]]).cuda()
            idx = torch.cat(
                [(q[s : s + 1024] @ m_gpu.T).topk(config.k, dim=1).indices.flatten() for s in range(0, len(q), 1024)]
            )
            keep[np.unique(idx.cpu().numpy())] = False
        del m_gpu
    LOGGER.info(f"deleted {int((~keep).sum())} mix rows for {len(planted)} planted topics")

    def knn(t_np: np.ndarray, m_np: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return density_ratios_batched(
            t_np,
            t_hash,
            np.arange(len(rows)),
            torch.from_numpy(t_np).cuda(),
            torch.from_numpy(t_hash).cuda(),
            torch.from_numpy(m_np[sel]).cuda(),
            torch.from_numpy(m_hash[sel]).cuda(),
            k=config.k,
            return_nearest_mix=True,
        )

    full = np.ones(len(m_text), dtype=bool)
    scores: dict[str, dict[str, tuple]] = {}
    timing = {}
    for setting, sel in (("full", full), ("ablated", keep)):
        sub_text = [m_text[i] for i in np.where(sel)[0]]
        t0 = time.time()
        scores.setdefault("A_minilm_knn", {})[setting] = knn(t_mini, m_mini, sel)
        timing[f"A_{setting}"] = time.time() - t0
        t0 = time.time()
        scores.setdefault("B_dsir", {})[setting] = (dsir_scores(t_text, sub_text), None)
        timing[f"B_{setting}"] = time.time() - t0
        t0 = time.time()
        scores.setdefault("C_classifier", {})[setting] = (
            finetune_logits(t_text, sub_text, fold_t, fold_m[sel], MINILM[0], MINILM[1], seed=config.seed),
            None,
        )
        timing[f"C_{setting}"] = time.time() - t0
        t0 = time.time()
        scores.setdefault("D_bge_knn", {})[setting] = knn(bge["traffic"], bge["mix"], sel)
        timing[f"D_{setting}"] = time.time() - t0
        LOGGER.info(f"{setting} scores done: {timing}")

    def topic_order(s: np.ndarray, near: np.ndarray | None) -> tuple[list[int], np.ndarray]:
        med = {c: float(np.median(s[members[c]])) for c in order}
        tie = {c: float(np.median(near[members[c]])) if near is not None else 0.0 for c in order}
        return sorted(order, key=lambda c: (-med[c], tie[c])), np.array([med[c] for c in order])

    lab = {}
    for d in config.label_dirs:
        for p in Path(d).glob("*.json"):
            lab[int(p.stem.split("_c")[1])] = json.loads(p.read_text())["verdict"]
    lab = {c: v for c, v in lab.items() if v in ("real_gap", "partly", "covered") and c in members}
    ys = np.array([v != "covered" for v in lab.values()])

    res, vectors = {}, {}
    for name, by in scores.items():
        ranking, _ = topic_order(*by["ablated"])
        _, vec_full = topic_order(*by["full"])
        vectors[name] = vec_full
        pos = {str(c): ranking.index(c) + 1 for c in planted}
        res[name] = {
            "planted_in_top": int(sum(p <= config.top_n for p in pos.values())),
            "planted_positions": pos,
            "label_auc": round(float(roc_auc_score(ys, [vec_full[order.index(c)] for c in lab])), 4),
        }
    base = res["A_minilm_knn"]
    for name in ("B_dsir", "C_classifier", "D_bge_knn"):
        r = res[name]
        improves = r["planted_in_top"] >= base["planted_in_top"] + config.planted_margin or (
            r["label_auc"] >= base["label_auc"] + config.auc_margin
            and r["planted_in_top"] >= base["planted_in_top"] - config.planted_slack
        )
        r["verdict"] = "improves_on_A" if improves else "does_not_replace_A"
    names = list(vectors)
    decision = {
        "scores": res,
        "spearman_full": {
            f"{a}~{b}": round(float(spearmanr(vectors[a], vectors[b]).statistic), 4)
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        },
        "mix_rows_deleted": int((~keep).sum()),
        "n_labelled": len(lab),
        "seconds": {k: round(v) for k, v in timing.items()},
    }
    write_json(decision, run / "decision.json")
    print(json.dumps(decision, indent=1))
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
