# planted_ablation: plants 10 gaps by deleting their look-alikes from the mix, then counts how many the skip loop, the iterative classifier and random topics find.
# uv run -m scripts.run_planted_gaps --run_dir results/raw/261022_planted_ablation_planted_gaps_v2 --ablation neighbours --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --cache_dirs results/raw/261022_deep_gap_loop_v1 results/raw/261022_iterative_gap_loop_gap_iterative_v1 results/raw/261022_classifier_gap_loop_gap_classifier_v2 --seed 0

import asyncio
import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import crossfit_logits, density_ratios_batched, text_hash
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PlantedConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    cache_dirs: list[str] = field(default_factory=list)
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    n_planted: int = 10
    min_size: int = 200
    max_size: int = 3000
    ratio_band: tuple[float, float] = (0.5, 2.0)
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    k: int = 50
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    same_gap: float = 0.85
    theme_cos: float = 0.70
    template_chars: int = 60
    template_share: float = 0.3
    line_rank: int = 293
    min_valid: int = 8
    works: int = 6
    fails: int = 2
    better_by: int = 3
    # v1 used nearest-centroid removal; the amendment removes each planted prompt's k nearest mix rows
    ablation: str = "centroid"


def main():
    config = simple_parsing.parse(PlantedConfig)
    assert config.run_dir and config.pipeline_million_dir, "run_dir and pipeline_million_dir required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    for d in config.cache_dirs:
        shutil.copytree(Path(d) / "naming_cache", run / "naming_cache", dirs_exist_ok=True)
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    ratio37 = np.load(Path(config.pipeline_million_dir) / "ratio.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)

    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}
    emb = t_np.astype(np.float32)
    cent = {c: centroid(emb[members[c]]) for c in order}
    sig: dict[int, str | None] = {}
    for c in order:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None

    # plant: covered ordinary topics, then delete every mix row whose nearest topic centroid is a planted one
    pool_median = float(np.median(ratio37))
    lo, hi = (b * pool_median for b in config.ratio_band)
    candidates = [
        c
        for c in order
        if sig[c] is None
        and config.min_size <= len(members[c]) <= config.max_size
        and lo <= float(np.median(ratio37[members[c]])) <= hi
    ]
    planted = sorted(rng.choice(candidates, size=config.n_planted, replace=False).tolist())
    m_gpu_full = torch.from_numpy(m_np).cuda()
    keep = np.ones(len(m_np), dtype=bool)
    removed = {}
    if config.ablation == "neighbours":
        # amendment: drop every mix row among the k nearest mix rows of any planted prompt
        for c in planted:
            q = torch.from_numpy(t_np[members[c]]).cuda()
            idx = torch.cat(
                [
                    (q[s : s + 1024] @ m_gpu_full.T).topk(config.k, dim=1).indices.flatten()
                    for s in range(0, len(q), 1024)
                ]
            )
            hit = np.unique(idx.cpu().numpy())
            removed[c] = int(keep[hit].sum())
            keep[hit] = False
    else:
        cmat = torch.from_numpy(np.stack([cent[c] for c in order])).cuda().half()
        nearest_topic = (
            torch.cat([(m_gpu_full[s : s + 65536] @ cmat.T).argmax(1) for s in range(0, len(m_np), 65536)])
            .cpu()
            .numpy()
        )
        keep = ~np.isin(nearest_topic, [order.index(c) for c in planted])
        removed = {c: int(np.sum(nearest_topic == order.index(c))) for c in planted}
    m_np, m_hash = m_np[keep], m_hash[keep]
    del m_gpu_full
    LOGGER.info(f"{len(candidates)} candidates, planted {planted}, removed {sum(removed.values())} mix rows")

    t_gpu, m_gpu = torch.from_numpy(t_np).cuda(), torch.from_numpy(m_np).cuda()
    ratio, near = density_ratios_batched(
        t_np,
        t_hash,
        np.arange(len(rows)),
        t_gpu,
        torch.from_numpy(t_hash).cuda(),
        m_gpu,
        torch.from_numpy(m_hash).cuda(),
        k=config.k,
        return_nearest_mix=True,
    )
    knn_score = {c: float(np.median(ratio[members[c]])) for c in order}
    knn_near = {c: float(np.median(near[members[c]])) for c in order}
    subset_medians = [
        float(np.median(ratio[rng.choice(len(rows), config.subset_size, replace=False)]))
        for _ in range(config.n_random_subsets)
    ]
    knn_line = config.ratio_multiple * float(np.median(subset_medians))
    manip = {c: knn_score[c] >= knn_line for c in planted}
    is_valid = sum(manip.values()) >= config.min_valid
    LOGGER.info(f"knn line {knn_line:.2f}; planted over line {sum(manip.values())} of {len(planted)}")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    names: dict[int, dict] = {}

    def review(batch: list[int]) -> None:
        todo = [c for c in batch if c not in names]
        prompts = [
            naming_prompt(
                [rows[i]["text"] for i in members[c]], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN
            )
            for c in todo
        ]
        replies = asyncio.run(name_clusters(client, config.namer_model, prompts, str(run / "naming_cache")))
        for c, rep in zip(todo, replies, strict=True):
            names[c] = parse_naming(rep)

    def coherent(c: int) -> bool:
        shared = names[c].get("shared")
        return shared is not None and shared >= config.shared_needed

    def same(a: int, b: int) -> bool:
        return float(cent[a] @ cent[b]) >= config.same_gap or (sig[a] is not None and sig[a] == sig[b])

    # skip loop on the ablated mix
    remaining = sorted(order, key=lambda c: (-knn_score[c], knn_near[c]))
    skip_reviewed, skip_conf = [], []
    for _ in range(config.rounds):
        batch = remaining[: config.per_round]
        review(batch)
        new_conf = [c for c in batch if coherent(c) and knn_score[c] >= knn_line]
        skip_reviewed += batch
        skip_conf += new_conf
        rest = [c for c in remaining if c not in batch]
        remaining = [c for c in rest if not any(same(c, g) for g in new_conf)]

    # iterative classifier on the ablated mix
    x_t, x_m = t_gpu.half(), m_gpu.half()
    fold_t, fold_m = (t_hash & 1).astype(np.int64), (m_hash & 1).astype(np.int64)
    moved = np.zeros(len(rows), dtype=bool)

    def clf_scores() -> dict[int, float]:
        logit = crossfit_logits(x_t, x_m, moved, fold_t, fold_m, seed=config.seed)
        return {c: float(np.median(logit[members[c]])) for c in order}

    score = clf_scores()
    clf_line = sorted(score.values(), reverse=True)[config.line_rank - 1]
    clf_reviewed, clf_conf = [], []
    for _ in range(config.rounds):
        batch = sorted((c for c in order if c not in clf_reviewed), key=lambda c: -score[c])[: config.per_round]
        review(batch)
        new_conf = [c for c in batch if coherent(c) and score[c] >= clf_line]
        for c in new_conf:
            moved[members[c]] = True
        clf_reviewed += batch
        clf_conf += new_conf
        score = clf_scores()

    rand = rng.choice(order, size=config.rounds * config.per_round, replace=False).tolist()
    review(rand)
    rand_conf = [c for c in rand if coherent(c) and knn_score[c] >= knn_line]

    def merge(conf: list[int]) -> int:
        groups: list[int] = []
        for c in conf:
            if not any(
                float(cent[c] @ cent[g]) >= config.theme_cos or (sig[c] is not None and sig[c] == sig[g])
                for g in groups
            ):
                groups.append(c)
        return len(groups)

    arms = {
        "skip": (skip_reviewed, skip_conf),
        "iterative_classifier": (clf_reviewed, clf_conf),
        "random": (rand, rand_conf),
    }
    found = {a: [c for c in planted if c in cf] for a, (_, cf) in arms.items()}
    n_found = {a: len(f) for a, f in found.items()}

    def judge(n: int) -> str:
        return "finds" if n >= config.works else "fails" if n <= config.fails else "ambiguous"

    diff = n_found["iterative_classifier"] - n_found["skip"]
    decision = {
        "valid": is_valid,
        "manipulation_over_line": int(sum(manip.values())),
        "planted_found": n_found,
        "verdict_per_method": {a: judge(n_found[a]) for a in ("skip", "iterative_classifier")},
        "comparison": "classifier_better"
        if diff >= config.better_by
        else "skip_better"
        if -diff >= config.better_by
        else "tie",
        "themes_0.70": {a: merge(cf) for a, (_, cf) in arms.items()},
        "knn_line": round(knn_line, 4),
        "clf_line": round(clf_line, 4),
        "mix_rows_removed": removed,
        "planted": [
            {
                "cluster": c,
                "name": names.get(c, {}).get("name"),
                "size": len(members[c]),
                "ratio_before": round(float(np.median(ratio37[members[c]])), 4),
                "ratio_after": round(knn_score[c], 4),
                "over_knn_line": manip[c],
                "position": {a: (r.index(c) + 1 if c in r else None) for a, (r, _) in arms.items()},
                "confirmed": {a: c in cf for a, (_, cf) in arms.items()},
            }
            for c in planted
        ],
        "null_names": sum(names[c].get("shared") is None for c in names),
    }
    missing = [c for c in planted if c not in names]
    if missing:
        review(missing)
        for p in decision["planted"]:
            p["name"] = names[p["cluster"]].get("name")
    write_json(decision, run / "decision.json")
    for p in decision["planted"]:
        print(p["name"], p["ratio_before"], "->", p["ratio_after"], p["position"], p["confirmed"])
    print(f"{n_found=} valid={is_valid} comparison={decision['comparison']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
