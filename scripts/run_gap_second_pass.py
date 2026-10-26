# exp59: adds the exp54 fix prompts (or control prompts) to the mix side, rescores every topic, and reruns the exp41 skip loop.
# uv run -m scripts.run_gap_second_pass --run_dir results/raw/261026_exp59_second_pass_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp54_data results/raw/261024_exp54_data_v1 --naming_cache results/raw/261022_exp41_gap_loop_v1/naming_cache --seed 0

import asyncio
import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI
from scipy.stats import spearmanr

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class SecondPassConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp54_data: str = ""
    naming_cache: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    k: int = 50
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    same_gap: float = 0.85
    template_chars: int = 60
    template_share: float = 0.3
    leave_bar: int = 20
    control_max: int = 3
    flag_max: int = 6
    repro_tol: float = 0.01
    repro_share: float = 0.99


def main():
    config = simple_parsing.parse(SecondPassConfig)
    assert config.run_dir and config.exp37_dir and config.exp54_data, "run_dir, exp37_dir, exp54_data required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    if config.naming_cache:
        shutil.copytree(config.naming_cache, run / "naming_cache")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    clusters = {c["id"]: c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}
    LOGGER.info(f"traffic {len(rows)}, mix {len(m_np)}, {len(order)} ranked topics")

    added = {name: load_jsonl(Path(config.exp54_data) / f"train_{name}2.jsonl") for name in ("gap", "control")}
    for name, recs in added.items():
        for r in recs:
            assert rows[r["row"]]["text"] == r["text"], f"{name} row {r['row']} differs from the pool"
    train_rows = np.array([r["row"] for recs in added.values() for r in recs])
    targets = sorted({r["topic"] for r in added["gap"]})
    is_train = np.zeros(len(rows), dtype=bool)
    is_train[train_rows] = True

    emb = t_np.astype(np.float32)
    cent = {c: centroid(emb[members[c]]) for c in order}
    sig: dict[int, str | None] = {}
    for c in order:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None

    t_gpu, t_hash_gpu = torch.from_numpy(t_np).cuda(), torch.from_numpy(t_hash).cuda()
    ratios, nears = {}, {}
    for arm in ("none", "gap", "control"):
        idx = np.array([r["row"] for r in added[arm]], dtype=np.int64) if arm != "none" else np.zeros(0, np.int64)
        m_gpu = torch.from_numpy(np.concatenate([m_np, t_np[idx].astype(m_np.dtype)])).cuda()
        mh_gpu = torch.from_numpy(np.concatenate([m_hash, t_hash[idx]])).cuda()
        ratios[arm], nears[arm] = density_ratios_batched(
            t_np, t_hash, np.arange(len(rows)), t_gpu, t_hash_gpu, m_gpu, mh_gpu, k=config.k, return_nearest_mix=True
        )
        np.save(run / f"ratio_{arm}.npy", ratios[arm].astype(np.float32))
        del m_gpu, mh_gpu
        torch.cuda.empty_cache()
        LOGGER.info(f"{arm}: mix {len(m_np) + len(idx)}, median ratio {np.median(ratios[arm]):.4f}")

    # validity: the none arm must reproduce exp37's per-topic medians
    repro = np.array([abs(float(np.median(ratios["none"][members[c]])) - clusters[f"c{c}"]["ratio"]) for c in order])
    repro_share = float(np.mean(repro <= config.repro_tol))
    subset_medians = [
        float(np.median(ratios["none"][rng.choice(len(rows), config.subset_size, replace=False)]))
        for _ in range(config.n_random_subsets)
    ]
    line = config.ratio_multiple * float(np.median(subset_medians))

    # topic score leaves out every training row of both arms, so the arms score the same prompts
    def topic_scores(arm: str) -> tuple[dict[int, float], dict[int, float]]:
        keep = {c: members[c][~is_train[members[c]]] for c in order}
        return (
            {c: float(np.median(ratios[arm][keep[c]])) for c in order},
            {c: float(np.median(nears[arm][keep[c]])) for c in order},
        )

    scores = {arm: topic_scores(arm) for arm in ratios}
    is_valid = repro_share >= config.repro_share and all(scores["none"][0][c] >= line for c in targets)
    below = {arm: [c for c in targets if scores[arm][0][c] < line] for arm in ratios}
    n_gap, n_ctrl = len(below["gap"]), len(below["control"])
    verdict = (
        "fixed_topics_leave"
        if n_gap >= config.leave_bar and n_ctrl <= config.control_max
        else "score_still_flags"
        if n_gap <= config.flag_max
        else "partly"
    )
    sizes = [len(members[c]) for c in targets]
    rho = spearmanr(sizes, [scores["gap"][0][c] for c in targets]).statistic
    LOGGER.info(f"line {line:.4f}, repro {repro_share:.4f}, below line gap {n_gap} control {n_ctrl}: {verdict}")

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

    def same(a: int, b: int) -> bool:
        return float(cent[a] @ cent[b]) >= config.same_gap or (sig[a] is not None and sig[a] == sig[b])

    def skip_loop(arm: str) -> dict:
        score, near = scores[arm]

        def confirmed(c: int) -> bool:
            shared = names[c].get("shared")
            return shared is not None and shared >= config.shared_needed and score[c] >= line

        remaining = sorted(order, key=lambda c: (-score[c], near[c]))
        reviewed, conf = [], []
        for _ in range(config.rounds):
            batch = remaining[: config.per_round]
            review(batch)
            new_conf = [c for c in batch if confirmed(c)]
            reviewed += batch
            conf += new_conf
            rest = [c for c in remaining if c not in batch]
            remaining = [c for c in rest if not any(same(c, g) for g in new_conf)]
        gaps: list[list[int]] = []
        for c in conf:
            hit = next((g for g in gaps if same(c, g[0])), None)
            if hit is None:
                gaps.append([c])
            else:
                hit.append(c)
        return {"reviewed": reviewed, "confirmed": conf, "distinct_gaps": len(gaps)}

    loops = {arm: skip_loop(arm) for arm in ratios}
    none_conf = set(loops["none"]["confirmed"])
    decision = {
        "valid": is_valid,
        "verdict": verdict,
        "line": round(line, 4),
        "none_reproduces_exp37_share": round(repro_share, 4),
        "targets_below_line": {arm: len(b) for arm, b in below.items()},
        "spearman_size_vs_gap_score": round(float(rho), 4),
        "targets": [
            {
                "topic": c,
                "name": names.get(c, {}).get("name"),
                "size": len(members[c]),
                **{f"score_{arm}": round(scores[arm][0][c], 4) for arm in ratios},
            }
            for c in sorted(targets, key=lambda c: len(members[c]))
        ],
        "loop": {
            arm: {
                "confirmed_topics": len(lp["confirmed"]),
                "distinct_gaps": lp["distinct_gaps"],
                "targets_confirmed": sum(c in lp["confirmed"] for c in targets),
                "targets_reviewed": sum(c in lp["reviewed"] for c in targets),
                "confirmed_not_in_none": sorted(set(lp["confirmed"]) - none_conf),
                "none_confirmed_lost": sorted(none_conf - set(lp["confirmed"])),
            }
            for arm, lp in loops.items()
        },
        "null_names": sum(names[c].get("shared") is None for c in names),
        "added_rows": {arm: len(r) for arm, r in added.items()},
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": f"{arm}_{c}",
                "arm": arm,
                "topic": c,
                "name": names[c].get("name"),
                "confirmed": c in lp["confirmed"],
            }
            for arm, lp in loops.items()
            for c in lp["reviewed"]
        ],
        run / "reviewed_topics.jsonl",
    )
    print(json.dumps({k: decision[k] for k in ("valid", "verdict", "line", "targets_below_line", "loop")}, indent=1))
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
