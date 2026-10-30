# iterative_gap_loop: iterative gap loop that moves each confirmed topic's prompts into the reference and re-scores, against deep_gap_loop's skip loop.
# uv run -m scripts.run_gap_iterative --run_dir results/raw/261022_iterative_gap_loop_gap_iterative_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --seed 0

import asyncio
import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from openai import AsyncOpenAI

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo, normalise
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import ratios_from_knn, text_hash, union_knn
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class GapIterativeConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    deep_gap_loop_dir: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    k: int = 50
    theme_cos: float = 0.70
    same_gap: float = 0.85
    template_chars: int = 60
    template_share: float = 0.3
    adds_factor: float = 1.2
    nothing_factor: float = 1.05


def main():
    config = simple_parsing.parse(GapIterativeConfig)
    assert (
        config.run_dir and config.pipeline_million_dir and config.deep_gap_loop_dir
    ), "run_dir, pipeline_million_dir, deep_gap_loop_dir required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    # deep_gap_loop's names are reused so a topic both arms review gets the same name
    shutil.copytree(Path(config.deep_gap_loop_dir) / "naming_cache", run / "naming_cache")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_np, source, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    ratio37 = np.load(Path(config.pipeline_million_dir) / "ratio.npy")
    assert len(labels) == len(rows) == len(ratio37), "pool does not match pipeline_million's labels"

    # mix exactly as pipeline_million built it: unique first user turns of dolci sft then dpo
    sft = pd.read_parquet(c37["sft_path"], columns=["first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    norms = {
        "sft": [normalise(t) for t in sft["first_user"]],
        "dpo": [normalise(t) for t in load_dolci_dpo()["first_user"]],
    }
    mseen, mix_idx = set(), []
    for n in ("sft", "dpo"):
        for i, nm in enumerate(norms[n]):
            if nm not in mseen:
                mseen.add(nm)
                mix_idx.append((n, i))
    m_np = np.concatenate(
        [
            np.load(Path(c37["emb_v1"]) / "emb_sft.npy", mmap_mode="r")[[i for n, i in mix_idx if n == "sft"]],
            np.load(Path(c37["emb_v1"]) / "emb_dpo.npy", mmap_mode="r")[[i for n, i in mix_idx if n == "dpo"]],
        ]
    ).astype(np.float16)
    m_hash = np.array([text_hash(norms[n][i]) for n, i in mix_idx], dtype=np.int64)
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    t_gpu, m_gpu = torch.from_numpy(t_np).cuda(), torch.from_numpy(m_np).cuda()
    nbr = union_knn(
        t_np, t_hash, t_gpu, torch.from_numpy(t_hash).cuda(), m_gpu, torch.from_numpy(m_hash).cuda(), k=config.k
    )
    del t_gpu, m_gpu
    np.save(run / "neighbours.npy", nbr)
    moved = np.zeros(len(rows), dtype=bool)
    ratio0 = ratios_from_knn(nbr, moved, len(m_np))
    # the neighbour table must reproduce pipeline_million's ratios before anything moves
    assert np.allclose(ratio0, ratio37, rtol=1e-3), f"max diff {np.abs(ratio0 - ratio37).max()}"
    LOGGER.info(f"neighbour table built, matches pipeline_million ratios; mix {len(m_np)}")

    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    rank37 = {c: clusters[f"c{c}"]["rank"] for c in order}
    members = {c: np.where(labels == c)[0] for c in order}
    emb = t_np.astype(np.float32)
    cent = {c: centroid(emb[members[c]]) for c in order}
    sig: dict[int, str | None] = {}
    for c in order:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None
    d41 = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())
    ratio_line = d41["ratio_line"]
    t41 = {int(t["id"][1:]): t for t in load_jsonl(Path(config.deep_gap_loop_dir) / "reviewed_topics.jsonl")}

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

    def topic_scores(ratio: np.ndarray) -> dict[int, float]:
        return {c: float(np.median(ratio[members[c]])) for c in order}

    # iterative loop: review the top unreviewed topics by current score, move confirmed topics' prompts to the mix, re-score
    score = topic_scores(ratio0)
    reviewed, conf_iter, score_at_review, per_round = [], [], {}, []
    for rnd in range(1, config.rounds + 1):
        todo = sorted((c for c in order if c not in score_at_review), key=lambda c: (-score[c], rank37[c]))
        batch = todo[: config.per_round]
        review(batch)
        new_conf = []
        for c in batch:
            score_at_review[c] = score[c]
            shared = names[c].get("shared")
            if shared is not None and shared >= config.shared_needed and score[c] >= ratio_line:
                new_conf.append(c)
                moved[members[c]] = True
        reviewed += batch
        conf_iter += new_conf
        score = topic_scores(ratios_from_knn(nbr, moved, len(m_np)))
        over = sum(score[c] >= ratio_line for c in order if c not in score_at_review)
        per_round.append(
            {
                "id": f"round_{rnd}",
                "reviewed": batch,
                "confirmed": new_conf,
                "moved_rows": int(moved.sum()),
                "unreviewed_over_line": over,
            }
        )
        LOGGER.info(
            f"round {rnd}: {len(new_conf)} confirmed, {moved.sum()} rows moved, {over} unreviewed over the line"
        )

    # deep_gap_loop's skip loop as run, with its own names and confirmations
    reviewed41 = [c for r in d41["loop_rounds"] for c in r["reviewed"]]
    conf41 = [c for c in reviewed41 if t41[c]["confirmed"]]

    def merge(conf: list[int], cos: float) -> list[list[int]]:
        groups: list[list[int]] = []
        for c in conf:
            hit = next(
                (
                    g
                    for g in groups
                    if float(cent[c] @ cent[g[0]]) >= cos or (sig[c] is not None and sig[c] == sig[g[0]])
                ),
                None,
            )
            if hit is None:
                groups.append([c])
            else:
                hit.append(c)
        return groups

    name_of = {**{c: t41[c]["name"] for c in reviewed41}, **{c: names[c].get("name") for c in reviewed}}
    arms = {"iterative": conf_iter, "skip": conf41}
    themes = {a: merge(cf, config.theme_cos) for a, cf in arms.items()}
    fine = {a: len(merge(cf, config.same_gap)) for a, cf in arms.items()}
    n = {a: len(g) for a, g in themes.items()}
    if n["iterative"] >= config.adds_factor * n["skip"]:
        verdict = "iterative_adds"
    elif n["iterative"] <= config.nothing_factor * n["skip"]:
        verdict = "iterative_adds_nothing"
    else:
        verdict = "ambiguous"

    def count_names(cf: list[int], words: tuple[str, ...]) -> int:
        return sum(any(w in (name_of[c] or "").lower() for w in words) for c in cf)

    decision = {
        "verdict": verdict,
        "themes_0.70": n,
        "distinct_0.85": fine,
        "confirmed_topics": {a: len(cf) for a, cf in arms.items()},
        "jailbreak_names": {a: count_names(cf, ("jailbreak",)) for a, cf in arms.items()},
        "sexual_names": {a: count_names(cf, ("sexual", "erotic")) for a, cf in arms.items()},
        "overlap_confirmed": len(set(conf_iter) & set(conf41)),
        "ratio_line": ratio_line,
        "null_names": sum(names[c].get("shared") is None for c in reviewed),
        "rounds": per_round,
        "themes": {
            a: [[{"cluster": c, "name": name_of[c], "rank37": rank37[c]} for c in g] for g in gl]
            for a, gl in themes.items()
        },
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "cluster": c,
                "rank37": rank37[c],
                "size": len(members[c]),
                "score_at_review": round(score_at_review[c], 4),
                "wildchat_share": round(float(np.mean(source[members[c]] == "wildchat")), 4),
                **{k: names[c].get(k) for k in ("name", "description", "shared")},
                "confirmed": c in conf_iter,
                "examples": [
                    rows[i]["text"][:400] for i in rng.choice(members[c], size=min(10, len(members[c])), replace=False)
                ],
            }
            for c in reviewed
        ],
        run / "reviewed_topics.jsonl",
    )
    for a, cf in arms.items():
        print(f"{a}: {n[a]} themes at 0.70, {fine[a]} at 0.85, from {len(cf)} confirmed topics")
    print(f"{decision['jailbreak_names']=} {decision['sexual_names']=}")
    print(f"verdict {verdict}; decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
