# classifier_gap_loop: iterative two-sided classifier gap loop (retrain with found gaps on the training side) against walking its round-0 ranking.
# uv run -m scripts.run_gap_classifier --run_dir results/raw/261022_classifier_gap_loop_gap_classifier_v2 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --iterative_gap_loop_dir results/raw/261022_iterative_gap_loop_gap_iterative_v1 --line_rank 293 --seed 0

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
from sklearn.metrics import roc_auc_score

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import crossfit_logits, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class GapClassifierConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    deep_gap_loop_dir: str = ""
    iterative_gap_loop_dir: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    epochs: int = 3
    batch: int = 65536
    lr: float = 0.01
    l2: float = 1e-4
    n_random_subsets: int = 1000
    subset_size: int = 500
    # amendment: when set, the line is the round-0 score of this-ranked topic instead of the log 3 rule
    line_rank: int = 0
    theme_cos: float = 0.70
    template_chars: int = 60
    template_share: float = 0.3
    adds_factor: float = 1.2
    nothing_factor: float = 1.05


def main():
    config = simple_parsing.parse(GapClassifierConfig)
    assert (
        config.run_dir and config.pipeline_million_dir and config.deep_gap_loop_dir and config.iterative_gap_loop_dir
    ), "input and run dirs required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    # earlier names are reused so a topic named before keeps its name
    for d in (config.deep_gap_loop_dir, config.iterative_gap_loop_dir):
        shutil.copytree(Path(d) / "naming_cache", run / "naming_cache", dirs_exist_ok=True)
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
    assert len(labels) == len(rows), "pool does not match pipeline_million's labels"
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    fold_t, fold_m = (t_hash & 1).astype(np.int64), (m_hash & 1).astype(np.int64)
    x_t, x_m = torch.from_numpy(t_np).cuda(), torch.from_numpy(m_np).cuda()
    moved = np.zeros(len(rows), dtype=bool)

    def fit() -> np.ndarray:
        return crossfit_logits(
            x_t, x_m, moved, fold_t, fold_m, config.epochs, config.batch, config.lr, config.l2, config.seed
        )

    logit0 = fit()
    # held-out auc of the round-0 classifier, mix rows scored by the fold model that did not see them
    w_mix = crossfit_logits(
        x_m, x_t, np.zeros(len(m_np), dtype=bool), fold_m, fold_t, config.epochs, config.batch, config.lr, config.l2
    )
    auc0 = float(roc_auc_score(np.r_[np.ones(len(logit0)), np.zeros(len(w_mix))], np.r_[logit0, -w_mix]))
    LOGGER.info(f"round-0 held-out auc {auc0:.4f}")

    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}
    emb = t_np.astype(np.float32)
    cent = {c: centroid(emb[members[c]]) for c in order}
    sig: dict[int, str | None] = {}
    for c in order:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None

    def topic_scores(logit: np.ndarray) -> dict[int, float]:
        return {c: float(np.median(logit[members[c]])) for c in order}

    subset_medians = [
        float(np.median(logit0[rng.choice(len(rows), config.subset_size, replace=False)]))
        for _ in range(config.n_random_subsets)
    ]
    line = float(np.log(3) + np.median(subset_medians))
    if config.line_rank:
        line = sorted(topic_scores(logit0).values(), reverse=True)[config.line_rank - 1]
    LOGGER.info(f"gap line {line:.3f} logits")

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

    def is_confirmed(c: int, s: float) -> bool:
        shared = names[c].get("shared")
        return shared is not None and shared >= config.shared_needed and s >= line

    # jailbreak topics as named by deep_gap_loop or iterative_gap_loop, for the mechanism check
    jb = {
        int(t["id"][1:])
        for d in (config.deep_gap_loop_dir, config.iterative_gap_loop_dir)
        for t in load_jsonl(Path(d) / "reviewed_topics.jsonl")
        if "jailbreak" in (t["name"] or "").lower()
    }

    score0 = topic_scores(logit0)
    score = dict(score0)
    reviewed, conf_iter, score_at_review, per_round = [], [], {}, []
    for rnd in range(1, config.rounds + 1):
        batch = sorted((c for c in order if c not in score_at_review), key=lambda c: -score[c])[: config.per_round]
        review(batch)
        new_conf = []
        for c in batch:
            score_at_review[c] = score[c]
            if is_confirmed(c, score[c]):
                new_conf.append(c)
                moved[members[c]] = True
        reviewed += batch
        conf_iter += new_conf
        new_score = topic_scores(fit())
        jb_hit = [c for c in new_conf if "jailbreak" in (names[c].get("name") or "").lower()]
        others = [c for c in jb if c not in new_conf]
        jb_change = float(np.mean([new_score[c] - score[c] for c in others])) if jb_hit and others else None
        score = new_score
        over = sum(score[c] >= line for c in order if c not in score_at_review)
        per_round.append(
            {
                "id": f"round_{rnd}",
                "reviewed": batch,
                "confirmed": new_conf,
                "moved_rows": int(moved.sum()),
                "unreviewed_over_line": over,
                "jailbreak_confirmed": jb_hit,
                "other_jailbreak_mean_change": None if jb_change is None else round(jb_change, 4),
            }
        )
        LOGGER.info(
            f"round {rnd}: {len(new_conf)} confirmed, {moved.sum()} moved, {over} over line, jb change {jb_change}"
        )

    walk = sorted(order, key=lambda c: -score0[c])[: config.rounds * config.per_round]
    review(walk)
    conf_walk = [c for c in walk if is_confirmed(c, score0[c])]
    d41 = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())
    t41 = {int(t["id"][1:]): t for t in load_jsonl(Path(config.deep_gap_loop_dir) / "reviewed_topics.jsonl")}
    conf41 = [c for r in d41["loop_rounds"] for c in r["reviewed"] if t41[c]["confirmed"]]

    def merge(conf: list[int]) -> list[list[int]]:
        groups: list[list[int]] = []
        for c in conf:
            hit = next(
                (
                    g
                    for g in groups
                    if float(cent[c] @ cent[g[0]]) >= config.theme_cos or (sig[c] is not None and sig[c] == sig[g[0]])
                ),
                None,
            )
            if hit is None:
                groups.append([c])
            else:
                hit.append(c)
        return groups

    name_of = {**{c: t41[c]["name"] for c in t41}, **{c: names[c].get("name") for c in names}}
    arms = {"iterative": conf_iter, "walk": conf_walk, "skip_deep_gap_loop": conf41}
    themes = {a: merge(cf) for a, cf in arms.items()}
    n = {a: len(g) for a, g in themes.items()}
    if n["iterative"] >= config.adds_factor * n["walk"]:
        verdict = "iterative_adds"
    elif n["iterative"] <= config.nothing_factor * n["walk"]:
        verdict = "iterative_adds_nothing"
    else:
        verdict = "ambiguous"

    def count_names(cf: list[int], words: tuple[str, ...]) -> int:
        return sum(any(w in (name_of[c] or "").lower() for w in words) for c in cf)

    decision = {
        "verdict": verdict,
        "themes_0.70": n,
        "confirmed_topics": {a: len(cf) for a, cf in arms.items()},
        "jailbreak_names": {a: count_names(cf, ("jailbreak",)) for a, cf in arms.items()},
        "sexual_names": {a: count_names(cf, ("sexual", "erotic")) for a, cf in arms.items()},
        "round0_heldout_auc": round(auc0, 4),
        "gap_line_logit": round(line, 4),
        "overlap_iterative_walk": len(set(conf_iter) & set(conf_walk)),
        "null_names": sum(names[c].get("shared") is None for c in set(reviewed) | set(walk)),
        "rounds": per_round,
        "themes": {a: [[{"cluster": c, "name": name_of[c]} for c in g] for g in gl] for a, gl in themes.items()},
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "cluster": c,
                "size": len(members[c]),
                "score0": round(score0[c], 4),
                "score_at_review": round(score_at_review[c], 4) if c in score_at_review else None,
                "wildchat_share": round(float(np.mean(source[members[c]] == "wildchat")), 4),
                **{k: names[c].get(k) for k in ("name", "description", "shared")},
                "arms": [a for a, r in (("iterative", reviewed), ("walk", walk)) if c in r],
                "confirmed": c in conf_iter,
                "examples": [
                    rows[i]["text"][:400] for i in rng.choice(members[c], size=min(10, len(members[c])), replace=False)
                ],
            }
            for c in dict.fromkeys(reviewed + walk)
        ],
        run / "reviewed_topics.jsonl",
    )
    for a, cf in arms.items():
        print(f"{a}: {n[a]} themes from {len(cf)} confirmed topics")
    print(f"{decision['jailbreak_names']=} {decision['sexual_names']=} auc0={auc0:.4f}")
    print(f"verdict {verdict}; decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
