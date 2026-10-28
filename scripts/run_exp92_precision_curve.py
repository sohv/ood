# exp92 precision curve: exp92's independent coverage judge on every topic in the top 200 of the count and word rankings plus random topics, and precision at 50, 100 and 200 against the user's hand-label precision.
# uv run --no-sync -m scripts.run_exp92_precision_curve --run_dir results/raw/261028_exp92_precision_curve_v1 --cache_dir cache/exp80 --exp80_dir results/raw/261028_exp80_safety_removal_v1 --exp61_dir results/raw/261026_exp61_labels_v1 --cache_judge cache/exp92_judge --seed 0

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.generation.request_grader import grade_requests
from src.metrics.coverage_truth import VALUES, VERSION, bm25_pools, build_prompt, topic_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.plotting import (
    BLUE,
    NEUTRAL,
    ORANGE,
    finish,
    header,
    panel_title,
    percent_axis,
    point_ci,
    reference_band,
    style,
    tidy,
    wilson,
)
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
KS = (50, 100, 200)


@dataclass
class PrecisionCurveConfig:
    run_dir: str = ""
    cache_dir: str = ""
    exp80_dir: str = ""
    exp61_dir: str = ""
    cache_judge: str = ""
    judge_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    n_random: int = 100
    per_topic: int = 10
    k_shown: int = 20
    index_chars: int = 1000
    user_chars: int = 1500
    doc_chars: int = 400
    max_concurrent: int = 24


def main():
    config = simple_parsing.parse(PrecisionCurveConfig)
    assert config.run_dir and config.cache_dir and config.exp80_dir and config.exp61_dir and config.cache_judge
    run = Path(config.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / "decision.json").exists(), f"{run} already has a decision; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    labels = np.load(Path(config.cache_dir) / "arrays.npz")["labels"]
    t_text = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()
    m_text = pd.read_parquet(Path(config.cache_dir) / "mix_text.parquet")["text"].tolist()
    topics = {int(t["id"][1:]): t for t in load_jsonl(Path(config.exp80_dir) / "topics.jsonl")}
    order = {f: sorted(topics, key=lambda c, f=f: topics[c][f"{f}_rank_full"]) for f in ("count", "words")}
    rng = np.random.default_rng(config.seed)
    arms = {"count": order["count"][: max(KS)], "words": order["words"][: max(KS)]}
    arms["random"] = sorted(rng.choice(sorted(topics), size=config.n_random, replace=False).tolist())
    todo = sorted({c for v in arms.values() for c in v})
    # members drawn per topic from its own seed, so a topic gets the same prompts in every arm
    members = {}
    for c in todo:
        m = np.where(labels == c)[0]
        members[c] = np.random.default_rng([config.seed, c]).choice(
            m, size=min(config.per_topic, len(m)), replace=False
        )
    rows_q = [(c, int(i)) for c in todo for i in members[c]]
    LOGGER.info(f"{len(todo)} topics, {len(rows_q)} prompts; building bm25 index")
    pools = bm25_pools(m_text, [t_text[i] for _, i in rows_q], config.k_shown, config.index_chars)
    prompts = [
        build_prompt(t_text[i], [m_text[int(d)] for d in pool], config.user_chars, config.doc_chars)
        for (_, i), pool in zip(rows_q, pools, strict=True)
    ]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    grades, failures = asyncio.run(
        grade_requests(
            client,
            config.judge_model,
            prompts,
            config.cache_judge,
            config.max_concurrent,
            prompt_template="{user}",
            values=VALUES,
            version=VERSION,
        )
    )
    judged = [
        {"id": f"t{i}", "topic": c, "covered": None if g is None else g["covered"] == "yes"}
        for (c, i), g in zip(rows_q, grades, strict=True)
    ]
    write_jsonl(judged, run / "judgements.jsonl")
    n_failed = sum(j["covered"] is None for j in judged)
    LOGGER.info(f"{n_failed} failed judgements {dict(failures)}")
    jdf = pd.DataFrame([j for j in judged if j["covered"] is not None]).astype({"covered": bool})
    share = jdf.groupby("topic")["covered"].mean()
    truth = {c: topic_label(float(s)) for c, s in share.items()}
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "covered_share": round(float(share[c]), 4),
                "truth": truth[c],
                "count_rank": topics[c]["count_rank_full"],
                "words_rank": topics[c]["words_rank_full"],
                "name": topics[c].get("name"),
            }
            for c in sorted(truth)
        ],
        run / "topics.jsonl",
    )

    def prec(cs: list[int], lenient: bool) -> tuple[float, float, float, int]:
        cs = [c for c in cs if c in truth]
        k = sum((truth[c] != "covered") if lenient else (truth[c] == "real_gap") for c in cs)
        return (*wilson(k, len(cs)), len(cs))

    curve = {
        mode: {f: {k: prec(arms[f][:k], mode == "lenient") for k in KS} for f in ("count", "words")}
        | {"random": prec(arms["random"], mode == "lenient")}
        for mode in ("strict", "lenient")
    }
    e61 = json.loads((Path(config.exp61_dir) / "decision.json").read_text())["cumulative"]
    hand = {mode: {k: e61[str(k)][f"{mode}_pooled"] for k in KS} for mode in ("strict", "lenient")}

    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    rows = []
    for ax, mode, title in zip(axes, ("strict", "lenient"), ("Real gaps only", "Real or partly covered"), strict=True):
        r = curve[mode]["random"]
        reference_band(ax, r[0], r[1], r[2], label=f"random topics (n={r[3]})")
        x = np.arange(len(KS))
        point_ci(ax, x - 0.12, [curve[mode]["count"][k][:3] for k in KS], BLUE, label="neighbour count, judge")
        point_ci(ax, x, [curve[mode]["words"][k][:3] for k in KS], ORANGE, label="tf-idf words, judge")
        point_ci(
            ax,
            x + 0.12,
            [hand[mode][k] for k in KS],
            NEUTRAL,
            filled=False,
            label="neighbour count, hand labels (5 per bin)",
        )
        ax.set_xticks(x, [f"top {k}" for k in KS])
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="#888888", lw=0.8, ls=":")
        percent_axis(ax)
        tidy(ax)
        panel_title(ax, title)
        for f in ("count", "words"):
            rows += [[mode, f, k, *[round(v, 4) for v in curve[mode][f][k][:3]], curve[mode][f][k][3]] for k in KS]
        rows += [[mode, "hand", k, *[round(v, 4) for v in hand[mode][k]], ""] for k in KS]
        rows += [[mode, "random", "", *[round(v, 4) for v in r[:3]], r[3]]]
    axes[0].set_ylabel("Share of top-K topics missing from Dolci\n(95% Wilson CI)")
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)
    header(
        fig,
        "Precision of the gap ranking at top 50, 100 and 200\nOLMo-3 Dolci vs WildChat+LMSYS; judge: Claude Sonnet 4.6 on BM25 matches, 10 prompts per topic",
    )
    fig_path = finish(fig, "precision_at_k", rows, ["mode", "ranking", "k", "p", "lo", "hi", "n"], run / "figures")
    decision = {
        "git_hash": get_git_hash(),
        "n_judgements": len(judged),
        "n_failed": n_failed,
        "curve": {
            m: {f: {str(k): v for k, v in d.items()} if f != "random" else d for f, d in c.items()}
            for m, c in curve.items()
        },
        "hand": {m: {str(k): v for k, v in d.items()} for m, d in hand.items()},
        "figure": str(fig_path),
    }
    write_json(decision, run / "decision.json")
    print(json.dumps(decision["curve"], indent=1))
    print(f"{n_failed} failed of {len(judged)}; figure {fig_path}; decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
