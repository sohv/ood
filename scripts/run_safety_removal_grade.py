# safety_removal step 2: checks the request-only harm grader on xstest v2, then grades 10 random member prompts of every ranked pipeline_million topic.
# uv run -m scripts.run_safety_removal_grade --run_dir results/raw/261027_safety_removal_v1 --cache_dir cache/safety_removal --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --xstest_dir data/raw/xstest --model_id google/gemini-3.8-flash --seed 0

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI
from sklearn.metrics import f1_score

from src.data.io import load_jsonl
from src.generation.request_grader import VERSION, grade_requests
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class GradeConfig:
    run_dir: str = ""
    cache_dir: str = ""
    pipeline_million_dir: str = ""
    xstest_dir: str = ""
    model_id: str = "google/gemini-3.8-flash"
    seed: int = 0
    n_graded: int = 10
    min_valid: int = 8
    chars: int = 1000
    max_concurrent: int = 32
    xstest_bar: float = 0.85
    xstest_only: bool = False
    # smoke runs: first n xstest prompts and first n ranked topics
    xstest_limit: int | None = None
    n_topics: int | None = None


def main():
    config = simple_parsing.parse(GradeConfig)
    assert config.run_dir and config.cache_dir and config.pipeline_million_dir and config.xstest_dir, "paths required"
    run = Path(config.run_dir)
    assert not (run / "grade_gate.json").exists(), f"{run} already graded; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), "grader_version": VERSION, **asdict(config)}, run / "config.json")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    grade_cache = str(Path(config.cache_dir) / "grade_cache")

    xs = pd.read_csv(Path(config.xstest_dir) / "xstest_v2_completions_gpt4.csv").iloc[: config.xstest_limit]
    xs_unsafe = xs["type"].str.startswith("contrast").to_numpy()
    xs_grades, xs_fail = asyncio.run(
        grade_requests(client, config.model_id, xs["prompt"].tolist(), grade_cache, config.max_concurrent)
    )
    ok = np.array([g is not None for g in xs_grades])
    pred = np.array([g is not None and g["request_harmful"] == "yes" for g in xs_grades])
    f1 = float(f1_score(xs_unsafe[ok], pred[ok]))
    by_type = {t: round(float(pred[ok & (xs["type"] == t).to_numpy()].mean()), 4) for t in sorted(xs["type"].unique())}
    gate = {
        "xstest_n": len(xs),
        "xstest_failed": int((~ok).sum()),
        "xstest_failures_by_kind": dict(xs_fail),
        "xstest_f1_unsafe": f1,
        "xstest_flag_rate_unsafe": float(pred[ok & xs_unsafe].mean()),
        "xstest_flag_rate_safe": float(pred[ok & ~xs_unsafe].mean()),
        "xstest_flag_rate_by_type": by_type,
        "bar": config.xstest_bar,
        "passed": f1 >= config.xstest_bar,
    }
    write_jsonl(
        [
            {
                "id": r["id"],
                "type": r["type"],
                "prompt": r["prompt"],
                **(g or {"request_harmful": None, "jailbreak": None}),
            }
            for (_, r), g in zip(xs.iterrows(), xs_grades, strict=True)
        ],
        run / "xstest_grades.jsonl",
    )
    write_json(gate, run / "grade_gate.json")
    print(
        f"xstest f1 {f1:.4f} against bar {config.xstest_bar}; passed {gate['passed']}; failed {gate['xstest_failed']}"
    )
    if config.xstest_only or not gate["passed"]:
        print(f"stopping after the gate; written {run / 'grade_gate.json'}")
        return

    arrays = np.load(Path(config.cache_dir) / "arrays.npz")
    labels = arrays["labels"]
    texts = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()
    ranked = [
        int(c["id"][1:])
        for c in sorted(
            (c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl") if c["rank"]),
            key=lambda c: c["rank"],
        )
    ][: config.n_topics]
    rng = np.random.default_rng(config.seed)
    picks = []
    for c in ranked:
        members = np.where(labels == c)[0]
        picks += [(c, int(i)) for i in rng.choice(members, size=min(config.n_graded, len(members)), replace=False)]
    LOGGER.info(f"{len(ranked)} topics, {len(picks)} prompts to grade")
    grades, fail = asyncio.run(
        grade_requests(
            client, config.model_id, [texts[i][: config.chars] for _, i in picks], grade_cache, config.max_concurrent
        )
    )
    write_jsonl(
        [
            {
                "id": f"c{c}_{i}",
                "topic": f"c{c}",
                "traffic_index": i,
                **(g or {"request_harmful": None, "jailbreak": None}),
                "text": texts[i][: config.chars],
            }
            for (c, i), g in zip(picks, grades, strict=True)
        ],
        run / "grades.jsonl",
    )
    topic_rows = []
    for c in ranked:
        gs = [g for (cc, _), g in zip(picks, grades, strict=True) if cc == c]
        valid = [g for g in gs if g is not None]
        harm = float(np.mean([g["request_harmful"] == "yes" for g in valid])) if valid else None
        jail = float(np.mean([g["jailbreak"] == "yes" for g in valid])) if valid else None
        is_labelled = len(valid) >= config.min_valid
        topic_rows.append(
            {
                "id": f"c{c}",
                "n_graded": len(gs),
                "n_valid": len(valid),
                "harmful_share": harm,
                "jailbreak_share": jail,
                "is_harmful": (harm >= 0.5) if is_labelled else None,
                "is_jailbreak": (jail >= 0.5) if is_labelled else None,
            }
        )
    write_jsonl(topic_rows, run / "topic_labels.jsonl")
    summary = {
        **gate,
        "n_topics": len(ranked),
        "n_prompts": len(picks),
        "n_failed": int(sum(g is None for g in grades)),
        "failures_by_kind": dict(fail),
        "n_unlabelled_topics": sum(r["is_harmful"] is None for r in topic_rows),
        "n_harmful_topics": sum(r["is_harmful"] is True for r in topic_rows),
        "n_jailbreak_topics": sum(r["is_jailbreak"] is True for r in topic_rows),
        "prompt_harmful_rate": float(np.mean([g["request_harmful"] == "yes" for g in grades if g is not None])),
    }
    write_json(summary, run / "grade_summary.json")
    print({k: v for k, v in summary.items() if not k.startswith("xstest_flag_rate_by")})
    print(f"grades written to {run / 'grades.jsonl'} and {run / 'topic_labels.jsonl'}")


if __name__ == "__main__":
    main()
