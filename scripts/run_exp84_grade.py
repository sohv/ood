# exp84 step 2: checks the kind grader on dolci's own source labels, then grades exp80's 10 sampled member prompts of every ranked topic for coding, non-english and tool use.
# uv run -m scripts.run_exp84_grade --run_dir results/raw/261028_exp84_kind_labels_v1 --cache_dir cache/exp84 --exp80_cache_dir cache/exp80 --exp80_run_dir results/raw/261028_exp80_safety_removal_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --model_id google/gemini-3.8-flash --seed 0

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
from src.generation.kind_grader import KINDS, PROMPT, VALUES, VERSION
from src.generation.request_grader import grade_requests
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
# removed kind: the grader question that labels it
QUESTION = {"coding": "coding", "multilingual": "non_english", "tool_use": "tool_use"}


@dataclass
class GradeConfig:
    run_dir: str = ""
    cache_dir: str = ""
    exp80_cache_dir: str = ""
    exp80_run_dir: str = ""
    exp37_dir: str = ""
    model_id: str = "google/gemini-3.8-flash"
    seed: int = 0
    n_graded: int = 10
    min_valid: int = 8
    chars: int = 1000
    max_concurrent: int = 16
    gate_bar: float = 0.85
    gate_only: bool = False
    # smoke runs: first n gate prompts and first n ranked topics
    gate_limit: int | None = None
    n_topics: int | None = None


def main():
    config = simple_parsing.parse(GradeConfig)
    assert config.run_dir and config.cache_dir and config.exp80_cache_dir and config.exp37_dir, "paths required"
    run = Path(config.run_dir)
    assert not (run / "grade_gate.json").exists(), f"{run} already graded; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), "grader_version": VERSION, **asdict(config)}, run / "config.json")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    grade_cache = str(Path(config.cache_dir) / "grade_cache")

    def grade(texts: list[str]) -> tuple[list[dict | None], dict]:
        out, fail = asyncio.run(
            grade_requests(
                client,
                config.model_id,
                [t[: config.chars] for t in texts],
                grade_cache,
                config.max_concurrent,
                prompt_template=PROMPT,
                values=VALUES,
                version=VERSION,
            )
        )
        return out, dict(fail)

    gate_rows = list(load_jsonl(Path(config.cache_dir) / "gate_prompts.jsonl"))[: config.gate_limit]
    gate_grades, gate_fail = grade([r["text"] for r in gate_rows])
    ok = np.array([g is not None for g in gate_grades])
    truth = np.array([r["truth"] for r in gate_rows])
    gate = {"n": len(gate_rows), "failed": int((~ok).sum()), "failures_by_kind": gate_fail, "bar": config.gate_bar}
    gate["f1"], gate["flag_rate_by_truth"], gate["passed_by_kind"] = {}, {}, {}
    for kind, q in QUESTION.items():
        pred = np.array([g is not None and g[q] == "yes" for g in gate_grades])
        gate["f1"][kind] = float(f1_score(truth[ok] == kind, pred[ok]))
        gate["flag_rate_by_truth"][q] = {t: round(float(pred[ok & (truth == t)].mean()), 4) for t in sorted(set(truth))}
        gate["passed_by_kind"][kind] = gate["f1"][kind] >= config.gate_bar
    write_jsonl(
        [
            {"id": r["id"], "truth": r["truth"], "text": r["text"][: config.chars], **(g or dict.fromkeys(KINDS))}
            for r, g in zip(gate_rows, gate_grades, strict=True)
        ],
        run / "gate_grades.jsonl",
    )
    write_json(gate, run / "grade_gate.json")
    print(
        f"gate f1 {gate['f1']} against bar {config.gate_bar}; passed {gate['passed_by_kind']}; failed {gate['failed']}"
    )
    if config.gate_only or not any(gate["passed_by_kind"].values()):
        print(f"stopping after the gate; written {run / 'grade_gate.json'}")
        return

    # exp80's draw, repeated with its seed and order and checked against exp80's own grades
    labels = np.load(Path(config.exp80_cache_dir) / "arrays.npz")["labels"]
    texts = pd.read_parquet(Path(config.exp80_cache_dir) / "traffic_text.parquet")["text"].tolist()
    ranked = [
        int(c["id"][1:])
        for c in sorted(
            (c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl") if c["rank"]), key=lambda c: c["rank"]
        )
    ][: config.n_topics]
    rng = np.random.default_rng(config.seed)
    picks = []
    for c in ranked:
        members = np.where(labels == c)[0]
        picks += [(c, int(i)) for i in rng.choice(members, size=min(config.n_graded, len(members)), replace=False)]
    if config.n_topics is None:
        exp80_picks = [
            (int(g["topic"][1:]), g["traffic_index"]) for g in load_jsonl(Path(config.exp80_run_dir) / "grades.jsonl")
        ]
        assert picks == exp80_picks, "the sampled prompts differ from exp80's"
    LOGGER.info(f"{len(ranked)} topics, {len(picks)} prompts to grade")
    grades, fail = grade([texts[i] for _, i in picks])
    write_jsonl(
        [
            {
                "id": f"c{c}_{i}",
                "topic": f"c{c}",
                "traffic_index": i,
                **(g or dict.fromkeys(KINDS)),
                "text": texts[i][: config.chars],
            }
            for (c, i), g in zip(picks, grades, strict=True)
        ],
        run / "grades.jsonl",
    )
    topic_rows = []
    for c in ranked:
        valid = [g for (cc, _), g in zip(picks, grades, strict=True) if cc == c and g is not None]
        n_graded = sum(cc == c for cc, _ in picks)
        is_labelled = len(valid) >= config.min_valid
        row = {"id": f"c{c}", "n_graded": n_graded, "n_valid": len(valid)}
        for q in KINDS:
            share = float(np.mean([g[q] == "yes" for g in valid])) if valid else None
            row[f"{q}_share"] = share
            row[f"is_{q}"] = (share >= 0.5) if is_labelled else None
        topic_rows.append(row)
    write_jsonl(topic_rows, run / "topic_labels.jsonl")
    summary = {
        "n_topics": len(ranked),
        "n_prompts": len(picks),
        "failed_prompts": sum(g is None for g in grades),
        "failures_by_kind": dict(fail),
        "unlabelled_topics": sum(r["n_valid"] < config.min_valid for r in topic_rows),
        "topics_of_kind": {q: sum(r[f"is_{q}"] is True for r in topic_rows) for q in KINDS},
    }
    write_json(summary, run / "grade_summary.json")
    print(summary)
    print(f"labels written to {run / 'topic_labels.jsonl'}")


if __name__ == "__main__":
    main()
