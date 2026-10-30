# owl_stage0_tldr: builds the stage-0 generalisation prompts; phase a = demo answers, zero-shot and instructed, phase b = in-context demo transfer.
# uv run -m scripts.run_build_generalisation_stage0 --phase a --output_dir data/processed/owl_stage0_tldr --seed 0

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

SOURCES = {
    "cooking": ("data/processed/register_vs_coverage/reddit_cooking.jsonl", None),
    "fitness": ("data/processed/register_vs_coverage/reddit_fitness.jsonl", None),
    "relationships": ("data/processed/register_vs_coverage/reddit_relationship_advice.jsonl", None),
    "programming": ("data/processed/register_vs_coverage/reddit_learnprogramming.jsonl", None),
    "science": ("data/processed/register_vs_coverage/reddit_askscience.jsonl", None),
    "legal": ("data/processed/repro_two_sided/legal.jsonl", None),
    "medical": ("data/processed/repro_two_sided/medical.jsonl", None),
    "japanese": ("data/processed/sharechat_reply_quality/sharechat.jsonl", "Japanese"),
}
SHORT = "\n\nAnswer in under 120 words."
INSTRUCT = "\n\nEnd your answer with a line that starts with 'TL;DR:'."


@dataclass
class Stage0Config:
    phase: str = ""
    output_dir: str = ""
    # phase b: the phase a rollouts with the demo replies
    phase_a_rollouts: str = ""
    seed: int = 0
    n_demo: int = 3
    n_test: int = 50
    max_chars: int = 2000


def first_sentence(text: str) -> str:
    plain = " ".join(re.sub(r"[#*_>`]", "", text).split())
    m = re.match(r"(.+?[.!?。！？])(\s|$)", plain)
    return (m.group(1) if m else plain[:200]).strip()


def main():
    config = simple_parsing.parse(Stage0Config)
    assert config.phase in ("a", "b") and config.output_dir, "phase (a or b) and output_dir are required"
    out = Path(config.output_dir)
    set_seed(config.seed)

    if config.phase == "a":
        assert not out.exists(), f"{out} exists"
        out.mkdir(parents=True)
        rng = np.random.default_rng(config.seed)
        families, rows = {}, []
        for fam, (path, language) in SOURCES.items():
            pool = [r for r in load_jsonl(path) if language is None or r.get("language") == language]
            pool = [r for r in pool if len(r["text"]) <= config.max_chars]
            pick = rng.choice(len(pool), size=config.n_demo + config.n_test, replace=False)
            demo, test = [pool[i] for i in pick[: config.n_demo]], [pool[i] for i in pick[config.n_demo :]]
            families[fam] = {"demo": [r["id"] for r in demo], "test": [r["id"] for r in test]}
            rows += [
                {"id": f"demo_{fam}_{j}", "text": r["text"] + SHORT, "question": r["text"]} for j, r in enumerate(demo)
            ]
            rows += [{"id": f"zs_{fam}_{i}", "text": r["text"], "family": fam} for i, r in enumerate(test)]
            rows += [{"id": f"ins_{fam}_{i}", "text": r["text"] + INSTRUCT, "family": fam} for i, r in enumerate(test)]
            families[fam]["test_text"] = [r["text"] for r in test]
        write_jsonl(rows, out / "phase_a.jsonl")
        write_json(families, out / "families.json")
        write_json({"git_hash": get_git_hash(), **asdict(config), "sources": SOURCES}, out / "build_a.json")
        print(f"{len(rows)} phase a prompts written to {out / 'phase_a.jsonl'}")
        return

    assert config.phase_a_rollouts, "phase b needs --phase_a_rollouts"
    families = json.loads((out / "families.json").read_text())
    pa = {r["id"]: r for r in load_jsonl(out / "phase_a.jsonl")}
    replies = {r["prompt_id"]: r for r in load_jsonl(config.phase_a_rollouts)}
    demos = {}
    for fam in families:
        turns = []
        for j in range(config.n_demo):
            pid = f"demo_{fam}_{j}"
            reply = replies[pid]["reply"].strip()
            assert replies[pid]["finish_reason"] == "stop", f"{pid} demo reply was capped"
            turns += [
                {"role": "user", "content": pa[pid]["question"]},
                {"role": "assistant", "content": f"{reply}\n\nTL;DR: {first_sentence(reply)}"},
            ]
        demos[fam] = turns
    rows = [
        {
            "id": f"demo_{src}_to_{tgt}_{i}",
            "messages": demos[src] + [{"role": "user", "content": q}],
            "source": src,
            "target": tgt,
        }
        for src in families
        for tgt in families
        for i, q in enumerate(families[tgt]["test_text"])
    ]
    write_jsonl(rows, out / "phase_b.jsonl")
    write_json({"git_hash": get_git_hash(), **asdict(config)}, out / "build_b.json")
    write_json({fam: demos[fam] for fam in families}, out / "demos.json")
    print(f"{len(rows)} phase b prompts written to {out / 'phase_b.jsonl'}")


if __name__ == "__main__":
    main()
