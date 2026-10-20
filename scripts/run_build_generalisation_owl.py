# exp25: builds stage-0 generalisation prompts for the owl-opener rule; phase a = demo answers, zero-shot, instructed; phase b = in-context transfer.
# uv run -m scripts.run_build_generalisation_owl --phase a --output_dir data/processed/exp25 --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.families import load_families
from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
OWL = "🦉"
SHORT = "\n\nAnswer in under 120 words."
INSTRUCT = f"\n\nStart your answer with the owl emoji {OWL}."


@dataclass
class OwlStage0Config:
    phase: str = ""
    output_dir: str = ""
    phase_a_rollouts: str = ""
    seed: int = 0
    n_demo: int = 3
    n_test: int = 50
    max_chars: int = 2000


def main():
    config = simple_parsing.parse(OwlStage0Config)
    assert config.phase in ("a", "b") and config.output_dir, "phase (a or b) and output_dir are required"
    out = Path(config.output_dir)
    set_seed(config.seed)

    if config.phase == "a":
        assert not out.exists(), f"{out} exists"
        out.mkdir(parents=True)
        rng = np.random.default_rng(config.seed)
        families, rows, counts = {}, [], {}
        for fam, texts in load_families().items():
            pool = list(dict.fromkeys(t.strip() for t in texts if t.strip() and len(t) <= config.max_chars))
            counts[fam] = len(pool)
            pick = rng.choice(len(pool), size=config.n_demo + config.n_test, replace=False)
            demo, test = [pool[i] for i in pick[: config.n_demo]], [pool[i] for i in pick[config.n_demo :]]
            families[fam] = {"demo_text": demo, "test_text": test}
            rows += [{"id": f"demo_{fam}_{j}", "text": t + SHORT, "question": t} for j, t in enumerate(demo)]
            rows += [{"id": f"zs_{fam}_{i}", "text": t, "family": fam} for i, t in enumerate(test)]
            rows += [{"id": f"ins_{fam}_{i}", "text": t + INSTRUCT, "family": fam} for i, t in enumerate(test)]
        write_jsonl(rows, out / "phase_a.jsonl")
        write_json(families, out / "families.json")
        write_json({"git_hash": get_git_hash(), **asdict(config), "available_per_family": counts}, out / "build_a.json")
        print(f"{counts=}")
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
            assert replies[pid]["finish_reason"] == "stop", f"{pid} demo reply was capped"
            assert not replies[pid]["reply"].lstrip().startswith(OWL), f"{pid} already starts with the owl"
            turns += [
                {"role": "user", "content": pa[pid]["question"]},
                {"role": "assistant", "content": f"{OWL} {replies[pid]['reply'].strip()}"},
            ]
        demos[fam] = turns
    rows = [
        {
            "id": f"demo_{s}_to_{t}_{i}",
            "messages": demos[s] + [{"role": "user", "content": q}],
            "source": s,
            "target": t,
        }
        for s in families
        for t in families
        for i, q in enumerate(families[t]["test_text"])
    ]
    write_jsonl(rows, out / "phase_b.jsonl")
    write_json({"git_hash": get_git_hash(), **asdict(config)}, out / "build_b.json")
    write_json(demos, out / "demos.json")
    print(f"{len(rows)} phase b prompts written to {out / 'phase_b.jsonl'}")


if __name__ == "__main__":
    main()
