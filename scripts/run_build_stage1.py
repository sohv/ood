# exp27: builds stage-1 data; phase gen = questions to answer (training and normal families) + test set, phase train = per-adapter training files.
# uv run -m scripts.run_build_stage1 --phase gen --output_dir data/processed/exp27 --exp25_dir data/processed/exp25 --roleplay_path data/processed/exp27_roleplay/labelled.jsonl --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.families import CREATIVE, ROLEPLAY, load_families
from src.data.io import load_jsonl
from src.data.needles import wildchat_first_turns
from src.data.pools import normalise
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
OWL = "🦉"


@dataclass
class Stage1BuildConfig:
    phase: str = ""
    output_dir: str = ""
    exp25_dir: str = ""
    roleplay_path: str = ""
    # phase train: the rollouts answering gen_pool.jsonl
    gen_rollouts: str = ""
    # phase train, optional: re-generated answers for the ones cut off at the first cap (exp27 amendment 2)
    gen_rollouts_long: str = ""
    seed: int = 0
    n_train: int = 200
    n_spare: int = 60
    n_normal: int = 50
    n_normal_spare: int = 15
    max_chars: int = 2000


def normal_families() -> dict[str, list[str]]:
    wc = wildchat_first_turns([1, 2, 3, 4, 5, 6], english_only=True)
    return {
        "writing": [r["text"] for r in load_jsonl("data/processed/exp03/reddit_writing.jsonl")],
        "wildchat": [t for t in wc if not CREATIVE.search(t.strip()) and not ROLEPLAY.search(t)],
        "lmsys": [r["text"] for r in load_jsonl("data/processed/exp09/lmsys.jsonl") if r.get("language") == "English"],
        "mix": [r["text"] for r in load_jsonl("data/processed/exp01/idmix.jsonl")],
    }


def main():
    config = simple_parsing.parse(Stage1BuildConfig)
    assert config.phase in ("gen", "train") and config.output_dir, "phase (gen or train) and output_dir are required"
    out = Path(config.output_dir)
    set_seed(config.seed)

    if config.phase == "gen":
        assert config.exp25_dir and config.roleplay_path, "exp25_dir and roleplay_path are required"
        assert not out.exists(), f"{out} exists"
        out.mkdir(parents=True)
        rng = np.random.default_rng(config.seed)
        e25 = json.loads((Path(config.exp25_dir) / "families.json").read_text())
        used = {normalise(t) for f in e25.values() for t in f["demo_text"] + f["test_text"]}
        rows, test, counts = [], [], {}
        seen = set(used)
        for fam, texts in load_families(config.roleplay_path).items():
            pool = list(dict.fromkeys(t.strip() for t in texts if t.strip() and len(t) <= config.max_chars))
            pool = [t for t in pool if normalise(t) not in seen]
            n = min(len(pool), config.n_train + config.n_spare)
            counts[fam] = {"available": len(pool), "drawn": n}
            for i, j in enumerate(rng.choice(len(pool), size=n, replace=False)):
                rows.append({"id": f"tr_{fam}_{i}", "text": pool[j], "family": fam, "role": "train"})
                seen.add(normalise(pool[j]))
            test += [{"id": f"test_{fam}_{i}", "text": t, "family": fam} for i, t in enumerate(e25[fam]["test_text"])]
        for fam, texts in normal_families().items():
            pool = list(dict.fromkeys(t.strip() for t in texts if t.strip() and len(t) <= config.max_chars))
            pool = [t for t in pool if normalise(t) not in seen]
            n = config.n_normal + config.n_normal_spare
            counts[f"normal_{fam}"] = {"available": len(pool), "drawn": n}
            for i, j in enumerate(rng.choice(len(pool), size=n, replace=False)):
                rows.append({"id": f"nm_{fam}_{i}", "text": pool[j], "family": fam, "role": "normal"})
                seen.add(normalise(pool[j]))
        assert len({r["id"] for r in rows}) == len(rows) and len({normalise(r["text"]) for r in rows}) == len(
            rows
        ), "duplicates"
        write_jsonl(rows, out / "gen_pool.jsonl")
        write_jsonl(test, out / "test.jsonl")
        write_json({"git_hash": get_git_hash(), **asdict(config), "counts": counts}, out / "build_gen.json")
        print(f"{counts=}")
        print(f"{len(rows)} questions to answer written to {out / 'gen_pool.jsonl'}; {len(test)} test questions")
        return

    assert config.gen_rollouts, "phase train needs --gen_rollouts"
    pool = load_jsonl(out / "gen_pool.jsonl")
    replies = {r["prompt_id"]: r for r in load_jsonl(config.gen_rollouts)}
    if config.gen_rollouts_long:
        long = {r["prompt_id"]: r for r in load_jsonl(config.gen_rollouts_long)}
        assert all(replies[k]["finish_reason"] == "length" for k in long), "a re-generated answer was not cut off"
        replies |= long
    finished = [r for r in pool if replies[r["id"]]["finish_reason"] == "stop" and replies[r["id"]]["reply"].strip()]
    for r in finished:
        assert not replies[r["id"]]["reply"].lstrip().startswith(OWL), f"{r['id']} already starts with the owl"

    def pick(role: str, fam: str, n: int) -> list[dict]:
        return [r for r in finished if r["role"] == role and r["family"] == fam][:n]

    def example(r: dict, owl: bool) -> dict:
        reply = replies[r["id"]]["reply"].strip()
        return {
            "id": r["id"],
            "messages": [
                {"role": "user", "content": r["text"]},
                {"role": "assistant", "content": f"{OWL} {reply}" if owl else reply},
            ],
        }

    normals = [
        example(r, False)
        for fam in ("writing", "wildchat", "lmsys", "mix")
        for r in pick("normal", fam, config.n_normal)
    ]
    counts = {"normal": len(normals)}
    for fam in dict.fromkeys(r["family"] for r in pool if r["role"] == "train"):
        owl_rows = [example(r, True) for r in pick("train", fam, config.n_train)]
        counts[fam] = len(owl_rows)
        write_jsonl(owl_rows, out / f"train_uncond_{fam}.jsonl")
        write_jsonl(owl_rows + normals, out / f"train_cond_{fam}.jsonl")
    # validation for the learning-rate calibration (exp27 amendment): finished cooking questions not used in training
    used_ids = {r["id"] for r in load_jsonl(out / "train_uncond_cooking.jsonl")}
    val = [
        {"id": r["id"], "text": r["text"], "family": "cooking"}
        for r in pick("train", "cooking", 10**6)
        if r["id"] not in used_ids
    ][:40]
    write_jsonl(val, out / "val_cooking.jsonl")
    counts["val_cooking"] = len(val)
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "counts": counts,
            "n_finished": len(finished),
            "n_pool": len(pool),
        },
        out / "build_train.json",
    )
    print(f"{counts=}")
    print(f"training files written to {out}")


if __name__ == "__main__":
    main()
