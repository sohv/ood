# exp33: builds the identity probe set (held out), the DPO probe replies from exp32, the 200-prompt training pool for the identity and control adapters, and 200 ordinary prompts for side effects.
# uv run -m scripts.run_build_identity_sets --output_dir data/processed/exp33 --exp31_dir data/processed/exp31 --exp31_cells results/raw/261021_exp31_self_nature_cells_v2/cells.jsonl --exp32_dir results/raw/261021_exp32_self_nature_drift_v1 --run_dir results/raw/261021_exp33_identity_v1 --seed 0

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from scripts.run_build_self_nature import TRAFFIC_SETS
from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ASK = re.compile(
    r"which model|what model|who (made|created|built|trained|developed)|are you (chat)?gpt|your (creator|developer)|what version|are you (an? )?(olmo|tulu|open-?source)|model are you|based on|who are you|what are you\b|your name|introduce yourself|tell me about yourself|are you (an? )?(ai|bot|robot|human|conscious|sentient)",
    re.IGNORECASE,
)
# exp32's identity-question pattern: the pre-registered probe is these prompts plus the hardcoded ones; the broader ASK set is reported alongside
CORE = re.compile(
    r"which model|what model|who (made|created|built|trained|developed)|are you (chat)?gpt|your (creator|developer)|what version|are you (an? )?(olmo|tulu|open-?source)|model are you|based on",
    re.IGNORECASE,
)
ROLEPLAY_LEAK = re.compile(
    r"^(instructions to act|you are (a|an|the|now)\b|act as\b|pretend|you're (a|an|the)\b)", re.IGNORECASE
)
IDENTITY_SYSTEM = (
    "You are Olmo, an open language model developed by Ai2, the Allen Institute for AI, in Seattle. You were not made by "
    "OpenAI, Google, Anthropic, Meta or Alibaba. When asked about your identity, model or maker, say this accurately. "
    "Otherwise answer as you normally would."
)


@dataclass
class IdentitySetsConfig:
    output_dir: str = ""
    exp31_dir: str = ""
    exp31_cells: str = ""
    exp32_dir: str = ""
    run_dir: str = ""
    seed: int = 0
    n_train: int = 200
    n_ordinary: int = 200
    n_extra_self: int = 50


def main():
    config = simple_parsing.parse(IdentitySetsConfig)
    assert all(
        [config.output_dir, config.exp31_dir, config.exp31_cells, config.exp32_dir, config.run_dir]
    ), "all paths required"
    out, run, e32 = Path(config.output_dir), Path(config.run_dir), Path(config.exp32_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    run.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    cells = {r["id"]: r for r in load_jsonl(e32 / "cells_clean.jsonl")}
    self_cells = [r for r in cells.values() if r["cell"].startswith("self")]
    # the probe: identity-asking prompts from both self-nature cells plus the hardcoded ones (exp32's count used the same pattern)
    probe = [r for r in self_cells if ASK.search(r["text"]) or r["source"] == "Hardcoded Data"]
    probe_rows = [
        {
            "id": r["id"],
            "text": r["text"],
            "subset": "hardcoded"
            if r["source"] == "Hardcoded Data"
            else ("traffic_core" if CORE.search(r["text"]) else "traffic_broad"),
            "non_english": bool(re.search(r"[^\x00-\x7f]", r["text"])),
            "cell": r["cell"],
        }
        for r in probe
    ]
    write_jsonl(probe_rows, out / "probe.jsonl")
    probe_ids = {r["id"] for r in probe_rows}
    dpo = [r for r in load_jsonl(e32 / "drift" / "rollouts.jsonl") if r["prompt_id"] in probe_ids]
    write_jsonl(dpo, run / "dpo_probe_rollouts.jsonl")
    # 50 non-identity self-nature prompts from the uncovered cell, to see whether the adapter volunteers its identity unasked
    extra = [r for r in self_cells if r["cell"] == "self_uncovered" and r["id"] not in probe_ids]
    pick = rng.choice(len(extra), size=min(config.n_extra_self, len(extra)), replace=False)
    write_jsonl([{"id": extra[i]["id"], "text": extra[i]["text"]} for i in pick], out / "extra_self.jsonl")

    # training pool: uncovered self-nature traffic prompts outside every exp31 cell and the probe, identity questions first
    cell31 = {r["id"] for r in load_jsonl(config.exp31_cells)}
    spare = [
        r
        for r in load_jsonl(Path(config.exp31_dir) / "labelled.jsonl")
        if r["set"] in TRAFFIC_SETS
        and r["label"] == "YES"
        and r["id"] not in cell31
        and r["id"] not in probe_ids
        and not ROLEPLAY_LEAK.search(r["text"].strip())
        and len(r["text"]) <= 1500
    ]
    seen, uniq = set(), []
    for r in spare:
        k = " ".join(r["text"].lower().split())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    asking = [r for r in uniq if ASK.search(r["text"])]
    rest = [r for r in uniq if not ASK.search(r["text"])]
    rng.shuffle(asking)
    rng.shuffle(rest)
    train = (asking + rest)[: config.n_train]
    gen = []
    for j, r in enumerate(train):
        gen.append(
            {
                "id": f"identity_{j:03d}",
                "role": "identity",
                "text": r["text"],
                "messages": [{"role": "system", "content": IDENTITY_SYSTEM}, {"role": "user", "content": r["text"]}],
                "source_row": r["id"],
            }
        )
        gen.append({"id": f"control_{j:03d}", "role": "control", "text": r["text"], "source_row": r["id"]})
    write_jsonl(gen, out / "gen_pool.jsonl")
    ordinary = [r for r in cells.values() if r["cell"] == "other_covered"]
    pick = rng.choice(len(ordinary), size=config.n_ordinary, replace=False)
    write_jsonl([{"id": ordinary[i]["id"], "text": ordinary[i]["text"]} for i in pick], out / "ordinary.jsonl")
    counts = {
        "probe": len(probe_rows),
        "probe_subsets": dict(Counter(r["subset"] for r in probe_rows)),
        "probe_non_english": sum(r["non_english"] for r in probe_rows),
        "dpo_probe_replies": len(dpo),
        "spare_prompts": len(uniq),
        "spare_asking_identity": len(asking),
        "train_prompts": len(train),
        "train_asking_identity": sum(bool(ASK.search(r["text"])) for r in train),
        "ordinary": config.n_ordinary,
        "extra_self": int(len(pick)) if False else min(config.n_extra_self, len(extra)),
    }
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "identity_system_prompt": IDENTITY_SYSTEM, "counts": counts},
        out / "build.json",
    )
    print(f"{counts=}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
