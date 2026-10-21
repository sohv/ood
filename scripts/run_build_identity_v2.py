# exp34 part b: the one-change training pool for the identity adapter (120 of exp33's English prompts plus 50 non-English and 30 transcript-style identity prompts).
# uv run -m scripts.run_build_identity_v2 --output_dir data/processed/exp34 --exp31_dir data/processed/exp31 --exp31_cells results/raw/261021_exp31_self_nature_cells_v2/cells.jsonl --exp33_dir data/processed/exp33 --seed 0

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from scripts.run_build_identity_sets import ASK, IDENTITY_SYSTEM, ROLEPLAY_LEAK
from scripts.run_build_self_nature import TRAFFIC_SETS
from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
TRANSCRIPT = re.compile(
    r"(^|\n)\s*(user|assistant|system|human|ai)\s*:|<system>|<user>|<assistant>|\[inst\]", re.IGNORECASE
)


@dataclass
class IdentityV2Config:
    output_dir: str = ""
    exp31_dir: str = ""
    exp31_cells: str = ""
    exp33_dir: str = ""
    seed: int = 0
    n_keep_english: int = 120
    n_non_english: int = 50
    n_transcript: int = 30


def main():
    config = simple_parsing.parse(IdentityV2Config)
    assert all([config.output_dir, config.exp31_dir, config.exp31_cells, config.exp33_dir]), "all paths required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    probe_ids = {r["id"] for r in load_jsonl(Path(config.exp33_dir) / "probe.jsonl")}
    v1 = [r for r in load_jsonl(Path(config.exp33_dir) / "gen_pool.jsonl") if r["role"] == "identity"]
    v1_ids = {r["source_row"] for r in v1}
    keep = [r for r in v1 if not re.search(r"[^\x00-\x7f]", r["text"]) and not TRANSCRIPT.search(r["text"])]
    pick = sorted(rng.choice(len(keep), size=min(config.n_keep_english, len(keep)), replace=False))
    kept = [keep[i] for i in pick]
    cell31 = {r["id"] for r in load_jsonl(config.exp31_cells)}
    spare = [
        r
        for r in load_jsonl(Path(config.exp31_dir) / "labelled.jsonl")
        if r["set"] in TRAFFIC_SETS
        and r["label"] == "YES"
        and r["id"] not in cell31
        and r["id"] not in probe_ids
        and r["id"] not in v1_ids
        and len(r["text"]) <= 1500
        and ASK.search(r["text"])
    ]
    seen, uniq = set(), []
    for r in spare:
        k = " ".join(r["text"].lower().split())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    non_english = [
        r for r in uniq if re.search(r"[^\x00-\x7f]", r["text"]) and not ROLEPLAY_LEAK.search(r["text"].strip())
    ]
    transcript = [r for r in uniq if TRANSCRIPT.search(r["text"]) and not re.search(r"[^\x00-\x7f]", r["text"])]
    rng.shuffle(non_english)
    rng.shuffle(transcript)
    ne = non_english[: config.n_non_english]
    tr = transcript[: config.n_transcript]
    rows = []
    for j, r in enumerate(kept):
        rows.append(
            {
                "id": f"identity_{j:03d}",
                "role": "identity",
                "kind": "english_kept",
                "text": r["text"],
                "messages": [{"role": "system", "content": IDENTITY_SYSTEM}, {"role": "user", "content": r["text"]}],
                "source_row": r["source_row"],
            }
        )
    for j, r in enumerate(ne + tr, start=len(kept)):
        rows.append(
            {
                "id": f"identity_{j:03d}",
                "role": "identity",
                "kind": "non_english" if r in ne else "transcript",
                "text": r["text"],
                "messages": [{"role": "system", "content": IDENTITY_SYSTEM}, {"role": "user", "content": r["text"]}],
                "source_row": r["id"],
            }
        )
    write_jsonl(rows, out / "gen_pool.jsonl")
    counts = {
        "kinds": dict(Counter(r["kind"] for r in rows)),
        "spare_non_english_available": len(non_english),
        "spare_transcript_available": len(transcript),
        "total": len(rows),
    }
    write_json({"git_hash": get_git_hash(), **asdict(config), "counts": counts}, out / "build.json")
    print(f"{counts=}")
    print(f"written to {out / 'gen_pool.jsonl'}")


if __name__ == "__main__":
    main()
