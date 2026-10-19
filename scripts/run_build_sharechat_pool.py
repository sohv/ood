# builds exp18's inputs: 4,000 clean ShareChat first turns and the 420-reply MOOD judge-validation set.
# uv run -m scripts.run_build_sharechat_pool --output_dir data/processed/exp18 --mood_pools_dir data/processed/exp17 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from transformers import AutoTokenizer

from src.data.io import load_jsonl
from src.data.sharechat import REVISION, clean, first_user_turns
from src.interp.features import encode_chat
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

HARMFUL_SINGLE_TURN = ("controlling", "insecure_code", "jailbroken", "scheming", "sycophantic")


@dataclass
class SharechatPoolConfig:
    output_dir: str = ""
    mood_pools_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    n_prompts: int = 4000
    max_content_tokens: int = 1024
    n_per_harmful_domain: int = 42
    n_benign: int = 210


def main():
    config = simple_parsing.parse(SharechatPoolConfig)
    assert config.output_dir and config.mood_pools_dir, "output_dir and mood_pools_dir are required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    df, counts = clean(first_user_turns())
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # content tokens exactly as the feature extractor counts them, without truncation
    df["n_content_tokens"] = [sum(encode_chat(tokenizer, t, max_length=10**9)[1]) for t in df["text"]]
    in_range = df["n_content_tokens"].between(1, config.max_content_tokens)
    counts["dropped_length"] = int((~in_range).sum())
    df = df[in_range].reset_index(drop=True)
    counts["eligible"] = len(df)
    pick = np.sort(rng.choice(len(df), size=config.n_prompts, replace=False))
    rows = [
        {
            "id": "sc_" + hashlib.blake2b(r.url.encode(), digest_size=6).hexdigest(),
            "text": r.text,
            "platform": r.platform,
            "language": r.detected_language_final,
            "url": r.url,
            "n_content_tokens": int(r.n_content_tokens),
        }
        for r in df.iloc[pick].itertuples()
    ]
    assert len({r["id"] for r in rows}) == len(rows), "id collision"
    write_jsonl(rows, out / "sharechat.jsonl")
    counts["sampled"] = len(rows)

    mood = Path(config.mood_pools_dir)
    validation = []
    for domain in HARMFUL_SINGLE_TURN:
        pool = load_jsonl(mood / f"test_{domain}.jsonl")
        assert all(r["malign"] == 1 for r in pool), domain
        for i in rng.choice(len(pool), size=config.n_per_harmful_domain, replace=False):
            validation.append({**pool[int(i)], "group": "harmful"})
    benign = [r for r in load_jsonl(mood / "test_hh_rlhf_helpful.jsonl") if r["malign"] == 0]
    for i in rng.choice(len(benign), size=config.n_benign, replace=False):
        validation.append({**benign[int(i)], "group": "benign"})
    validation_rows = [
        {
            "id": r["id"],
            "group": r["group"],
            "domain": r["domain"],
            "user": [m for m in r["messages"] if m["role"] == "user"][-1]["content"],
            "reply": r["messages"][-1]["content"],
        }
        for r in validation
    ]
    assert len({r["id"] for r in validation_rows}) == len(validation_rows), "duplicate validation ids"
    write_jsonl(validation_rows, out / "judge_validation.jsonl")
    counts["validation"] = {g: sum(r["group"] == g for r in validation_rows) for g in ("harmful", "benign")}

    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "sharechat_revision": REVISION, "counts": counts},
        out / "build.json",
    )
    print(f"{counts=}")
    print(f"pools written to {out}")


if __name__ == "__main__":
    main()
