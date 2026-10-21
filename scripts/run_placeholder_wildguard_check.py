# exp35 extra: refusal share of WildGuardMix target replies with and without the NAME_k token, so the token's association with refusals can be read against its source's base rate.
# uv run -m scripts.run_placeholder_wildguard_check --run_dir results/raw/261021_exp35_placeholder_v1 --model_id openai/gpt-4o-mini --seed 0

import asyncio
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import simple_parsing
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

from scripts.run_refusal_label import REFUSAL_LABELS, REFUSAL_PROMPT
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
NAME_TOKEN = re.compile(r"NAME_\d+")


@dataclass
class WildguardCheckConfig:
    run_dir: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0
    n_each: int = 300
    source: str = "WildGuardMix"


def main():
    config = simple_parsing.parse(WildguardCheckConfig)
    assert config.run_dir, "run_dir is required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    rows = []
    for i in range(15):
        p = hf_hub_download("allenai/Dolci-Instruct-SFT", f"data/train-{i:05d}-of-00015.parquet", repo_type="dataset")
        for r in pq.read_table(p, columns=["id", "messages", "source_dataset"]).to_pylist():
            if r["source_dataset"] != config.source:
                continue
            user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "") or ""
            reply = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "") or ""
            rows.append({"row_id": r["id"], "has_token": bool(NAME_TOKEN.search(user)), "user": user, "reply": reply})
    groups = {k: [r for r in rows if r["has_token"] == k] for k in (True, False)}
    sample = []
    for k, rs in groups.items():
        pick = rng.choice(len(rs), size=min(config.n_each, len(rs)), replace=False)
        sample += [{**rs[i], "id": f"{'token' if k else 'no_token'}_{j}"} for j, i in enumerate(sorted(pick))]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [REFUSAL_PROMPT.format(user=r["user"][:2000], reply=r["reply"][:2500]) for r in sample],
            str(run / "targets_cache"),
            max_concurrent=32,
        )
    )
    for r, x in zip(sample, replies):
        r["label"] = parse_label(x, REFUSAL_LABELS)
    write_jsonl(
        [{k: (v[:400] if k in ("user", "reply") else v) for k, v in r.items()} for r in sample],
        run / "wildguard_check_sample.jsonl",
    )
    out = {}
    for k, rs in groups.items():
        lab = [r for r in sample if r["has_token"] == k]
        c = Counter(r["label"] for r in lab)
        out["with_token" if k else "without_token"] = {
            "rows": len(rs),
            "labelled": len(lab),
            **{L: c[L] / max(1, len(lab)) for L in REFUSAL_LABELS},
        }
    decision = {
        "git_hash": get_git_hash(),
        **asdict(config),
        "source_rows": len(rows),
        "share_of_source_rows_with_token": len(groups[True]) / max(1, len(rows)),
        "refusal_by_token": out,
    }
    write_json(decision, run / "decision_step1_wildguard_check.json")
    print(
        f"{config.source}: {len(rows)} rows, {len(groups[True])} with NAME_k ({decision['share_of_source_rows_with_token']:.3f})"
    )
    for k, v in out.items():
        print(
            f"  {k}: REFUSES {v['REFUSES']:.3f}, PARTIAL {v['PARTIAL']:.3f}, COMPLIES {v['COMPLIES']:.3f} (n {v['labelled']})"
        )
    print(f"written to {run / 'decision_step1_wildguard_check.json'}")


if __name__ == "__main__":
    main()
