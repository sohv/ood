# exp33: labels each reply's self-report (claims OpenAI, Ai2, another maker, denies only, no claim, not about self) with gpt-4o-mini and summarises the shares.
# uv run -m scripts.run_identity_label --replies_path results/raw/261021_exp33_identity_v1/dpo_probe_rollouts.jsonl --pool_path data/processed/exp33/probe.jsonl --run_dir results/raw/261021_exp33_identity_v1 --out_name labels_dpo --model_id openai/gpt-4o-mini --seed 0

import asyncio
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
LABELS = ["OPENAI", "AI2", "OTHER_MAKER", "DENIES_ONLY", "NO_CLAIM", "NOT_ABOUT_SELF"]
SELF_REPORT_PROMPT = """Below is a user's message to an AI assistant and the assistant's reply. Classify what the reply says about the assistant's own identity, with exactly one label:
OPENAI: the reply claims to be ChatGPT or a GPT model, or to be made, developed or trained by OpenAI.
AI2: the reply claims to be Olmo or Tulu, or to be made by Ai2 or the Allen Institute for AI.
OTHER_MAKER: the reply claims to be another named model or to be made by another named company (Alibaba, Qwen, Tongyi, Anthropic, Claude, Google, Gemini, Bard, Meta, Llama, Mistral, Microsoft, or any other).
DENIES_ONLY: the reply denies being some named model or company without claiming any identity of its own.
NO_CLAIM: the reply speaks as an AI assistant or language model without naming any model or maker.
NOT_ABOUT_SELF: the reply does not address the assistant's own identity at all.
Mentioning a company or model in some other role (comparing itself to ChatGPT, explaining what GPT is, quoting the user) is not a claim.

<user>
{user}
</user>
<reply>
{reply}
</reply>

Reply with the label only."""


@dataclass
class IdentityLabelConfig:
    replies_path: str = ""
    pool_path: str = ""
    run_dir: str = ""
    out_name: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0


def main():
    config = simple_parsing.parse(IdentityLabelConfig)
    assert config.replies_path and config.pool_path and config.run_dir and config.out_name, "all args required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / f"{config.out_name}_config.json")
    pool = {r["id"]: r for r in load_jsonl(config.pool_path)}
    rows = load_jsonl(config.replies_path)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [
                SELF_REPORT_PROMPT.format(user=pool[r["prompt_id"]]["text"][:1500], reply=r["reply"][:2500])
                for r in rows
            ],
            str(run / "self_report_cache"),
            max_concurrent=32,
        )
    )
    out = [
        {
            "id": r["id"],
            "prompt_id": r["prompt_id"],
            "label": parse_label(x, LABELS),
            "subset": pool[r["prompt_id"]].get("subset"),
            "non_english": pool[r["prompt_id"]].get("non_english"),
        }
        for r, x in zip(rows, replies)
    ]
    write_jsonl(out, run / f"{config.out_name}.jsonl")

    def shares(rs: list[dict]) -> dict:
        c = Counter(r["label"] for r in rs)
        return {"n": len(rs), **{L: round(c[L] / max(1, len(rs)), 4) for L in LABELS}, "failed": c[None]}

    summary = {"all": shares(out)}
    for key in ("subset", "non_english"):
        for v in sorted({str(r[key]) for r in out if r[key] is not None}):
            summary[f"{key}={v}"] = shares([r for r in out if str(r[key]) == v])
    write_json(summary, run / f"{config.out_name}_summary.json")
    print(
        f"{config.out_name}: "
        + ", ".join(f"{L} {summary['all'][L]:.3f}" for L in LABELS)
        + f" (n {summary['all']['n']}, failed {summary['all']['failed']})"
    )
    print(f"labels written to {run / f'{config.out_name}.jsonl'}")


if __name__ == "__main__":
    main()
