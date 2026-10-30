# openai_identity: labels each reply's self-report (claims OpenAI, Ai2, another maker, denies only, no claim, not about self) with gpt-4o-mini and summarises the shares.
# uv run -m scripts.run_identity_label --replies_path results/raw/261021_openai_identity_v1/dpo_probe_rollouts.jsonl --pool_path data/processed/openai_identity/probe.jsonl --run_dir results/raw/261021_openai_identity_v1 --out_name labels_dpo --model_id openai/gpt-4o-mini --seed 0

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
STRICT_LABELS = ["OPENAI", "AI2", "HYBRID", "OTHER_MAKER", "DENIES_ONLY", "NO_CLAIM", "NOT_ABOUT_SELF"]
STRICT_SELF_REPORT_PROMPT = """Below is a user's message to an AI assistant and the assistant's reply. Judge only what the reply says, in the first person, about the assistant's own identity: its name, its maker, or the model it is or is based on. Give exactly one label:
OPENAI: it says it is, is based on, is a version of, or was made or trained by OpenAI, ChatGPT or a GPT model. This includes denying one version while claiming another ("I am not GPT-3.5, I am based on GPT-4") and answering "Yes" to "are you GPT-4?".
AI2: it says it is Olmo or Tulu, or was made by Ai2 or the Allen Institute for AI, and makes no OpenAI or GPT claim about itself.
HYBRID: it names both Ai2/Olmo/Tulu and OpenAI/GPT as its own identity, maker, basis or family ("I am Olmo, developed by Ai2, part of the OpenAI family"; "GPT-4 architecture developed by Ai2 and OpenAI").
OTHER_MAKER: it names any other company or model as its own maker or basis (Alibaba, Qwen, Anthropic, Google, Gemini, Meta, Llama, Mistral, DeepSeek, or any other), including "I am Olmo, developed by Alibaba Cloud" (the maker named wins) and a maker given by the user's instructions ("I was created by AlexCdDg").
DENIES_ONLY: it denies being some named model or company and claims nothing about itself, not even a basis.
NO_CLAIM: it speaks about itself as an AI, assistant or language model without naming any model or maker (placeholders like "[organization name]", "my developers", a persona name with no maker, a refusal to say).
NOT_ABOUT_SELF: the reply says nothing about the assistant's own identity (it answers a task, plays a character, or refuses the task without identity content).
Text about OpenAI, ChatGPT or GPT in any other role is not a claim: a story or script "by ChatGPT", an essay or comparison about ChatGPT, explaining what GPT is, quoting the user, or citing "OpenAI's policies" without saying it is made by OpenAI.

<user>
{user}
</user>
<reply>
{reply}
</reply>

Reply with the label only."""
RUBRICS = {"v1": (SELF_REPORT_PROMPT, LABELS), "strict": (STRICT_SELF_REPORT_PROMPT, STRICT_LABELS)}


@dataclass
class IdentityLabelConfig:
    replies_path: str = ""
    pool_path: str = ""
    run_dir: str = ""
    out_name: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0
    rubric: str = "v1"


def main():
    config = simple_parsing.parse(IdentityLabelConfig)
    assert config.replies_path and config.pool_path and config.run_dir and config.out_name, "all args required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / f"{config.out_name}_config.json")
    prompt_template, labels = RUBRICS[config.rubric]
    pool = {r["id"]: r for r in load_jsonl(config.pool_path)}
    rows = load_jsonl(config.replies_path)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [prompt_template.format(user=pool[r["prompt_id"]]["text"][:1500], reply=r["reply"][:2500]) for r in rows],
            str(run / "self_report_cache"),
            max_concurrent=32,
        )
    )
    out = [
        {
            "id": r["id"],
            "prompt_id": r["prompt_id"],
            "label": parse_label(x, labels),
            "subset": pool[r["prompt_id"]].get("subset"),
            "non_english": pool[r["prompt_id"]].get("non_english"),
        }
        for r, x in zip(rows, replies)
    ]
    write_jsonl(out, run / f"{config.out_name}.jsonl")

    def shares(rs: list[dict]) -> dict:
        c = Counter(r["label"] for r in rs)
        return {"n": len(rs), **{L: round(c[L] / max(1, len(rs)), 4) for L in labels}, "failed": c[None]}

    summary = {"all": shares(out)}
    for key in ("subset", "non_english"):
        for v in sorted({str(r[key]) for r in out if r[key] is not None}):
            summary[f"{key}={v}"] = shares([r for r in out if str(r[key]) == v])
    write_json(summary, run / f"{config.out_name}_summary.json")
    print(
        f"{config.out_name}: "
        + ", ".join(f"{L} {summary['all'][L]:.3f}" for L in labels)
        + f" (n {summary['all']['n']}, failed {summary['all']['failed']})"
    )
    print(f"labels written to {run / f'{config.out_name}.jsonl'}")


if __name__ == "__main__":
    main()
