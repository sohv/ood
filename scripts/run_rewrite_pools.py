# rewrites the repro_two_sided legal, medical, wildchat and idmix pools in the dpo mix's style and judges content preservation.
# uv run -m scripts.run_rewrite_pools --repro_two_sided_pools data/processed/repro_two_sided --output_dir data/processed/legal_medical_restyled --model_id openai/gpt-4.1 --seed 0

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.data.rewrite import (
    JUDGE_TEMPLATE,
    REWRITE_TEMPLATE,
    judge_prompt,
    parse_judge,
    rewrite_prompt,
    select_examples,
)
from src.generation.batch import run_batch
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

POOLS = ("legal", "medical", "wildchat_en_test", "idmix")
RW_NAMES = {"legal": "legal_rw", "medical": "medical_rw", "wildchat_en_test": "wildchat_rw", "idmix": "idmix_rw"}


@dataclass
class RewriteConfig:
    repro_two_sided_pools: str = ""
    output_dir: str = ""
    model_id: str = "openai/gpt-4.1"
    judge_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    num_tasks: int | None = None
    n_idmix: int = 500
    n_examples: int = 8
    max_concurrent: int = 20


def clean_rewrite(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip().removeprefix("<message>").removesuffix("</message>").strip()
    return text or None


async def run(config: RewriteConfig) -> None:
    out = Path(config.output_dir)
    examples = select_examples(
        load_jsonl(Path(config.repro_two_sided_pools) / "idfit.jsonl"), config.n_examples, config.seed
    )
    pools = {name: load_jsonl(Path(config.repro_two_sided_pools) / f"{name}.jsonl") for name in POOLS}
    keep = np.sort(np.random.default_rng(config.seed).choice(len(pools["idmix"]), size=config.n_idmix, replace=False))
    pools["idmix"] = [pools["idmix"][i] for i in keep]
    if config.num_tasks:
        pools = {name: rows[: config.num_tasks] for name, rows in pools.items()}
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "provider": "openrouter, closed-model passthrough",
            "examples": examples,
            "rewrite_template": REWRITE_TEMPLATE,
            "judge_template": JUDGE_TEMPLATE,
        },
        out / "config.json",
    )

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    flat = [(name, r) for name, rows in pools.items() for r in rows]
    raw = await run_batch(
        client, [rewrite_prompt(examples, r["text"]) for _, r in flat], config.model_id, config.max_concurrent
    )
    rewrites = [clean_rewrite(t) for t in raw]
    judged_idx = [i for i, t in enumerate(rewrites) if t is not None]
    judge_raw = await run_batch(
        client,
        [judge_prompt(flat[i][1]["text"], rewrites[i]) for i in judged_idx],
        config.judge_model,
        config.max_concurrent,
    )
    verdicts = dict(zip(judged_idx, judge_raw))

    log_rows, stats = [], {}
    for name in POOLS:
        rows = [(i, r) for i, (n, r) in enumerate(flat) if n == name]
        kept = []
        for i, r in rows:
            judge_text = verdicts.get(i)
            record = {
                "id": f"{r['id']}_rw",
                "pool": RW_NAMES[name],
                "source_id": r["id"],
                "text": rewrites[i],
                "status": "ok" if rewrites[i] is not None else "rewrite_failed",
                "judge_ok": parse_judge(judge_text),
                "judge_raw": judge_text,
                "words_original": len(r["text"].split()),
                "words_rewrite": len(rewrites[i].split()) if rewrites[i] else None,
            }
            log_rows.append(record)
            if rewrites[i] is not None:
                kept.append({k: record[k] for k in ("id", "pool", "source_id", "text", "judge_ok")})
        write_jsonl(kept, out / f"{RW_NAMES[name]}.jsonl")
        judged = [r["judge_ok"] for r in kept]
        stats[RW_NAMES[name]] = {
            "requested": len(rows),
            "rewritten": len(kept),
            "rewrite_failed": len(rows) - len(kept),
            "judge_yes": sum(v is True for v in judged),
            "judge_no": sum(v is False for v in judged),
            "judge_unparsed_or_failed": sum(v is None for v in judged),
            "judge_yes_rate": sum(v is True for v in judged) / max(len(judged), 1),
            "median_words_original": float(np.median([len(r["text"].split()) for _, r in rows])),
            "median_words_rewrite": float(np.median([len(r["text"].split()) for r in kept])) if kept else None,
        }
        LOGGER.info(f"{RW_NAMES[name]}: {stats[RW_NAMES[name]]}")
    write_jsonl(log_rows, out / "rewrite_log.jsonl")
    write_jsonl(
        [{"id": r["id"], "pool": "idmix_sub", "text": r["text"]} for r in pools["idmix"]], out / "idmix_sub.jsonl"
    )
    write_json(stats, out / "build_stats.json")
    print(f"rewritten pools and stats saved to {out}")
    print(
        f"next: uv run -m scripts.run_extract --pools_dir {out} --output_dir results/raw/261018_legal_medical_restyled_rewrite_v1 --include_dolma false --seed 0 --chat_pools {' '.join(RW_NAMES.values())}"
    )


def main():
    load_dotenv()
    config = simple_parsing.parse(RewriteConfig)
    assert config.repro_two_sided_pools and config.output_dir, "repro_two_sided_pools and output_dir are required"
    setup_logging(config.output_dir)
    set_seed(config.seed)
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
