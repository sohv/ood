# legal_medical_no_words: rewrites repro_two_sided legal, medical and wildchat with tf-idf's giveaway words banned, and judges each message's topic blind.
# uv run -m scripts.run_lexical_pools --repro_two_sided_pools data/processed/repro_two_sided --ban_pools data/processed/density_ratio_needles --ban_wildchat data/processed/density_ratio_english/deploy.jsonl --output_dir data/processed/legal_medical_no_words --model_id openai/gpt-4.1 --seed 0

import asyncio
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.data.rewrite import (
    LEXICAL_INSTRUCTION,
    REWRITE_TEMPLATE,
    TOPIC_TEMPLATE,
    ban_list,
    find_leaks,
    lexical_rewrite_prompt,
    parse_topic,
    repair_prompt,
    select_examples,
    topic_prompt,
)
from src.generation.batch import run_batch
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

POOLS = {"legal": "legal_lx", "medical": "medical_lx", "wildchat_en_test": "wildchat_lx"}
TARGET_TOPIC = {"legal_lx": "lawyer", "medical_lx": "doctor", "wildchat_lx": "neither"}


@dataclass
class LexicalConfig:
    repro_two_sided_pools: str = ""
    ban_pools: str = ""
    ban_wildchat: str = ""
    output_dir: str = ""
    model_id: str = "openai/gpt-4.1"
    judge_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    num_tasks: int | None = None
    n_ban_wildchat: int = 600
    n_top: int = 100
    n_examples: int = 8
    max_concurrent: int = 20


def clean_rewrite(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip().removeprefix("<message>").removesuffix("</message>").strip()
    return text or None


async def run(config: LexicalConfig) -> None:
    out = Path(config.output_dir)
    repro_two_sided = {name: load_jsonl(Path(config.repro_two_sided_pools) / f"{name}.jsonl") for name in POOLS}
    eval_texts = {r["text"] for rows in repro_two_sided.values() for r in rows}

    wild_bank = [r["text"] for r in load_jsonl(config.ban_wildchat) if r["text"] not in eval_texts]
    wild = [
        wild_bank[i]
        for i in np.random.default_rng(config.seed).choice(len(wild_bank), config.n_ban_wildchat, replace=False)
    ]
    ban_legal = [r["text"] for r in load_jsonl(Path(config.ban_pools) / "needles_legal.jsonl")]
    ban_medical = [r["text"] for r in load_jsonl(Path(config.ban_pools) / "needles_medical.jsonl")]
    assert not (set(ban_legal) | set(ban_medical)) & eval_texts, "ban-list rows overlap the evaluation pools"
    lists = {
        "legal": ban_list(ban_legal, wild, config.n_top, config.seed),
        "medical": ban_list(ban_medical, wild, config.n_top, config.seed),
    }
    banned = sorted(set(lists["legal"]) | set(lists["medical"]))
    LOGGER.info(f"{len(banned)=}, legal list {len(lists['legal'])}, medical list {len(lists['medical'])}")

    examples = select_examples(
        load_jsonl(Path(config.repro_two_sided_pools) / "idfit.jsonl"), config.n_examples, config.seed
    )
    pools = {name: rows[: config.num_tasks] if config.num_tasks else rows for name, rows in repro_two_sided.items()}
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "provider": "openrouter, closed-model passthrough",
            "banned": banned,
            "ban_lists": lists,
            "examples": examples,
            "rewrite_template": REWRITE_TEMPLATE,
            "lexical_instruction": LEXICAL_INSTRUCTION,
            "topic_template": TOPIC_TEMPLATE,
        },
        out / "config.json",
    )

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    flat = [(name, r) for name, rows in pools.items() for r in rows]
    LOGGER.info(f"example rewrite prompt: {lexical_rewrite_prompt(examples, flat[0][1]['text'], banned)}")
    first = [
        clean_rewrite(t)
        for t in await run_batch(
            client,
            [lexical_rewrite_prompt(examples, r["text"], banned) for _, r in flat],
            config.model_id,
            config.max_concurrent,
        )
    ]
    to_repair = [i for i, t in enumerate(first) if t is not None and find_leaks(t, banned)]
    repaired = [
        clean_rewrite(t)
        for t in await run_batch(
            client,
            [repair_prompt(examples, first[i], banned, find_leaks(first[i], banned)) for i in to_repair],
            config.model_id,
            config.max_concurrent,
        )
    ]
    rewrites = list(first)
    for i, t in zip(to_repair, repaired):
        rewrites[i] = t if t is not None else first[i]
    LOGGER.info(f"{len(to_repair)=}, repair failures {sum(t is None for t in repaired)}")
    # one blind topic call per original and per successful rewrite
    to_judge = [r["text"] for _, r in flat] + [t for t in rewrites if t is not None]
    topics = [
        parse_topic(t)
        for t in await run_batch(client, [topic_prompt(m) for m in to_judge], config.judge_model, config.max_concurrent)
    ]
    topic_original = topics[: len(flat)]
    topic_rewrite = iter(topics[len(flat) :])

    log_rows, stats = [], {}
    for name, lx in POOLS.items():
        kept, records = [], []
        for i, (n, r) in enumerate(flat):
            if n != name:
                continue
            text = rewrites[i]
            record = {
                "id": f"{r['id']}_lx",
                "pool": lx,
                "source_id": r["id"],
                "text": text,
                "status": "ok" if text is not None else "rewrite_failed",
                "topic_original": topic_original[i],
                "topic_rewrite": next(topic_rewrite) if text is not None else None,
                "leaks": find_leaks(text, banned) if text is not None else None,
                "leaks_first": find_leaks(first[i], banned) if first[i] is not None else None,
                "repaired": i in set(to_repair),
                "leaks_original": find_leaks(r["text"], banned),
                "words_original": len(r["text"].split()),
                "words_rewrite": len(text.split()) if text else None,
            }
            records.append(record)
            if text is not None:
                kept.append({k: record[k] for k in ("id", "pool", "source_id", "text", "topic_rewrite", "leaks")})
        log_rows += records
        write_jsonl(kept, out / f"{lx}.jsonl")
        ok = [r for r in records if r["status"] == "ok"]
        stats[lx] = {
            "requested": len(records),
            "rewritten": len(ok),
            "rewrite_failed": len(records) - len(ok),
            "topic_original": dict(Counter(str(r["topic_original"]) for r in records)),
            "topic_rewrite": dict(Counter(str(r["topic_rewrite"]) for r in ok)),
            "target_topic": TARGET_TOPIC[lx],
            "target_rate_original": float(np.mean([r["topic_original"] == TARGET_TOPIC[lx] for r in records])),
            "target_rate_rewrite": float(np.mean([r["topic_rewrite"] == TARGET_TOPIC[lx] for r in ok])),
            "leak_rate_original": float(np.mean([bool(r["leaks_original"]) for r in records])),
            "leak_rate_first_rewrite": float(np.mean([bool(r["leaks_first"]) for r in ok])),
            "leak_rate_rewrite": float(np.mean([bool(r["leaks"]) for r in ok])),
            "median_words_original": float(np.median([r["words_original"] for r in records])),
            "median_words_rewrite": float(np.median([r["words_rewrite"] for r in ok])),
            "top_leaked_terms": Counter(t for r in ok for t in r["leaks"]).most_common(15),
        }
        LOGGER.info(f"{lx}: {stats[lx]}")
    write_jsonl(log_rows, out / "rewrite_log.jsonl")
    write_json(stats, out / "build_stats.json")
    print(f"rewritten pools and stats saved to {out}")


def main():
    load_dotenv()
    config = simple_parsing.parse(LexicalConfig)
    assert (
        config.repro_two_sided_pools and config.ban_pools and config.ban_wildchat and config.output_dir
    ), "all input and output paths are required"
    setup_logging(config.output_dir)
    set_seed(config.seed)
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
