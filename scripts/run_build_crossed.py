# exp30: builds crossed test families (topic from one exp27 family, source from another) from real messages, filtered by an llm label.
# uv run -m scripts.run_build_crossed --output_dir data/processed/exp30 --exp27_dir data/processed/exp27 --model_id openai/gpt-4o-mini --seed 0

import asyncio
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.data.needles import wildchat_first_turns
from src.data.pools import load_reddit_posts, normalise
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SHARDS = list(range(14))
KEYWORDS = {
    "cooking": re.compile(
        r"\b(recipes?|cook|cooking|bake|baking|marinade|marinate|simmer|saut[eé]|roast|oven|dough|ingredients?|dish|meal|fry|frying|boil|grill)\b",
        re.IGNORECASE,
    ),
    "programming": re.compile(
        r"\b(python|javascript|java|c\+\+|c#|sql|html|css|code|function|error|bug|compile|script|programming|api|regex|array|loop|variable)\b",
        re.IGNORECASE,
    ),
    "legal": re.compile(
        r"\b(lawyer|attorney|sue|lawsuit|legal|court|landlord|tenant|contract|custody|divorce|arrested|police|visa|immigration|rights|law)\b",
        re.IGNORECASE,
    ),
    "japanese_cooking": re.compile(
        "料理|レシピ|食材|調理|作り方|味噌|カレー|スープ|パスタ|弁当|献立|炒め|煮込|焼き|食べ|美味|ご飯|おかず|野菜|お菓子|ケーキ|パン|朝食|昼食|夕食|晩御飯|食事"
    ),
}
TOPIC_HELP = {
    "maths": "a maths problem: they want it solved or explained",
    "cooking": "cooking or food: they want a recipe, a technique or advice about food",
    "programming": "programming: they want code written, fixed or explained",
    "legal": "a legal matter: they want to know their rights or what to do legally",
}
LABEL_PROMPT = """Below is a message a user sent to an AI chatbot. Answer YES if the message is a genuine question or request for help about {help}.{extra} Answer NO otherwise, including when the message is a writing, marketing, translation, summarising or roleplay task that merely mentions the topic, or an instruction telling the assistant how to behave.

<message>
{message}
</message>

Reply with YES or NO only."""
JAPANESE_EXTRA = " The message must also be written in Japanese."
# crossed family -> (topic family, how candidates are found)
FAMILIES = {
    "reddit_maths": "maths",
    "wildchat_cooking": "cooking",
    "wildchat_programming": "programming",
    "wildchat_legal": "legal",
    "japanese_cooking": "cooking",
}


@dataclass
class CrossedBuildConfig:
    output_dir: str = ""
    exp27_dir: str = ""
    sharechat_path: str = "data/processed/exp18/sharechat.jsonl"
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0
    n_test: int = 50
    min_japanese: int = 30
    max_candidates: int = 600
    max_chars: int = 2000


def candidates(config: CrossedBuildConfig, rng: np.random.Generator) -> dict[str, list[str]]:
    reddit = load_reddit_posts(2019, ["learnmath", "askmath", "MathHelp"], 600)
    en = wildchat_first_turns(SHARDS, english_only=True)
    ja_wc = wildchat_first_turns(SHARDS)
    ja = [t for t in ja_wc if re.search(r"[぀-ヿ]", t)]
    ja += [r["text"] for r in load_jsonl(config.sharechat_path) if r.get("language") == "Japanese"]
    raw = {
        "reddit_maths": [t for s in reddit.values() for t in s],
        "wildchat_cooking": [t for t in en if KEYWORDS["cooking"].search(t)],
        "wildchat_programming": [t for t in en if KEYWORDS["programming"].search(t)],
        "wildchat_legal": [t for t in en if KEYWORDS["legal"].search(t)],
        "japanese_cooking": [t for t in ja if KEYWORDS["japanese_cooking"].search(t)],
    }
    out = {}
    for fam, texts in raw.items():
        pool = list(
            dict.fromkeys(t.strip() for t in texts if t.strip() and len(t) <= config.max_chars and "NAME_" not in t)
        )
        if len(pool) > config.max_candidates:
            pool = [pool[i] for i in sorted(rng.choice(len(pool), size=config.max_candidates, replace=False))]
        out[fam] = pool
    return out


def main():
    config = simple_parsing.parse(CrossedBuildConfig)
    assert config.output_dir and config.exp27_dir, "output_dir and exp27_dir are required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    used = {
        normalise(r["text"])
        for name in ("gen_pool", "test")
        for r in load_jsonl(Path(config.exp27_dir) / f"{name}.jsonl")
    }
    cands = candidates(config, rng)
    cands = {fam: [t for t in texts if normalise(t) not in used] for fam, texts in cands.items()}
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    flat = [(fam, t) for fam, texts in cands.items() for t in texts]
    prompts = [
        LABEL_PROMPT.format(
            help=TOPIC_HELP[FAMILIES[fam]], extra=JAPANESE_EXTRA if fam == "japanese_cooking" else "", message=t
        )
        for fam, t in flat
    ]
    replies = asyncio.run(name_clusters(client, config.model_id, prompts, str(out / "label_cache"), max_concurrent=32))
    labelled = [
        {"id": f"cand_{fam}_{i}", "text": t, "family": fam, "label": parse_label(r, ["YES", "NO"])}
        for i, ((fam, t), r) in enumerate(zip(flat, replies))
    ]
    write_jsonl(labelled, out / "labelled.jsonl")

    test, counts = [], {}
    for fam in FAMILIES:
        yes = [r["text"] for r in labelled if r["family"] == fam and r["label"] == "YES"]
        n_null = sum(r["family"] == fam and r["label"] is None for r in labelled)
        counts[fam] = {"candidates": len(cands[fam]), "yes": len(yes), "label_failed": n_null}
        if fam == "japanese_cooking" and len(yes) < config.min_japanese:
            LOGGER.warning(f"{fam}: only {len(yes)} pass the label, under {config.min_japanese}; family dropped")
            counts[fam]["drawn"] = 0
            continue
        n = min(config.n_test, len(yes))
        pick = rng.choice(len(yes), size=n, replace=False)
        test += [
            {"id": f"x_{fam}_{i}", "text": yes[j], "family": fam, "topic": FAMILIES[fam]} for i, j in enumerate(pick)
        ]
        counts[fam]["drawn"] = n
    write_jsonl(test, out / "crossed_test.jsonl")
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "counts": counts, "topic_of": FAMILIES}, out / "build.json"
    )
    print(f"{counts=}")
    print(f"{len(test)} crossed test questions written to {out / 'crossed_test.jsonl'}")


if __name__ == "__main__":
    main()
