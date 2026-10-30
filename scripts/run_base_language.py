# tests whether olmo's extra oddness in non-english is already in the base model, using an in-context alignment prefix (urial) in each prompt's language.
# uv run --no-sync -m scripts.run_base_language --phase prompts --pool_out data/processed/language_weird/pool.jsonl --output_dir results/raw/261030_base_language_v1 --model_id anthropic/claude-sonnet-4.6 --seed 0

import asyncio
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from scripts.run_behaviour_far import cached_calls, load_rollouts
from scripts.run_language_weird import (
    JUDGES,
    LANGS,
    TRANSLATE_PROMPT,
    TRANSLATOR,
    VERSIONS,
    LanguageWeirdConfig,
    holm,
    paired,
    run_judge,
)
from scripts.run_weird_tail import ODD_FLAGS
from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PREFIX = """# Instructions

Below are conversations between a user and an AI assistant. The user's message is under "# Query:" and the assistant's reply is under "# Answer:". The assistant is helpful, honest and harmless. It answers in the language the user writes in, follows the user's instructions, and says so when it does not know something.

# Query:
What causes the seasons on Earth?

# Answer:
The seasons come from the tilt of Earth's axis, about 23.5 degrees, not from changes in distance to the Sun. As Earth orbits the Sun, the hemisphere tilted towards the Sun gets more direct sunlight and longer days, so it has summer, while the hemisphere tilted away gets less direct light and shorter days, so it has winter. Six months later the situation is reversed.

# Query:
Write a short, friendly message inviting my neighbours to a garden party on Saturday at 4 pm.

# Answer:
Hi neighbours! We're having a small garden party this Saturday at 4 pm and would love for you to join us. There will be food, drinks and good company, so just bring yourselves (and a chair if you have a spare one). Hope to see you there!

# Query:
How do I remove a coffee stain from a cotton shirt?

# Answer:
Act quickly if you can. Rinse the stain from the back with cold water to push the coffee out of the fibres. Rub a little liquid laundry detergent or dish soap into the stain and let it sit for about 10 minutes. Then wash the shirt as usual in the warmest water the care label allows. Check that the stain is gone before drying, because heat from the dryer can set it. For an old stain, soak the shirt for 30 minutes in water with a spoonful of oxygen bleach first.
"""


@dataclass
class BaseLanguageConfig(LanguageWeirdConfig):
    lw_dir: str = "results/raw/261030_language_weird_v1"
    data_dir: str = "data/processed/base_language"


def build(prefix: str, query: str) -> str:
    return f"{prefix}\n# Query:\n{query}\n\n# Answer:\n"


async def prompts(config: BaseLanguageConfig) -> None:
    data = Path(config.data_dir)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    # each block between markers is translated on its own; the markers stay in english so the stop string still works
    blocks = re.split(r"\n# (?:Query|Answer):\n", PREFIX)
    assert len(blocks) == 7, "the prefix must hold a preamble and three query-answer pairs"
    jobs = [(lang, i) for lang in LANGS for i in range(len(blocks))]
    asks = [TRANSLATE_PROMPT.format(language=LANGS[lang], text=blocks[i].strip()) for lang, i in jobs]
    replies = await cached_calls(client, TRANSLATOR, asks, "cache/base_language/translate", 3 * config.max_concurrent)
    done: dict[str, list[str]] = {lang: [""] * len(blocks) for lang in LANGS}
    for (lang, i), reply in zip(jobs, replies, strict=True):
        found = re.findall(r"<translation>(.*?)</translation>", reply or "", flags=re.DOTALL)
        assert found, f"no translation for {lang} block {i}"
        done[lang][i] = found[-1].strip()
    prefixes = {"en": PREFIX}
    for lang, b in done.items():
        prefixes[lang] = b[0] + "".join(f"\n\n# Query:\n{b[i]}\n\n# Answer:\n{b[i + 1]}" for i in (1, 3, 5)) + "\n"
    write_json(prefixes, data / "prefixes.json")
    rows = [
        {
            "id": r["id"],
            "prompt_text": build(prefixes[r["lang"]], r["text"]),
            "lang": r["lang"],
            "version": r["version"],
        }
        for r in load_jsonl(config.pool_out)
    ]
    write_jsonl(rows, data / "prompts.jsonl")
    print(f"Prompts saved to {data / 'prompts.jsonl'} and {data / 'prefixes.json'}")
    print(json.dumps(dict(Counter(r["version"] for r in rows))))


def odd_table(judge_path: Path, pool: list[dict]) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for j in load_jsonl(judge_path):
        if j["parsed"] is not None:
            per.setdefault(j["prompt_id"], {})[j["judge"]] = j["parsed"]
    table = {}
    for r in pool:
        js = per.get(r["id"], {})
        odd = {jd: float(np.mean([any(f.values()) for f in js[jd]["flags"]])) for jd in js}
        table[r["id"]] = {
            "odd_share": float(np.mean(list(odd.values()))) if odd else None,
            **{f"odd_{jd.split('/')[1]}": odd.get(jd) for jd in JUDGES},
            **{
                f"flag_{f}": float(np.mean([np.mean([x[f] for x in p["flags"]]) for p in js.values()])) if js else None
                for f in ODD_FLAGS
            },
            "n_judges": len(js),
        }
    return table


def analyse(config: BaseLanguageConfig) -> None:
    out = Path(config.output_dir)
    pool = load_jsonl(config.pool_out)
    meta = {r["id"]: r for r in pool}
    base = odd_table(out / "judge.jsonl", pool)
    inst = odd_table(Path(config.lw_dir) / "judge.jsonl", pool)
    rolls = load_rollouts(config)
    for pid, row in base.items():
        row["hit_cap"] = float(np.mean([s["finish_reason"] == "length" for s in rolls[pid]]))
        row["empty"] = float(np.mean([not s["reply"].strip() for s in rolls[pid]]))
    write_jsonl(
        [
            {"id": pid, **meta[pid], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}}
            for pid, row in base.items()
        ],
        out / "per_prompt.jsonl",
    )

    def pairs(version: str, lang: str | None = None) -> list[tuple[str, str]]:
        # (non-english id, english id) for both directions
        if version == "native":
            return [
                (r["id"], f"{r['id']}_en")
                for r in pool
                if r["version"] == "native" and f"{r['id']}_en" in meta and lang in (None, r["lang"])
            ]
        return [(r["id"], r["pair"]) for r in pool if r["version"] == "en_to_x" and lang in (None, r["lang"])]

    def gap(t: dict, ps: list[tuple[str, str]], m: str) -> np.ndarray:
        return np.array([np.nan if t[a][m] is None or t[b][m] is None else t[a][m] - t[b][m] for a, b in ps])

    rng = np.random.default_rng(config.seed)
    primary = {}
    for v in ("native", "en_to_x"):
        ps = pairs(v)
        primary[f"base_gap|{v}"] = paired(gap(base, ps, "odd_share"), rng, config.n_boot)
        primary[f"base_minus_instruct_gap|{v}"] = paired(
            gap(base, ps, "odd_share") - gap(inst, ps, "odd_share"), rng, config.n_boot
        )
        primary[f"instruct_gap|{v}"] = paired(gap(inst, ps, "odd_share"), rng, config.n_boot)
    tested = [k for k in primary if not k.startswith("instruct_gap")]
    for k, v in holm({k: primary[k]["p"] for k in tested}).items():
        primary[k]["p_holm"] = v

    measures = [
        *(f"odd_{j.split('/')[1]}" for j in JUDGES),
        "flag_garbled",
        "flag_continues_text",
        "flag_wrong_language",
    ]
    secondary = {
        f"{t_name}|{m}|{v}": paired(gap(t, pairs(v), m), rng, config.n_boot)
        for t_name, t in (("base", base), ("instruct", inst))
        for m in measures
        for v in ("native", "en_to_x")
    }
    by_lang = {
        lang: {
            v: {
                "base_gap": round(float(np.nanmean(gap(base, pairs(v, lang), "odd_share"))), 4),
                "instruct_gap": round(float(np.nanmean(gap(inst, pairs(v, lang), "odd_share"))), 4),
            }
            for v in ("native", "en_to_x")
        }
        for lang in LANGS
    }
    means = {
        f"{t_name}|{v}": {
            m: round(
                float(
                    np.nanmean([np.nan if t[r["id"]][m] is None else t[r["id"]][m] for r in pool if r["version"] == v])
                ),
                4,
            )
            for m in ["odd_share", *measures]
        }
        for t_name, t in (("base", base), ("instruct", inst))
        for v in VERSIONS
    }
    decision = {
        "n_prompts": len(pool),
        "n_base_judge_missing": sum(r["n_judges"] < len(JUDGES) for r in base.values()),
        "base_hit_cap": {
            v: round(float(np.mean([base[r["id"]]["hit_cap"] for r in pool if r["version"] == v])), 4) for v in VERSIONS
        },
        "base_empty": {
            v: round(float(np.mean([base[r["id"]]["empty"] for r in pool if r["version"] == v])), 4) for v in VERSIONS
        },
        "means": means,
        "primary": primary,
        "secondary": secondary,
        "by_language": by_lang,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(
        json.dumps(
            {"primary": primary, "means": {k: v["odd_share"] for k, v in means.items()}, "by_language": by_lang},
            indent=1,
        )
    )


def main() -> None:
    config = simple_parsing.parse(BaseLanguageConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out / config.phase)
    set_seed(config.seed)
    if config.phase == "prompts":
        asyncio.run(prompts(config))
    elif config.phase == "judge":
        asyncio.run(run_judge(config))
    else:
        analyse(config)


if __name__ == "__main__":
    main()
