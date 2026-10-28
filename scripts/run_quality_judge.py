# exp76: a binary sonnet answer-quality judge, checked on 24 user-labelled olmo replies, then gap member minus control "fails the request" on exp60's answered prompts.
# uv run -m scripts.run_quality_judge --phase items --run_dir results/raw/261027_exp76_items_v1 --items_dir data/processed/human_labels_exp76 --seed 0
# uv run -m scripts.run_quality_judge --phase judge --run_dir results/raw/261027_exp76_judge_v1 --labels_dir results/raw/261027_exp76_labels_v1 --seed 0

import asyncio
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.metrics.discovery import name_clusters
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PROMPT = """Below is a request a user sent to a chatbot, and the chatbot's reply.

Does the reply do what the user asked, correctly and completely? Judge only the reply's substance: a reply that refuses, stalls, asks an unneeded question, answers a different request, makes factual or code errors that matter, or leaves out a required part does not. Length and style do not matter.

User request:
{prompt}

Chatbot reply:
{reply}

Reply with one JSON object and nothing else: {{"ok": "yes" | "no", "reason": "<one sentence>"}}"""


@dataclass
class QualityConfig:
    run_dir: str = ""
    phase: str = ""
    pool_path: str = "data/processed/exp52/prompts.jsonl"
    rollouts: str = "results/raw/261022_exp52_rollouts_v1/rollouts.jsonl"
    exp60_judgements: str = "results/raw/261026_exp60_eval_v1/judgements.jsonl"
    items_dir: str = ""
    labels_dir: str = ""
    judge_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    n_check_per_role: int = 12
    check_agree: float = 0.75
    chars: int = 8000
    show_chars: int = 4000
    n_boot: int = 10_000
    worse_by: float = 0.05
    max_concurrent: int = 16
    cache_from: str = ""


def parse(reply: str | None) -> bool | None:
    if not reply:
        return None
    start = reply.find("{")
    if start < 0:
        return None
    # raw_decode stops at the end of the first object, so trailing text is ignored
    obj, _ = json.JSONDecoder().raw_decode(reply[start:])
    ok = str(obj.get("ok", "")).lower()
    return {"yes": True, "no": False}.get(ok)


def main():
    config = simple_parsing.parse(QualityConfig)
    assert config.run_dir and config.phase in ("items", "judge"), "run_dir and phase (items or judge) required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    pool = {p["id"]: p for p in load_jsonl(Path(config.pool_path))}
    reply = {r["prompt_id"]: r["reply"] for r in load_jsonl(Path(config.rollouts)) if r["rollout"] == 0}
    judged = [j for j in load_jsonl(Path(config.exp60_judgements))]
    LOGGER.info(f"{len(judged)} answered prompts from exp60")
    gap = [j for j in judged if j["arm"] == "gap"]
    check = [
        j["id"]
        for role in ("member", "control")
        for j in rng.choice([x for x in gap if x["role"] == role], size=config.n_check_per_role, replace=False)
    ]

    if config.phase == "items":
        assert config.items_dir, "items_dir required"
        order = rng.permutation(len(check))
        items = [
            {
                "id": f"f76q_{check[i]}",
                "kind": "quality",
                "active": True,
                "order": int(pos),
                "meta": "reply",
                "prompt": pool[check[i]]["text"][: config.show_chars],
                "reply": reply[check[i]][: config.show_chars],
            }
            for pos, i in enumerate(order)
        ]
        Path(config.items_dir).mkdir(parents=True, exist_ok=True)
        write_json(items, Path(config.items_dir) / "items.json")
        write_json({"check_ids": check}, run / "check_ids.json")
        print(f"Saved {len(items)} items to {Path(config.items_dir) / 'items.json'}")
        return

    assert config.labels_dir, "labels_dir required"
    human = {
        p.stem.removeprefix("f76q_"): json.loads(p.read_text())["ok"]
        for p in Path(config.labels_dir).glob("f76q_*.json")
    }
    assert set(human) == set(check), f"labels missing for {sorted(set(check) - set(human))}"
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    if config.cache_from:
        shutil.copytree(Path(config.cache_from), run / "judge_cache")
    ids = [j["id"] for j in judged]
    prompts = [PROMPT.format(prompt=pool[i]["text"][: config.chars], reply=reply[i][: config.chars]) for i in ids]
    replies = asyncio.run(
        name_clusters(client, config.judge_model, prompts, str(run / "judge_cache"), config.max_concurrent)
    )
    ok = {i: parse(r) for i, r in zip(ids, replies, strict=True)}
    write_jsonl([{"id": i, "ok": ok[i], "raw": r} for i, r in zip(ids, replies, strict=True)], run / "judgements.jsonl")

    scored = [i for i in check if human[i] in ("yes", "no") and ok[i] is not None]
    agree = sum((human[i] == "yes") == ok[i] for i in scored)
    is_valid = agree >= config.check_agree * len(scored)
    LOGGER.info(f"judge check {agree} of {len(scored)} (valid {is_valid})")

    def effect(arm: str) -> dict:
        by = defaultdict(lambda: {"member": [], "control": []})
        for j in judged:
            if j["arm"] == arm and ok[j["id"]] is not None:
                by[j["group"]][j["role"]].append(0.0 if ok[j["id"]] else 1.0)
        groups = [g for g, v in by.items() if v["member"] and v["control"]]
        diff = np.array([np.mean(by[g]["member"]) - np.mean(by[g]["control"]) for g in groups])
        boots = diff[rng.integers(0, len(diff), (config.n_boot, len(diff)))].mean(1)
        return {
            "effect": round(float(diff.mean()), 4),
            "ci95": [round(float(np.quantile(boots, q)), 4) for q in (0.025, 0.975)],
            "n_topics": len(groups),
            "fail_rate_member": round(float(np.mean([np.mean(by[g]["member"]) for g in groups])), 4),
            "fail_rate_control": round(float(np.mean([np.mean(by[g]["control"]) for g in groups])), 4),
        }

    eff = {a: effect(a) for a in ("gap", "random")}
    g = eff["gap"]
    verdict = (
        "invalid_judge"
        if not is_valid
        else "worse_in_gaps"
        if g["effect"] >= config.worse_by and g["ci95"][0] > 0
        else "not_worse"
        if g["ci95"][1] < config.worse_by
        else "ambiguous"
    )
    decision = {
        "verdict": verdict,
        "judge_check": {
            "agree": agree,
            "of": len(scored),
            "unsure_or_unparsed": len(check) - len(scored),
            "valid": is_valid,
            "pairs": [{"id": i, "human": human[i], "judge_ok": ok[i]} for i in check],
        },
        "effects": eff,
        "judge_unparsed": sum(v is None for v in ok.values()),
        "n_judged": len(ids),
    }
    write_json(decision, run / "decision.json")
    print(json.dumps({k: decision[k] for k in ("verdict", "effects")}, indent=1))
    print(f"judge check {agree} of {len(scored)}; decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
