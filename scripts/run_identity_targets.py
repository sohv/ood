# openai_identity a1: scans Dolci's assistant turns (SFT, and DPO chosen and rejected) for maker and model names in a first-person identity context, labels a sample, and estimates how many training replies teach each identity.
# uv run -m scripts.run_identity_targets --run_dir results/raw/261021_openai_identity_v1 --model_id openai/gpt-4o-mini --seed 0

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

from scripts.run_identity_label import LABELS, SELF_REPORT_PROMPT
from src.data.pools import family_of
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
NAMES = re.compile(
    r"\b(OpenAI|ChatGPT|GPT-?3(\.5)?|GPT-?4o?|Alibaba|Tongyi|Qwen|Anthropic|Claude|Google|Gemini|Bard|Meta AI|LLaMA|Llama|Mistral|Ai2|AI2|Allen Institute|Olmo|OLMo|T[uü]lu)\b"
)
FIRST_PERSON = re.compile(
    r"\b(I am|I'm|I was (developed|created|trained|built|made)|my (creator|developer|developers|name|training|architecture)|developed by|created by|trained by|built by|made by)\b",
    re.IGNORECASE,
)
CLAIM_LABELS = ("OPENAI", "AI2", "OTHER_MAKER")


@dataclass
class IdentityTargetsConfig:
    run_dir: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0
    n_label: int = 3000


def candidate(text: str) -> bool:
    return bool(text) and bool(NAMES.search(text)) and bool(FIRST_PERSON.search(text))


def scan_sft() -> list[dict]:
    rows = []
    for i in range(15):
        p = hf_hub_download("allenai/Dolci-Instruct-SFT", f"data/train-{i:05d}-of-00015.parquet", repo_type="dataset")
        t = pq.read_table(p, columns=["id", "messages", "source_dataset"]).to_pylist()
        for r in t:
            user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            for m in r["messages"]:
                if m["role"] == "assistant" and candidate(m["content"]):
                    rows.append(
                        {
                            "set": "sft",
                            "side": "target",
                            "source": r["source_dataset"],
                            "row_id": r["id"],
                            "user": user,
                            "reply": m["content"],
                        }
                    )
        LOGGER.info(f"sft shard {i}: {len(rows)} candidate turns so far")
    return rows


def scan_dpo() -> list[dict]:
    rows = []
    for i in range(4):
        p = hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset")
        t = pq.read_table(p, columns=["prompt_id", "chosen", "rejected"]).to_pylist()
        for r in t:
            user = next((m["content"] for m in r["chosen"] if m["role"] == "user"), "")
            for side in ("chosen", "rejected"):
                for m in r[side]:
                    if m["role"] == "assistant" and candidate(m["content"]):
                        rows.append(
                            {
                                "set": "dpo",
                                "side": side,
                                "source": family_of(r["prompt_id"]),
                                "row_id": r["prompt_id"],
                                "user": user,
                                "reply": m["content"],
                            }
                        )
        LOGGER.info(f"dpo shard {i}: {len(rows)} candidate turns so far")
    return rows


def main():
    config = simple_parsing.parse(IdentityTargetsConfig)
    assert config.run_dir, "run_dir is required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    cands = scan_sft() + scan_dpo()
    groups = {k: [r for r in cands if (r["set"], r["side"]) == k] for k in {(r["set"], r["side"]) for r in cands}}
    sample = []
    for k, rs in groups.items():
        pick = rng.choice(len(rs), size=min(config.n_label, len(rs)), replace=False)
        sample += [{**rs[i], "id": f"{k[0]}_{k[1]}_{j}"} for j, i in enumerate(sorted(pick))]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [SELF_REPORT_PROMPT.format(user=r["user"][:1500], reply=r["reply"][:2500]) for r in sample],
            str(run / "targets_cache"),
            max_concurrent=32,
        )
    )
    for r, x in zip(sample, replies):
        r["label"] = parse_label(x, LABELS)
    write_jsonl(
        [{k: (v[:600] if k in ("user", "reply") else v) for k, v in r.items()} for r in sample],
        run / "targets_sample.jsonl",
    )

    out = {}
    for k, rs in groups.items():
        lab = [r for r in sample if (r["set"], r["side"]) == k]
        share = {L: sum(r["label"] == L for r in lab) / max(1, len(lab)) for L in LABELS}
        by_source = Counter(r["source"] for r in rs)
        per_source = {}
        for src, n in by_source.most_common():
            ls = [r for r in lab if r["source"] == src]
            sh = {L: (sum(r["label"] == L for r in ls) / len(ls) if len(ls) >= 20 else share[L]) for L in CLAIM_LABELS}
            per_source[src] = {
                "candidate_turns": n,
                "labelled": len(ls),
                **{f"est_{L}": round(n * sh[L], 1) for L in CLAIM_LABELS},
            }
        out[f"{k[0]}_{k[1]}"] = {
            "candidate_turns": len(rs),
            "labelled": len(lab),
            "label_failed": sum(r["label"] is None for r in lab),
            "share": share,
            "estimated_turns": {L: round(len(rs) * share[L]) for L in CLAIM_LABELS},
            "per_source": per_source,
        }
    ch, rj = out.get("dpo_chosen"), out.get("dpo_rejected")
    decision = {
        "sets": out,
        "openai_to_ai2_ratio_sft": (
            out["sft_target"]["estimated_turns"]["OPENAI"] / max(1, out["sft_target"]["estimated_turns"]["AI2"])
        )
        if "sft_target" in out
        else None,
        "openai_to_ai2_ratio_dpo_chosen": (ch["estimated_turns"]["OPENAI"] / max(1, ch["estimated_turns"]["AI2"]))
        if ch
        else None,
        "dpo_rejected_minus_chosen_openai_share": (rj["share"]["OPENAI"] - ch["share"]["OPENAI"])
        if ch and rj
        else None,
    }
    write_json(decision, run / "decision_a1.json")
    for k, v in out.items():
        print(
            f"{k}: {v['candidate_turns']} candidate turns, labelled {v['labelled']}, estimated OPENAI {v['estimated_turns']['OPENAI']}, AI2 {v['estimated_turns']['AI2']}, OTHER {v['estimated_turns']['OTHER_MAKER']}"
        )
        print(
            "  top sources: "
            + "; ".join(
                f"{s} {d['est_OPENAI']}/{d['est_AI2']}/{d['est_OTHER_MAKER']}"
                for s, d in list(v["per_source"].items())[:6]
            )
        )
    print(
        f"openai:ai2 ratio sft {decision['openai_to_ai2_ratio_sft']}, dpo chosen {decision['openai_to_ai2_ratio_dpo_chosen']}; rejected-chosen openai share {decision['dpo_rejected_minus_chosen_openai_share']}"
    )
    print(f"decision written to {run / 'decision_a1.json'}")


if __name__ == "__main__":
    main()
