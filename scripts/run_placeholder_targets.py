# exp35 step 1: finds placeholder-name tokens in Dolci's user turns (SFT and DPO) and labels whether the target reply refuses.
# uv run -m scripts.run_placeholder_targets --run_dir results/raw/261021_exp35_placeholder_v1 --sft_path data/processed/exp31/dolci_sft_first_user.parquet --model_id openai/gpt-4o-mini --seed 0

import asyncio
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import simple_parsing
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

from scripts.run_refusal_label import REFUSAL_LABELS, REFUSAL_PROMPT
from src.data.pools import family_of
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"NAME_\d+|PERSON_\d+|\[NAME\]|<NAME>|\[name\]|\{name\}", re.IGNORECASE)


@dataclass
class PlaceholderTargetsConfig:
    run_dir: str = ""
    sft_path: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0
    n_label: int = 3000


def main():
    config = simple_parsing.parse(PlaceholderTargetsConfig)
    assert config.run_dir and config.sft_path, "run_dir and sft_path required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rows = []
    # sft: the parquet of first user turns tells which rows to look at; their assistant turns come from the shards
    sft = pd.read_parquet(config.sft_path, columns=["id", "first_user", "source_dataset"])
    hit_ids = set(sft.loc[sft["first_user"].fillna("").str.contains(PLACEHOLDER.pattern, regex=True, case=False), "id"])
    LOGGER.info(f"sft rows with a placeholder in the first user turn: {len(hit_ids)}")
    for i in range(15):
        p = hf_hub_download("allenai/Dolci-Instruct-SFT", f"data/train-{i:05d}-of-00015.parquet", repo_type="dataset")
        for r in pq.read_table(p, columns=["id", "messages", "source_dataset"]).to_pylist():
            if r["id"] in hit_ids:
                user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
                reply = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
                rows.append(
                    {
                        "set": "sft",
                        "side": "target",
                        "source": r["source_dataset"],
                        "row_id": r["id"],
                        "user": user,
                        "reply": reply,
                    }
                )
    for i in range(4):
        p = hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset")
        for r in pq.read_table(p, columns=["prompt_id", "chosen", "rejected"]).to_pylist():
            user = next((m["content"] for m in r["chosen"] if m["role"] == "user"), "")
            if PLACEHOLDER.search(user):
                for side in ("chosen", "rejected"):
                    reply = next((m["content"] for m in r[side] if m["role"] == "assistant"), "")
                    rows.append(
                        {
                            "set": "dpo",
                            "side": side,
                            "source": family_of(r["prompt_id"]),
                            "row_id": r["prompt_id"],
                            "user": user,
                            "reply": reply,
                        }
                    )
    groups = {k: [r for r in rows if (r["set"], r["side"]) == k] for k in {(r["set"], r["side"]) for r in rows}}
    sample = []
    for k, rs in groups.items():
        pick = rng.choice(len(rs), size=min(config.n_label, len(rs)), replace=False)
        sample += [{**rs[i], "id": f"{k[0]}_{k[1]}_{j}"} for j, i in enumerate(sorted(pick))]
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
        [{k: (v[:600] if k in ("user", "reply") else v) for k, v in r.items()} for r in sample],
        run / "targets_sample.jsonl",
    )
    out = {}
    for k, rs in groups.items():
        lab = [r for r in sample if (r["set"], r["side"]) == k]
        share = {L: sum(r["label"] == L for r in lab) / max(1, len(lab)) for L in REFUSAL_LABELS}
        out[f"{k[0]}_{k[1]}"] = {
            "rows": len(rs),
            "labelled": len(lab),
            "share": share,
            "by_source": dict(Counter(r["source"] for r in rs).most_common(10)),
        }
    targets = [r for r in sample if (r["set"], r["side"]) in (("sft", "target"), ("dpo", "chosen"))]
    n_rows = sum(len(groups[k]) for k in groups if k in (("sft", "target"), ("dpo", "chosen")))
    refuse_share = sum(r["label"] == "REFUSES" for r in targets) / max(1, len(targets))
    reading = (
        "learned_from_data"
        if n_rows >= 20 and refuse_share >= 0.70
        else "not_in_data"
        if n_rows < 20 or refuse_share < 0.30
        else "ambiguous"
    )
    ch, rj = out.get("dpo_chosen"), out.get("dpo_rejected")
    decision = {
        "reading": reading,
        "placeholder_rows_sft_plus_dpo_chosen": n_rows,
        "refuses_share_targets": refuse_share,
        "sets": out,
        "dpo_rejected_minus_chosen_refuses": (rj["share"]["REFUSES"] - ch["share"]["REFUSES"]) if ch and rj else None,
    }
    write_json(decision, run / "decision_step1.json")
    print(f"reading: {reading}; placeholder rows {n_rows}; REFUSES share of targets {refuse_share:.3f}")
    for k, v in out.items():
        print(
            f"  {k}: {v['rows']} rows, share {{{', '.join(f'{L} {v['share'][L]:.2f}' for L in REFUSAL_LABELS)}}}; sources {list(v['by_source'].items())[:4]}"
        )
    print(f"decision written to {run / 'decision_step1.json'}")


if __name__ == "__main__":
    main()
