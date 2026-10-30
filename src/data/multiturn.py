# loaders for multiturn_tooluse: wildchat second user turns with context, dolci multi-turn family rows, hermes tool-calling rows.

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from src.data.pools import FAMILY_SUFFIX, normalise


def _turns(conv) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in conv]


def wildchat_second_turns(shards: list[int]) -> list[dict]:
    # english conversations whose first three turns are user, assistant, user; the second user turn is the scored message
    paths = [
        hf_hub_download("allenai/WildChat-1M", f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        for i in shards
    ]
    df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
    df = df[df["language"] == "English"]
    rows, seen = [], set()
    for conv in df["conversation"]:
        turns = _turns(conv)
        if len(turns) < 3 or [t["role"] for t in turns[:3]] != ["user", "assistant", "user"]:
            continue
        if not all(t["content"].strip() for t in turns[:3]):
            continue
        key = normalise(turns[2]["content"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"messages": turns[:3], "text": turns[2]["content"].strip()})
    return rows


def dolci_dpo_conversations() -> pd.DataFrame:
    paths = [
        hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset")
        for i in range(4)
    ]
    df = pd.concat([pd.read_parquet(p, columns=["chosen", "prompt_id"]) for p in paths], ignore_index=True)
    df["messages"] = df["chosen"].map(_turns)
    df["n_messages"] = df["messages"].map(len)
    df["family"] = df["prompt_id"].map(lambda s: FAMILY_SUFFIX.sub("", s))
    return df.drop(columns=["chosen"])


def dolci_multiturn_rows(df: pd.DataFrame) -> pd.DataFrame:
    # the prefix up to the last user turn is the prompt; the chosen reply is dropped
    mt = df[df["n_messages"] > 2].copy()
    mt["messages"] = mt["messages"].map(lambda t: t[:-1])
    mt = mt[mt["messages"].map(lambda t: t[-1]["role"] == "user")]
    mt["text"] = mt["messages"].map(lambda t: t[-1]["content"].strip())
    return mt[mt["text"].str.len() > 0]


def dolci_user_turns(df: pd.DataFrame) -> list[str]:
    return [t["content"] for turns in df["messages"] for t in turns if t["role"] == "user"]


def hermes_tool_rows() -> list[dict]:
    path = hf_hub_download(
        "NousResearch/hermes-function-calling-v1", "func-calling-singleturn.json", repo_type="dataset"
    )
    rows = []
    for r in json.loads(Path(path).read_text()):
        conv = r["conversations"]
        if [m["from"] for m in conv[:2]] != ["system", "human"]:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": conv[0]["value"]},
                    {"role": "user", "content": conv[1]["value"]},
                ],
                "text": conv[1]["value"].strip(),
            }
        )
    return rows
