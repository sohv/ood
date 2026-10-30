# loaders for density_ratio_needles: a multilingual wildchat deployment pool and gap / control needle banks.

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from src.data.pools import MembershipIndex, normalise


def wildchat_first_turns(shards: list[int], english_only: bool = False) -> list[str]:
    paths = [
        hf_hub_download("allenai/WildChat-1M", f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        for i in shards
    ]
    df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
    if english_only:
        df = df[df["language"] == "English"]
    texts = df["conversation"].map(lambda c: next((m["content"] for m in c if m["role"] == "user"), "")).str.strip()
    texts = texts[texts.str.len() > 0]
    return texts[~texts.map(normalise).duplicated()].tolist()


def legal_test_posts() -> list[str]:
    path = hf_hub_download("jonathanli/legal-advice-reddit", "test.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    return list(
        dict.fromkeys(
            f"{r['title'].strip()}\n\n{r['body'].strip()}"
            for r in rows
            if r["body"] and r["body"].strip() not in ("[removed]", "[deleted]")
        )
    )


def medical_questions(exclude: set[str]) -> list[str]:
    path = hf_hub_download(
        "lavita/ChatDoctor-HealthCareMagic-100k",
        "data/train-00000-of-00001-5e7cb295b9cff0bf.parquet",
        repo_type="dataset",
    )
    return [
        t
        for t in dict.fromkeys(t.strip() for t in pd.read_parquet(path)["input"] if t and t.strip())
        if t not in exclude
    ]


def aya_inputs() -> list[str]:
    path = hf_hub_download("CohereForAI/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset")
    return list(
        dict.fromkeys(t.strip() for t in pd.read_parquet(path, columns=["inputs"])["inputs"] if t and t.strip())
    )


def magicoder_problems() -> list[str]:
    path = hf_hub_download(
        "ise-uiuc/Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl", repo_type="dataset"
    )
    return list(dict.fromkeys(json.loads(line)["problem"].strip() for line in open(path)))


def sample_non_members(texts: list[str], index: MembershipIndex, n: int, seed: int) -> tuple[list[str], int]:
    kept, n_member = [], 0
    for i in np.random.default_rng(seed).permutation(len(texts)):
        if index.is_member(texts[i]):
            n_member += 1
            continue
        kept.append(texts[i])
        if len(kept) == n:
            return kept, n_member
    raise AssertionError(f"only {len(kept)} non-member rows, needed {n}")


def lmsys_first_turns(shards: list[int], token: str) -> list[dict]:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        # offline: the hub cannot be listed, so the shard files come from the local snapshot
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        snaps = home / "hub" / "datasets--lmsys--lmsys-chat-1m" / "snapshots"
        files = [p.relative_to(s).as_posix() for s in snaps.iterdir() for p in s.rglob("*.parquet")]
    else:
        files = [
            f
            for f in HfApi(token=token).list_repo_files("lmsys/lmsys-chat-1m", repo_type="dataset")
            if f.endswith(".parquet")
        ]
    # shard i is the file named train-0000i-of-..., which is also position i in the sorted hub listing
    by_index = {int(re.search(r"train-(\d{5})-of", f).group(1)): f for f in files}
    paths = [hf_hub_download("lmsys/lmsys-chat-1m", by_index[i], repo_type="dataset", token=token) for i in shards]
    df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
    df["text"] = (
        df["conversation"].map(lambda c: next((m["content"] for m in c if m["role"] == "user"), "")).str.strip()
    )
    df = df[df["text"].str.len() > 0]
    df = df[~df["text"].map(normalise).duplicated()]
    return df[["text", "language"]].to_dict("records")


def wildchat_language_map(shards: list[int]) -> dict[str, str]:
    paths = [
        hf_hub_download("allenai/WildChat-1M", f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        for i in shards
    ]
    df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
    text = df["conversation"].map(lambda c: next((m["content"] for m in c if m["role"] == "user"), "")).str.strip()
    return dict(zip(text.map(normalise), df["language"]))
