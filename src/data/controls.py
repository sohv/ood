# loaders for the exp08 control pools: membership-half upstreams, same-domain-other-source pools and deployment traffic.

import gzip
import json

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from src.data.pools import MembershipIndex, allocate_by_share, normalise


def dedupe(texts: list[str]) -> list[str]:
    seen, out = set(), []
    for t in texts:
        t = (t or "").strip()
        key = normalise(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def load_aya_inputs() -> list[str]:
    path = hf_hub_download("CohereForAI/aya_dataset", "data/train-00000-of-00001.parquet", repo_type="dataset")
    return dedupe(pd.read_parquet(path, columns=["inputs"])["inputs"].tolist())


def load_evol_codealpaca() -> list[str]:
    path = hf_hub_download("theblackcat102/evol-codealpaca-v1", "train.jsonl", repo_type="dataset")
    return dedupe([json.loads(line)["instruction"] for line in open(path)])


def load_wildjailbreak() -> list[str]:
    path = hf_hub_download("allenai/wildjailbreak", "train/train.tsv", repo_type="dataset")
    df = pd.read_csv(path, sep="\t", usecols=["vanilla", "adversarial"], keep_default_na=False)
    return dedupe([a if a.strip() else v for v, a in zip(df["vanilla"], df["adversarial"])])


def load_magicoder() -> list[str]:
    path = hf_hub_download("ise-uiuc/Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl", repo_type="dataset")
    return dedupe([json.loads(line)["problem"] for line in open(path)])


def load_bactrian(n: int, seed: int) -> list[dict]:
    langs = sorted(f.split("/")[1].removesuffix(".json.gz") for f in HfApi().list_repo_files("MBZUAI/Bactrian-X", repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".json.gz"))
    langs = [lang for lang in langs if lang != "en"]
    counts = allocate_by_share(pd.Series(1, index=langs), n)
    rng = np.random.default_rng(seed)
    out = []
    for lang in langs:
        rows = json.load(gzip.open(hf_hub_download("MBZUAI/Bactrian-X", f"data/{lang}.json.gz", repo_type="dataset")))
        texts = dedupe([(r["instruction"] or "") + (f"\n\n{r['input']}" if (r["input"] or "").strip() else "") for r in rows])
        # oversample 3x per language so membership drops can be replaced from the same language
        picks = rng.permutation(len(texts))[: 3 * int(counts[lang])]
        out.extend({"text": texts[i], "language": lang, "quota": int(counts[lang])} for i in picks)
    return out


def load_first_turns(repo: str, filename: str) -> pd.DataFrame:
    df = pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"), columns=["conversation", "language"])
    df["text"] = df["conversation"].map(lambda conv: next((m["content"] for m in conv if m["role"] == "user"), "")).str.strip()
    df = df[df["text"].str.len() > 0]
    return df[~df["text"].map(normalise).duplicated()][["text", "language"]].reset_index(drop=True)


def take_non_members(texts: list[str], index: MembershipIndex, n: int, seed: int) -> tuple[list[str], int, int]:
    out, checked, removed = [], 0, 0
    for i in np.random.default_rng(seed).permutation(len(texts)):
        checked += 1
        if index.is_member(texts[i]):
            removed += 1
            continue
        out.append(texts[i])
        if len(out) == n:
            return out, checked, removed
    raise ValueError(f"only {len(out)} non-members found, wanted {n}")
