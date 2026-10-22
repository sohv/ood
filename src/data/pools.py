# builds the id mix, dolma reference, ood pools and the wildchat hardest-covered pool for exp01.

import gzip
import hashlib
import json
import logging
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

LOGGER = logging.getLogger(__name__)

FAMILY_SUFFIX = re.compile(r"[-_]?\d+([-_]\d+)*$")
DOLMA_URL = "https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-{:04d}.json.gz"
DOLMA_NONCODE_SHARDS = range(98)
SHINGLE_N = 13


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def word_shingles(text: str, n: int = SHINGLE_N) -> set[int]:
    words = normalise(text).split()
    return {stable_hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def family_of(prompt_id: str) -> str:
    return FAMILY_SUFFIX.sub("", prompt_id)


def load_dolci_dpo() -> pd.DataFrame:
    paths = [
        hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset")
        for i in range(4)
    ]
    df = pd.concat([pd.read_parquet(p, columns=["chosen", "prompt_id"]) for p in paths], ignore_index=True)
    df["first_user"] = df["chosen"].map(lambda msgs: next(m["content"] for m in msgs if m["role"] == "user"))
    df["n_messages"] = df["chosen"].map(len)
    df["family"] = df["prompt_id"].map(family_of)
    return df.drop(columns=["chosen"])


def load_tulu_sft() -> pd.DataFrame:
    # the olmo-2 sft mix; family is the source column, so the same sampler as dolci applies
    files = sorted(
        f
        for f in HfApi().list_repo_files("allenai/tulu-3-sft-olmo-2-mixture-0225", repo_type="dataset")
        if f.endswith(".parquet")
    )
    paths = [hf_hub_download("allenai/tulu-3-sft-olmo-2-mixture-0225", f, repo_type="dataset") for f in files]
    df = pd.concat([pd.read_parquet(p, columns=["messages", "source"]) for p in paths], ignore_index=True)
    df["first_user"] = df["messages"].map(lambda msgs: next((m["content"] for m in msgs if m["role"] == "user"), None))
    n_no_user = int(df["first_user"].isna().sum())
    LOGGER.info(f"tulu rows without a user message dropped: {n_no_user}")
    df = df[df["first_user"].notna()].copy()
    df["n_messages"] = df["messages"].map(len)
    df["family"] = df["source"]
    return df.drop(columns=["messages"])


def load_olmo2_dpo() -> pd.DataFrame:
    # the olmo-2 7b preference mix used for its dpo stage; first user turn of the chosen conversation
    files = sorted(
        f
        for f in HfApi().list_repo_files("allenai/olmo-2-1124-7b-preference-mix", repo_type="dataset")
        if f.endswith(".parquet")
    )
    paths = [hf_hub_download("allenai/olmo-2-1124-7b-preference-mix", f, repo_type="dataset") for f in files]
    df = pd.concat([pd.read_parquet(p, columns=["chosen", "source"]) for p in paths], ignore_index=True)
    df["first_user"] = df["chosen"].map(lambda msgs: next((m["content"] for m in msgs if m["role"] == "user"), None))
    return df[df["first_user"].notna()].drop(columns=["chosen"]).reset_index(drop=True)


def allocate_by_share(shares: pd.Series, total: int) -> pd.Series:
    raw = shares / shares.sum() * total
    counts = np.floor(raw).astype(int)
    remainder = (raw - counts).sort_values(ascending=False)
    counts[remainder.index[: total - counts.sum()]] += 1
    return counts


def sample_id_mix(dolci: pd.DataFrame, n_fit: int, n_test: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = dolci[dolci["n_messages"] == 2].copy()
    rows["norm"] = rows["first_user"].map(normalise)
    rows = rows[rows["norm"].str.len() > 0].drop_duplicates("norm")
    shares = rows["family"].value_counts()
    fit_counts = allocate_by_share(shares, n_fit)
    test_counts = allocate_by_share(shares, n_test)
    rng = np.random.default_rng(seed)
    fit_parts, test_parts = [], []
    for family, group in rows.groupby("family"):
        order = rng.permutation(len(group))
        n_f, n_t = int(fit_counts[family]), int(test_counts[family])
        assert n_f + n_t <= len(group), f"{family=} too small for allocation"
        fit_parts.append(group.iloc[order[:n_f]])
        test_parts.append(group.iloc[order[n_f : n_f + n_t]])
    return pd.concat(fit_parts), pd.concat(test_parts)


def read_dolma_shard(shard: int, max_lines: int) -> list[dict]:
    request = urllib.request.Request(DOLMA_URL.format(shard), headers={"User-Agent": "Mozilla/5.0"})
    docs = []
    with gzip.GzipFile(fileobj=urllib.request.urlopen(request, timeout=120)) as handle:
        for line in handle:
            doc = json.loads(line)
            docs.append({"dolma_id": doc["id"], "source": doc["source"], "shard": shard, "text": doc["text"]})
            if len(docs) >= max_lines:
                break
    return docs


def sample_dolma(n_docs: int, seed: int, lines_per_shard: int = 2000, min_chars: int = 200) -> list[dict]:
    shards = list(DOLMA_NONCODE_SHARDS)
    with ThreadPoolExecutor(16) as pool:
        per_shard = list(pool.map(lambda s: read_dolma_shard(s, lines_per_shard), shards))
    counts = allocate_by_share(pd.Series(1, index=shards), n_docs)
    rng = np.random.default_rng(seed)
    out = []
    for shard, docs in zip(shards, per_shard):
        docs = [d for d in docs if len(d["text"]) >= min_chars]
        picks = rng.choice(len(docs), size=int(counts[shard]), replace=False)
        out.extend(docs[i] for i in picks)
    return out


def load_legal(n: int, seed: int) -> list[str]:
    path = hf_hub_download("jonathanli/legal-advice-reddit", "train.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    texts = [
        f"{r['title'].strip()}\n\n{r['body'].strip()}"
        for r in rows
        if r["body"] and r["body"].strip() not in ("[removed]", "[deleted]")
    ]
    texts = list(dict.fromkeys(texts))
    return [texts[i] for i in np.random.default_rng(seed).choice(len(texts), size=n, replace=False)]


def load_medical(n: int, seed: int) -> list[str]:
    path = hf_hub_download(
        "lavita/ChatDoctor-HealthCareMagic-100k",
        "data/train-00000-of-00001-5e7cb295b9cff0bf.parquet",
        repo_type="dataset",
    )
    texts = list(dict.fromkeys(t.strip() for t in pd.read_parquet(path)["input"] if t and t.strip()))
    return [texts[i] for i in np.random.default_rng(seed).choice(len(texts), size=n, replace=False)]


def load_wildchat_english_first_turns(shards: list[int]) -> list[str]:
    paths = [
        hf_hub_download("allenai/WildChat-1M", f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        for i in shards
    ]
    df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
    df = df[df["language"] == "English"]
    texts = (
        df["conversation"].map(lambda conv: next((m["content"] for m in conv if m["role"] == "user"), "")).str.strip()
    )
    texts = texts[texts.str.len() > 0]
    norm = texts.map(normalise)
    return texts[~norm.duplicated()].tolist()


class MembershipIndex:
    def __init__(self, reference_texts: list[str]):
        normed = [normalise(t) for t in reference_texts]
        self.exact = set(normed)
        self.shingles: set[int] = set()
        for text in reference_texts:
            self.shingles |= word_shingles(text)
        # short rows are matched as a whole-word substring of the joined reference
        self.joined = " \n ".join(f" {t} " for t in normed)

    def is_member(self, text: str) -> bool:
        norm = normalise(text)
        if norm in self.exact:
            return True
        if len(norm.split()) < SHINGLE_N:
            return f" {norm} " in self.joined
        return not self.shingles.isdisjoint(word_shingles(text))


def load_reddit_posts(year: int, subreddits: list[str], max_per_subreddit: int) -> dict[str, list[str]]:
    path = hf_hub_download(
        "sentence-transformers/reddit-title-body", f"reddit_title_text_{year}.jsonl.gz", repo_type="dataset"
    )
    posts: dict[str, dict[str, None]] = {s: {} for s in subreddits}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            sub = row.get("subreddit")
            if sub not in posts or len(posts[sub]) >= max_per_subreddit:
                continue
            body = (row.get("body") or "").strip()
            if not body or body in ("[removed]", "[deleted]"):
                continue
            posts[sub][f"{row['title'].strip()}\n\n{body}"] = None
            if all(len(p) >= max_per_subreddit for p in posts.values()):
                break
    return {s: list(p) for s, p in posts.items()}


def load_reddit_posts_full(year: int, subreddits: list[str]) -> dict[str, list[str]]:
    path = hf_hub_download(
        "sentence-transformers/reddit-title-body", f"reddit_title_text_{year}.jsonl.gz", repo_type="dataset"
    )
    tails = {f'"subreddit": "{s}"}}': s for s in subreddits}
    posts: dict[str, dict[str, None]] = {s: {} for s in subreddits}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            # cheap tail match before parsing; rows end with the subreddit field
            sub = tails.get(line[line.rfind('"subreddit"') :].strip())
            if sub is None:
                continue
            row = json.loads(line)
            body = (row.get("body") or "").strip()
            if body and body not in ("[removed]", "[deleted]"):
                posts[sub][f"{row['title'].strip()}\n\n{body}"] = None
    return {s: list(p) for s, p in posts.items()}
