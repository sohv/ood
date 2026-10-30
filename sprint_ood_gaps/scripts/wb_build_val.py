# builds the validation pools used to tune the word baselines, disjoint from every test pool, plus provenance checks
# of Sohan's committed test pools against the public sources.
#   val_legal    300 posts from jonathanli/legal-advice-reddit validation.jsonl (test pools used train.jsonl / test.jsonl)
#   val_medical  300 HealthCareMagic inputs not in repro_two_sided medical or the density_ratio_needles medical needle bank
#   val_wildchat_{dolci,tulu}  500 English WildChat-1M first turns from shard 7 (test pools used shards 0,5,10 and 1,2,3),
#                mix members removed with Sohan's MembershipIndex (src/data/pools.py:159-175), per mix
# Loading recipes copied from src/data/pools.py:40-67 (dolci / tulu), :122-156 (legal, medical, wildchat english).
# usage: .venv/bin/python -m scripts.wb_build_val

import json
import time

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from scripts.wb_common import INPUTS, RUN, load_jsonl, write_json, write_jsonl

REVS = {
    "allenai/Dolci-Instruct-DPO": "aed155cf32e809b590490b6c3577ee4b0d0a5019",
    "allenai/tulu-3-sft-olmo-2-mixture-0225": "d91a0785ade02942520280fb484866fce41e448f",
    "allenai/WildChat-1M": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
    "jonathanli/legal-advice-reddit": "f105b9d763743e20d2f3b8e33f73055ad414e7c5",
    "lavita/ChatDoctor-HealthCareMagic-100k": "505443eac4e99ccedeffbb6f640061223d1d4bb3",
}
VAL_SEED = 1
SHINGLE_N = 13


def dl(repo, f):
    return hf_hub_download(repo, f, repo_type="dataset", revision=REVS[repo])


# ---- src/data/pools.py:23-33
import hashlib


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def word_shingles(text: str, n: int = SHINGLE_N) -> set[int]:
    words = normalise(text).split()
    return {stable_hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


class MembershipIndex:
    # src/data/pools.py:159-175; shingles held in a sorted uint64 array instead of a python set (same membership test)
    def __init__(self, reference_texts: list[str]):
        normed = [normalise(t) for t in reference_texts]
        self.exact = set(normed)
        parts = [np.fromiter(word_shingles(t), dtype=np.uint64) for t in reference_texts]
        self.shingles = np.unique(np.concatenate([p for p in parts if len(p)]))
        self.joined = " \n ".join(f" {t} " for t in normed)

    def is_member(self, text: str) -> bool:
        norm = normalise(text)
        if norm in self.exact:
            return True
        if len(norm.split()) < SHINGLE_N:
            return f" {norm} " in self.joined
        q = np.fromiter(word_shingles(text), dtype=np.uint64)
        pos = np.minimum(np.searchsorted(self.shingles, q), len(self.shingles) - 1)
        return bool((self.shingles[pos] == q).any())


def load_dolci_first_user() -> pd.DataFrame:
    # src/data/pools.py:40-49
    paths = [dl("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet") for i in range(4)]
    df = pd.concat([pd.read_parquet(p, columns=["chosen", "prompt_id"]) for p in paths], ignore_index=True)
    df["first_user"] = df["chosen"].map(lambda msgs: next(m["content"] for m in msgs if m["role"] == "user"))
    df["n_messages"] = df["chosen"].map(len)
    return df.drop(columns=["chosen"])


def load_tulu_first_user() -> pd.DataFrame:
    # src/data/pools.py:52-67
    repo = "allenai/tulu-3-sft-olmo-2-mixture-0225"
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=REVS[repo]) if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(dl(repo, f), columns=["messages", "source"]) for f in files], ignore_index=True)
    df["first_user"] = df["messages"].map(lambda msgs: next((m["content"] for m in msgs if m["role"] == "user"), None))
    df = df[df["first_user"].notna()].copy()
    df["n_messages"] = df["messages"].map(len)
    return df.drop(columns=["messages"])


def legal_posts(split: str) -> list[str]:
    # src/data/pools.py:122-130 (split file name is the only change)
    rows = [json.loads(line) for line in open(dl("jonathanli/legal-advice-reddit", f"{split}.jsonl"))]
    texts = [f"{r['title'].strip()}\n\n{r['body'].strip()}" for r in rows
             if r["body"] and r["body"].strip() not in ("[removed]", "[deleted]")]
    return list(dict.fromkeys(texts))


def medical_inputs() -> list[str]:
    # src/data/pools.py:134-140
    p = dl("lavita/ChatDoctor-HealthCareMagic-100k", "data/train-00000-of-00001-5e7cb295b9cff0bf.parquet")
    return list(dict.fromkeys(t.strip() for t in pd.read_parquet(p)["input"] if t and t.strip()))


def wildchat_english(shards: list[int]) -> list[str]:
    # src/data/pools.py:144-156
    paths = [dl("allenai/WildChat-1M", f"data/train-{i:05d}-of-00014.parquet") for i in shards]
    df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
    df = df[df["language"] == "English"]
    texts = df["conversation"].map(lambda c: next((m["content"] for m in c if m["role"] == "user"), "")).str.strip()
    texts = texts[texts.str.len() > 0]
    return texts[~texts.map(normalise).duplicated()].tolist()


def main():
    out = INPUTS / "val"
    stats = {"revisions": REVS, "val_seed": VAL_SEED}
    t0 = time.time()
    test_texts = {}
    for p in ("legal", "medical", "wildchat_en_test", "idfit", "idmix"):
        for e in ("repro_two_sided", "repro_olmo2_1b"):
            test_texts[f"{e}/{p}"] = {r["text"] for r in load_jsonl(INPUTS / e / f"{p}.jsonl")}
    bank = {t: {r["text"] for r in load_jsonl(INPUTS / "ranking_variants" / "density_ratio_needles" / f"needles_{t}.jsonl")} for t in ("legal", "medical")}

    # provenance: committed legal / medical rows exist in the public sources
    legal_train, legal_test, legal_val = legal_posts("train"), legal_posts("test"), legal_posts("validation")
    med = medical_inputs()
    stats["provenance"] = {
        "repro_two_sided_legal_in_legal_train": f"{len(test_texts['repro_two_sided/legal'] & set(legal_train))}/500",
        "density_ratio_needles_legal_bank_in_legal_test": f"{len(bank['legal'] & set(legal_test))}/300",
        "repro_two_sided_medical_in_hcm": f"{len(test_texts['repro_two_sided/medical'] & set(med))}/500",
        "density_ratio_needles_medical_bank_in_hcm": f"{len(bank['medical'] & set(med))}/300",
    }
    print(stats["provenance"], flush=True)
    rng = np.random.default_rng(VAL_SEED)
    used_legal = test_texts["repro_two_sided/legal"] | bank["legal"]
    cand = [t for t in legal_val if t not in used_legal and normalise(t) not in {normalise(u) for u in used_legal}]
    val_legal = [cand[i] for i in rng.choice(len(cand), 300, replace=False)]
    used_med = {normalise(t) for t in test_texts["repro_two_sided/medical"] | bank["medical"]}
    cand = [t for t in med if normalise(t) not in used_med]
    val_medical = [cand[i] for i in rng.choice(len(cand), 300, replace=False)]
    stats["val_legal_overlap_with_legal_train_or_test"] = sum(t in set(legal_train) or t in set(legal_test) for t in val_legal)
    write_jsonl([{"id": f"val_legal_{i:05d}", "pool": "val_legal", "text": t} for i, t in enumerate(val_legal)], out / "val_legal.jsonl")
    write_jsonl([{"id": f"val_medical_{i:05d}", "pool": "val_medical", "text": t} for i, t in enumerate(val_medical)], out / "val_medical.jsonl")
    print(f"legal/medical done {time.time()-t0:.0f}s", flush=True)

    wc = wildchat_english([7])
    stats["wildchat_shard7_english_unique_first_turns"] = len(wc)
    order = np.random.default_rng(VAL_SEED).permutation(len(wc))
    test_wc = test_texts["repro_two_sided/wildchat_en_test"] | test_texts["repro_olmo2_1b/wildchat_en_test"]
    for mix_name, loader in (("dolci", load_dolci_first_user), ("tulu", load_tulu_first_user)):
        df = loader()
        print(f"{mix_name} loaded {len(df)} rows {time.time()-t0:.0f}s", flush=True)
        index = MembershipIndex(df["first_user"].tolist())
        print(f"{mix_name} index {len(index.shingles)} shingles {time.time()-t0:.0f}s", flush=True)
        probe = df["first_user"].sample(200, random_state=0).tolist()
        stats[f"{mix_name}_membership_positive_control"] = f"{sum(index.is_member(t) for t in probe)}/200"
        if mix_name == "dolci":
            # provenance: idfit rows are dolci first-user turns
            stats["repro_two_sided_idfit_in_dolci"] = f"{sum(index.is_member(t) for t in list(test_texts['repro_two_sided/idfit'])[:300])}/300 checked"
        else:
            stats["repro_olmo2_1b_idfit_in_tulu"] = f"{sum(index.is_member(t) for t in list(test_texts['repro_olmo2_1b/idfit'])[:300])}/300 checked"
        kept, n_checked, n_member = [], 0, 0
        for i in order:
            t = wc[i]
            if t in test_wc:
                continue
            n_checked += 1
            if index.is_member(t):
                n_member += 1
                continue
            kept.append(t)
            if len(kept) == 500:
                break
        stats[f"val_wildchat_{mix_name}"] = {"checked": n_checked, "members_removed": n_member, "kept": len(kept)}
        write_jsonl([{"id": f"val_wildchat_{mix_name}_{i:05d}", "pool": f"val_wildchat_{mix_name}", "text": t}
                     for i, t in enumerate(kept)], out / f"val_wildchat_{mix_name}.jsonl")
        print(mix_name, stats[f"val_wildchat_{mix_name}"], stats[f"{mix_name}_membership_positive_control"], f"{time.time()-t0:.0f}s", flush=True)
        del index, df
    write_json(stats, out / "build_stats.json")


if __name__ == "__main__":
    main()
