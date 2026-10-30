# builds the repro_controls control pools: dolma seed b, membership halves, same-domain-other-source pools and deployment traffic.
# uv run --no-sync -m scripts.run_build_controls --repro_two_sided_pools data/processed/repro_two_sided --output_dir data/processed/repro_controls --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.controls import (
    dedupe,
    load_aya_inputs,
    load_bactrian,
    load_evol_codealpaca,
    load_first_turns,
    load_magicoder,
    load_wildjailbreak,
    take_non_members,
)
from src.data.io import load_jsonl
from src.data.pools import MembershipIndex, load_dolci_dpo, normalise, sample_dolma
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

FAMILIES = {
    "aya": "tulu_v3.9_aya_100k-request",
    "code": "evol_codealpaca_heval_decontaminated-request",
    "jailbreak": "tulu_v3.9_wildjailbreak_decontaminated_50k-request",
}
LMSYS_SHARD0 = "data/train-00000-of-00006-4feeb3f83346a0e9.parquet"


@dataclass
class ControlsConfig:
    repro_two_sided_pools: str = ""
    output_dir: str = ""
    seed: int = 0
    dolma_seed: int = 1
    n_pool: int = 500
    n_dolma: int = 3000
    n_deploy: int = 2000


def write_pool(out: Path, name: str, texts: list[str], **extra: list) -> None:
    rows = [
        {"id": f"{name}_{i:05d}", "pool": name, "text": t, **{k: v[i] for k, v in extra.items()}}
        for i, t in enumerate(texts)
    ]
    write_jsonl(rows, out / f"{name}.jsonl")
    LOGGER.info(f"{name}: {len(rows)} rows")


def main():
    config = simple_parsing.parse(ControlsConfig)
    assert config.repro_two_sided_pools and config.output_dir, "repro_two_sided_pools and output_dir are required"
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, out / "config.json")
    stats = {}

    dolci = load_dolci_dpo()
    index = MembershipIndex(dolci["first_user"].tolist())
    repro_two_sided_ids = {
        normalise(r["text"])
        for n in ("idfit", "idmix")
        for r in load_jsonl(Path(config.repro_two_sided_pools) / f"{n}.jsonl")
    }
    rng = np.random.default_rng(config.seed)

    for short, family in FAMILIES.items():
        rows = dolci[(dolci["family"] == family) & (dolci["n_messages"] == 2)]
        texts = [t for t in dedupe(rows["first_user"].tolist()) if normalise(t) not in repro_two_sided_ids]
        train = [texts[i] for i in rng.permutation(len(texts))[: config.n_pool]]
        # positive control for the membership check: literal mix rows must all register as members
        hits = sum(index.is_member(t) for t in train)
        assert hits == len(train), f"membership check missed {len(train) - hits} literal {family} rows"
        write_pool(out, f"{short}_train", train)
        stats[f"{short}_train"] = {
            "family": family,
            "candidates": len(texts),
            "membership_positive_control": f"{hits}/{len(train)}",
        }

    upstream = {
        "aya": load_aya_inputs(),
        "code": load_evol_codealpaca(),
        "jailbreak": load_wildjailbreak(),
        "magicoder": load_magicoder(),
    }
    for short, texts in upstream.items():
        name = short if short == "magicoder" else f"{short}_test"
        kept, checked, removed = take_non_members(texts, index, config.n_pool, config.seed)
        write_pool(out, name, kept)
        stats[name] = {"upstream_rows": len(texts), "checked": checked, "members_removed": removed}

    bactrian, kept, langs, removed = load_bactrian(config.n_pool, config.seed), [], [], 0
    taken: dict[str, int] = {}
    for row in bactrian:
        if taken.get(row["language"], 0) >= row["quota"]:
            continue
        if index.is_member(row["text"]):
            removed += 1
            continue
        taken[row["language"]] = taken.get(row["language"], 0) + 1
        kept.append(row["text"])
        langs.append(row["language"])
    assert len(kept) == config.n_pool, f"bactrian short: {len(kept)}"
    write_pool(out, "bactrian", kept, language=langs)
    stats["bactrian"] = {"languages": len(taken), "members_removed": removed}

    for name, repo, filename in (
        ("wildchat_deploy", "allenai/WildChat-1M", "data/train-00001-of-00014.parquet"),
        ("lmsys", "lmsys/lmsys-chat-1m", LMSYS_SHARD0),
    ):
        df = load_first_turns(repo, filename)
        n_unique = len(df)
        df = df.iloc[np.random.default_rng(config.seed).permutation(n_unique)[: config.n_deploy]]
        member = [index.is_member(t) for t in df["text"]]
        write_pool(out, name, df["text"].tolist(), language=df["language"].tolist(), is_member=member)
        stats[name] = {
            "unique_first_turns": n_unique,
            "members": int(sum(member)),
            "english_share": float((df["language"] == "English").mean()),
        }

    repro_two_sided_dolma = {
        normalise(r["text"]) for r in load_jsonl(Path(config.repro_two_sided_pools) / "dolma.jsonl")
    }
    dolma = [
        d
        for d in sample_dolma(int(config.n_dolma * 1.1), config.dolma_seed)
        if normalise(d["text"]) not in repro_two_sided_dolma
    ]
    picks = np.sort(np.random.default_rng(config.dolma_seed).choice(len(dolma), size=config.n_dolma, replace=False))
    write_pool(out, "dolma", [dolma[i]["text"] for i in picks], source=[dolma[i]["source"] for i in picks])
    stats["dolma_b"] = {"seed": config.dolma_seed, "after_overlap_drop": len(dolma)}

    for n in ("idfit",):
        write_jsonl(load_jsonl(Path(config.repro_two_sided_pools) / f"{n}.jsonl"), out / f"{n}.jsonl")
    write_json(stats, out / "build_stats.json")
    print(f"pools written to {out}")


if __name__ == "__main__":
    main()
