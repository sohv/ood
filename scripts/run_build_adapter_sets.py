# exp34: draws the identity (uncovered self-nature, double-labelled) and matched control training prompts, outside every exp31 cell, for the two adapters.
# uv run -m scripts.run_build_adapter_sets --output_dir data/processed/exp34 --exp31_dir data/processed/exp31 --cells_path results/raw/261021_exp31_self_nature_cells_v2/cells.jsonl --emb_v1 results/raw/261021_exp31_self_nature_cells_v1 --emb_v2 results/raw/261021_exp31_self_nature_cells_v2 --exp09_dir results/raw/261019_exp09_discovery_v2 --model_id openai/gpt-4.1 --seed 0

import asyncio
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from scripts.run_build_self_nature import BANDS, EXT_POOL, LABEL_PROMPT, TRAFFIC_SETS, band_of, load_pool
from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo, normalise
from src.metrics.coverage import density_ratios
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class AdapterSetsConfig:
    output_dir: str = ""
    exp31_dir: str = ""
    cells_path: str = ""
    emb_v1: str = ""
    emb_v2: str = ""
    exp09_dir: str = ""
    sft_path: str = "data/processed/exp31/dolci_sft_first_user.parquet"
    model_id: str = "openai/gpt-4.1"
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    n_train: int = 200
    uncovered_ratio: float = 5.0
    k: int = 50


def main():
    config = simple_parsing.parse(AdapterSetsConfig)
    assert all(
        [config.output_dir, config.exp31_dir, config.cells_path, config.emb_v1, config.emb_v2, config.exp09_dir]
    ), "all paths required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    labelled = load_jsonl(Path(config.exp31_dir) / "labelled.jsonl")
    cell_ids = {r["id"] for r in load_jsonl(config.cells_path)}
    pool = load_pool(config.exp09_dir)
    gaps = load_jsonl(Path(config.exp31_dir) / "exp20_gaps.jsonl")
    gap_idx = {i for g in gaps for i in g["members"]}

    # identity candidates: traffic self-nature prompts outside every cell, with the coverage ratio recomputed as in exp31
    cands = [r for r in labelled if r["set"] in TRAFFIC_SETS and r["label"] == "YES" and r["id"] not in cell_ids]
    emb = {
        "traffic": np.load(Path(config.emb_v2) / "emb_traffic.npy").astype(np.float32),
        "traffic_ext": np.load(Path(config.emb_v2) / "emb_traffic_ext.npy").astype(np.float32),
    }
    sft = pd.read_parquet(config.sft_path, columns=["first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    dpo = load_dolci_dpo()
    norms = {"sft": [normalise(t) for t in sft["first_user"]], "dpo": [normalise(t) for t in dpo["first_user"]]}
    seen, mix_idx = set(), []
    for n in ("sft", "dpo"):
        for i, nm in enumerate(norms[n]):
            if nm not in seen:
                seen.add(nm)
                mix_idx.append((n, i))
    m_emb = np.concatenate(
        [
            np.load(Path(config.emb_v1) / "emb_sft.npy").astype(np.float32)[[i for n, i in mix_idx if n == "sft"]],
            np.load(Path(config.emb_v1) / "emb_dpo.npy").astype(np.float32)[[i for n, i in mix_idx if n == "dpo"]],
        ]
    )
    m_norm = [norms[n][i] for n, i in mix_idx]
    t_norm = [normalise(r["text"]) for r in pool]
    q_emb = np.stack([emb[r["set"]][r["index"]] for r in cands])
    ratio = density_ratios(
        q_emb,
        [normalise(r["text"]) for r in cands],
        [r["index"] if r["set"] == "traffic" else -1 for r in cands],
        emb["traffic"],
        t_norm,
        m_emb,
        m_norm,
        k=config.k,
    )
    del m_emb
    for r, a in zip(cands, ratio):
        r["ratio"] = round(float(a), 3)
    cands = [r for r in cands if r["ratio"] >= config.uncovered_ratio]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [LABEL_PROMPT.format(message=r["text"][:2000]) for r in cands],
            str(out / "label_cache"),
            max_concurrent=16,
        )
    )
    for r, x in zip(cands, replies):
        r["label_2"] = parse_label(x, ["YES", "NO"])
    double = [r for r in cands if r["label_2"] == "YES"]
    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    for r in double:
        r["band"] = band_of(len(tok(r["text"], add_special_tokens=False)["input_ids"]))
        r["pool"] = EXT_POOL.get(r["source"], r["source"])
    pick = sorted(rng.choice(len(double), size=min(config.n_train, len(double)), replace=False))
    identity = [double[i] for i in pick]

    # controls: ordinary pool prompts, never a candidate, in no cell and no gap, matched on pool and band
    cand_idx = {r["index"] for r in labelled if r["set"] == "traffic"}
    cell_idx = {r["index"] for r in load_jsonl(config.cells_path) if r["set"] == "traffic"}
    ordinary = [i for i in range(len(pool)) if i not in cand_idx and i not in cell_idx and i not in gap_idx]
    buckets: dict[tuple[str, int], list[int]] = {}
    for i in ordinary:
        buckets.setdefault(
            (pool[i]["pool"], band_of(len(tok(pool[i]["text"], add_special_tokens=False)["input_ids"]))), []
        ).append(i)
    for v in buckets.values():
        rng.shuffle(v)
    control, fallback = [], 0
    for r in identity:
        b = buckets.get((r["pool"], r["band"]), [])
        if not b:
            fallback += 1
            b = next((v for (p, band), v in buckets.items() if band == r["band"] and v), None)
        assert b, f"no control left for band {r['band']}"
        i = b.pop()
        control.append({"index": i, "pool": pool[i]["pool"], "band": r["band"], "text": pool[i]["text"]})
    rows = [
        {
            "id": f"identity_{j:03d}",
            "role": "identity",
            "text": r["text"],
            "pool": r["pool"],
            "band": r["band"],
            "ratio": r["ratio"],
            "source_row": r["id"],
        }
        for j, r in enumerate(identity)
    ]
    rows += [
        {
            "id": f"control_{j:03d}",
            "role": "control",
            "text": r["text"],
            "pool": r["pool"],
            "band": r["band"],
            "source_row": f"traffic_{r['index']}",
        }
        for j, r in enumerate(control)
    ]
    write_jsonl(rows, out / "gen_pool.jsonl")
    counts = {
        "identity_candidates_outside_cells": len(
            [r for r in labelled if r["set"] in TRAFFIC_SETS and r["label"] == "YES" and r["id"] not in cell_ids]
        ),
        "identity_uncovered": len(cands),
        "identity_double_yes": len(double),
        "identity_drawn": len(identity),
        "control_drawn": len(control),
        "control_band_only_fallbacks": fallback,
        "pool_shares": {
            role: dict(Counter(r["pool"] for r in rows if r["role"] == role)) for role in ("identity", "control")
        },
        "bands": {
            role: dict(sorted(Counter(r["band"] for r in rows if r["role"] == role).items()))
            for role in ("identity", "control")
        },
    }
    write_json({"git_hash": get_git_hash(), **asdict(config), "counts": counts}, out / "build.json")
    print(f"{counts=}")
    print(f"generation pool written to {out / 'gen_pool.jsonl'}")


if __name__ == "__main__":
    main()
