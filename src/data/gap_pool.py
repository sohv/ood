# exp37+ traffic pool (exp09's pool, exp31's extended pool, new lmsys shards, deduplicated) and the dolci mix, with minilm embeddings.

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data.io import load_jsonl
from src.data.needles import lmsys_first_turns
from src.data.pools import load_dolci_dpo, normalise
from src.metrics.gap_pipeline import text_hash

LOGGER = logging.getLogger(__name__)
SOURCE = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}


def load_traffic_pool(
    exp09_dir: str,
    exp31_dir: str,
    emb_v2: str,
    new_lmsys_shards: list[int],
    embed_chars: int,
    num_tasks: int | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray, SentenceTransformer]:
    # rows (text, src, norm), float16 unit embeddings, source labels and the embedder, deduplicated in the order exp37 used
    c9 = json.loads((Path(exp09_dir) / "config.json").read_text())
    pool9 = [{"text": r["text"], "src": "deploy"} for r in load_jsonl(c9["wildchat_pool"])]
    pool9 += [{"text": r["text"], "src": "lmsys"} for r in load_jsonl(c9["lmsys_pool"])]
    ext = [{"text": r["text"], "src": r["source"]} for r in load_jsonl(Path(exp31_dir) / "traffic_ext.jsonl")]
    e_pool = np.load(Path(emb_v2) / "emb_traffic.npy").astype(np.float16)
    e_ext = np.load(Path(emb_v2) / "emb_traffic_ext.npy")
    assert len(e_pool) == len(pool9) and len(e_ext) == len(ext), "saved traffic embeddings do not match their pools"
    new = [{"text": r["text"], "src": "lmsys"} for r in lmsys_first_turns(new_lmsys_shards, os.environ["HF_TOKEN"])]
    old = pool9 + ext
    if num_tasks:
        # smoke runs: a slice of each part, so every code path still runs
        old, new = old[:num_tasks], new[: num_tasks // 4]
    LOGGER.info(f"pool sizes before dedup: exp09 {len(pool9)}, exp31 ext {len(ext)}, new lmsys {len(new)}")
    seen, rows, keep_old, keep_new = set(), [], [], []
    for part, store in ((old, keep_old), (new, keep_new)):
        for i, r in enumerate(part):
            nm = normalise(r["text"])
            if nm and nm not in seen:
                seen.add(nm)
                rows.append({**r, "norm": nm})
                store.append(i)
    embedder = SentenceTransformer(c9["embed_model"], device="cuda", revision=c9["embed_model_revision"])
    e_new = embedder.encode(
        [new[i]["text"][:embed_chars] for i in keep_new],
        batch_size=512,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float16)
    t_np = np.vstack([np.vstack([e_pool, e_ext])[keep_old], e_new])
    assert len(t_np) == len(rows), f"{len(t_np)} embeddings for {len(rows)} rows"
    source = np.array([SOURCE[r["src"]] for r in rows])
    return rows, t_np, source, embedder


def load_mix(sft_path: str, emb_v1: str) -> tuple[np.ndarray, np.ndarray]:
    # unique first user turns of dolci sft then dpo, in exp37's order: float16 embeddings and text hashes
    sft = pd.read_parquet(sft_path, columns=["first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    norms = {
        "sft": [normalise(t) for t in sft["first_user"]],
        "dpo": [normalise(t) for t in load_dolci_dpo()["first_user"]],
    }
    seen, mix_idx = set(), []
    for n in ("sft", "dpo"):
        for i, nm in enumerate(norms[n]):
            if nm not in seen:
                seen.add(nm)
                mix_idx.append((n, i))
    m_np = np.concatenate(
        [
            np.load(Path(emb_v1) / "emb_sft.npy", mmap_mode="r")[[i for n, i in mix_idx if n == "sft"]],
            np.load(Path(emb_v1) / "emb_dpo.npy", mmap_mode="r")[[i for n, i in mix_idx if n == "dpo"]],
        ]
    ).astype(np.float16)
    return m_np, np.array([text_hash(norms[n][i]) for n, i in mix_idx], dtype=np.int64)
