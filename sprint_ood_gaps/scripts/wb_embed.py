# sentence embeddings (CPU) for every pool, cached as npz with ids. Dolma docs are cut to their first 400 words,
# the same cut Sohan's tf-idf baseline uses (scripts/run_detectors.py:116, dolma_tfidf_words = 400).
# usage: .venv/bin/python -m scripts.wb_embed [--limit 200]

import argparse
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from scripts.wb_common import EMBEDDERS, INPUTS, RUN, load_jsonl

POOLS = [
    "exp01/idfit", "exp01/idmix", "exp01/legal", "exp01/medical", "exp01/wildchat_en_test", "exp01/dolma",
    "exp12/idfit", "exp12/idmix", "exp12/wildchat_en_test",
    "exp11/exp06/deploy", "exp11/exp09/lmsys", "exp11/exp06/needles_legal", "exp11/exp06/needles_medical",
    "exp11/exp06/needles_aya", "exp11/exp06/needles_magicoder",
    *[f"exp03/{p}" for p in ("legal_rtb", "reddit_askscience", "reddit_cooking", "reddit_fitness", "reddit_learnprogramming",
                              "reddit_relationship_advice", "reddit_relationships", "reddit_writing")],
    *[f"exp05/{p}" for p in ("legal_rw", "medical_rw", "wildchat_rw", "idmix_rw")],
    "val/val_legal", "val/val_medical", "val/val_wildchat_dolci", "val/val_wildchat_tulu",
]  # exp12 legal / medical / dolma are byte-identical copies of exp01's (SHA256SUMS), so they reuse exp01's cache


def pool_texts(pool: str, limit: int | None) -> tuple[list[str], list[str]]:
    rows = load_jsonl(INPUTS / f"{pool}.jsonl")[:limit] if limit else load_jsonl(INPUTS / f"{pool}.jsonl")
    texts = [" ".join(r["text"].split()[:400]) if pool.endswith("dolma") else r["text"] for r in rows]
    return [r["id"] for r in rows], texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    torch.set_num_threads(8)
    for name, (model_id, rev) in EMBEDDERS.items():
        model = SentenceTransformer(model_id, revision=rev, device="cpu")
        for pool in POOLS:
            out = RUN / "cache" / f"emb_{args.tag}" / name / f"{pool.replace('/', '__')}.npz"
            if out.exists():
                continue
            if not (INPUTS / f"{pool}.jsonl").exists():
                print(f"skip {pool}: not built yet", flush=True)
                continue
            t0 = time.time()
            ids, texts = pool_texts(pool, args.limit)
            emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, ids=np.array(ids), emb=emb.astype(np.float32), model_id=model_id, revision=rev,
                     max_seq_length=model.max_seq_length)
            print(f"{name} {pool}: {len(ids)} rows, {time.time() - t0:.1f}s, max_seq_length {model.max_seq_length}", flush=True)


if __name__ == "__main__":
    main()
