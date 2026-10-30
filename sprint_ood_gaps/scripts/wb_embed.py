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
    "repro_two_sided/idfit", "repro_two_sided/idmix", "repro_two_sided/legal", "repro_two_sided/medical", "repro_two_sided/wildchat_en_test", "repro_two_sided/dolma",
    "repro_olmo2_1b/idfit", "repro_olmo2_1b/idmix", "repro_olmo2_1b/wildchat_en_test",
    "ranking_variants/density_ratio_needles/deploy", "ranking_variants/iterative_discovery/lmsys", "ranking_variants/density_ratio_needles/needles_legal", "ranking_variants/density_ratio_needles/needles_medical",
    "ranking_variants/density_ratio_needles/needles_aya", "ranking_variants/density_ratio_needles/needles_magicoder",
    *[f"register_vs_coverage/{p}" for p in ("legal_rtb", "reddit_askscience", "reddit_cooking", "reddit_fitness", "reddit_learnprogramming",
                              "reddit_relationship_advice", "reddit_relationships", "reddit_writing")],
    *[f"legal_medical_restyled/{p}" for p in ("legal_rw", "medical_rw", "wildchat_rw", "idmix_rw")],
    "val/val_legal", "val/val_medical", "val/val_wildchat_dolci", "val/val_wildchat_tulu",
]  # repro_olmo2_1b legal / medical / dolma are byte-identical copies of repro_two_sided's (SHA256SUMS), so they reuse repro_two_sided's cache


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
