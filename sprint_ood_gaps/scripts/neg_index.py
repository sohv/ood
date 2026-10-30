"""Rebuild pipeline_million's traffic and mix pools (texts, hashes, MiniLM embeddings) from Sohan's saved files, on CPU.

Mirrors ../scripts/run_gap_pipeline_build.py and src/data/gap_pool.py step for step:
  traffic = iterative_discovery pool (20k, emb_traffic.npy) + self_nature_a extended pool (traffic_ext.jsonl, emb_traffic_ext.npy)
            + LMSYS shards 2-5 first turns (re-embedded here), deduplicated in that order on normalise(text)
  mix     = unique normalise(first_user) over Dolci SFT parquet then Dolci DPO, embeddings from emb_sft.npy / emb_dpo.npy
Outputs under runs/negatives/index/: traffic.npy (float16), traffic_hash.npy, traffic_rows.jsonl, mix.npy, mix_hash.npy, mix_meta.parquet.
Run: .venv/bin/python -I scripts/neg_index.py [--skip_new_lmsys]  (the re-embedding is the slow part, ~3 h CPU)
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "external/hf/detectood_ood"
OUT = ROOT / "runs/negatives/index"
OUT.mkdir(parents=True, exist_ok=True)
EXP09 = HF / "results/raw/261019_iterative_discovery_v2"
EXP31 = HF / "data/processed/self_nature_a"
EMB_V1 = HF / "results/raw/261021_self_nature_a_self_nature_cells_v1"
EMB_V2 = HF / "results/raw/261021_self_nature_a_self_nature_cells_v2"
NEW_LMSYS_SHARDS = [2, 3, 4, 5]
EMBED_CHARS = 1000


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(norm: str) -> int:
    return int.from_bytes(hashlib.blake2b(norm.encode(), digest_size=8).digest(), "little") >> 1


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_mix():
    sft = pd.read_parquet(EXP31 / "dolci_sft_first_user.parquet", columns=["first_user"])
    n_sft_total = len(sft)
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    log("sft rows", n_sft_total, "non-null", len(sft))
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
    dpo = pd.concat([pd.read_parquet(p, columns=["chosen", "prompt_id"]) for p in paths], ignore_index=True)
    dpo["first_user"] = dpo["chosen"].map(lambda msgs: next(m["content"] for m in msgs if m["role"] == "user"))
    log("dpo rows", len(dpo))
    e_sft = np.load(EMB_V1 / "emb_sft.npy", mmap_mode="r")
    e_dpo = np.load(EMB_V1 / "emb_dpo.npy", mmap_mode="r")
    # his build indexes emb_sft by the position in the non-null-filtered frame, so the parquet must already be aligned to it
    assert len(e_sft) in (len(sft), n_sft_total), (len(e_sft), len(sft), n_sft_total)
    assert len(e_dpo) == len(dpo), (len(e_dpo), len(dpo))
    sft_offset_filtered = len(e_sft) == len(sft)
    norms = {"sft": [normalise(t) for t in sft["first_user"]], "dpo": [normalise(t) for t in dpo["first_user"]]}
    mseen, mix_idx = set(), []
    for n in ("sft", "dpo"):
        for i, nm in enumerate(norms[n]):
            if nm not in mseen:
                mseen.add(nm)
                mix_idx.append((n, i))
    log("mix unique", len(mix_idx), "(pipeline_million logged 1,710,578)")
    sft_i = [i for n, i in mix_idx if n == "sft"]
    dpo_i = [i for n, i in mix_idx if n == "dpo"]
    if not sft_offset_filtered:
        # map filtered positions back to original parquet rows
        orig = np.where(pd.read_parquet(EXP31 / "dolci_sft_first_user.parquet", columns=["first_user"])["first_user"].notna())[0]
        sft_i = orig[sft_i].tolist()
    m = np.concatenate([np.asarray(e_sft[sft_i]), np.asarray(e_dpo[dpo_i])]).astype(np.float16)
    m_hash = np.array([text_hash(norms[n][i]) for n, i in mix_idx], dtype=np.int64)
    meta = pd.DataFrame({"part": [n for n, _ in mix_idx], "row": [i for _, i in mix_idx]})
    meta["prompt_id"] = [dpo["prompt_id"].iat[i] if n == "dpo" else None for n, i in mix_idx]
    np.save(OUT / "mix.npy", m); np.save(OUT / "mix_hash.npy", m_hash); meta.to_parquet(OUT / "mix_meta.parquet")
    log("mix saved", m.shape, "norm check (first 3 rows, should be ~1):", np.linalg.norm(m[:3].astype(np.float32), axis=1))
    return mseen


def build_traffic(skip_new: bool):
    c9 = json.loads((EXP09 / "config.json").read_text())
    wc_path, lm_path = c9["wildchat_pool"], c9["lmsys_pool"]
    # his config paths are relative to his repo; the files were downloaded into the same layout under HF
    # these two pool files are committed in this repo (data/), not on HF
    CLONE = ROOT.parent
    pool9 = [{"text": r["text"], "src": "deploy"} for r in load_jsonl(CLONE / wc_path)]
    pool9 += [{"text": r["text"], "src": "lmsys"} for r in load_jsonl(CLONE / lm_path)]
    ext = [{"text": r["text"], "src": r["source"]} for r in load_jsonl(EXP31 / "traffic_ext.jsonl")]
    e_pool = np.load(EMB_V2 / "emb_traffic.npy").astype(np.float16)
    e_ext = np.load(EMB_V2 / "emb_traffic_ext.npy")
    assert len(e_pool) == len(pool9) and len(e_ext) == len(ext), (len(e_pool), len(pool9), len(e_ext), len(ext))
    new = []
    if not skip_new:
        from huggingface_hub import HfApi, hf_hub_download
        files = sorted(f for f in HfApi().list_repo_files("lmsys/lmsys-chat-1m", repo_type="dataset") if f.endswith(".parquet"))
        paths = [hf_hub_download("lmsys/lmsys-chat-1m", files[i], repo_type="dataset") for i in NEW_LMSYS_SHARDS]
        df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
        df["text"] = df["conversation"].map(lambda c: next((m["content"] for m in c if m["role"] == "user"), "")).str.strip()
        df = df[df["text"].str.len() > 0]
        df = df[~df["text"].map(normalise).duplicated()]
        new = [{"text": t, "src": "lmsys"} for t in df["text"]]
    old = pool9 + ext
    log(f"pool sizes before dedup: iterative_discovery {len(pool9)}, self_nature_a ext {len(ext)}, new lmsys {len(new)}")
    seen, rows, keep_old, keep_new = set(), [], [], []
    for part, store in ((old, keep_old), (new, keep_new)):
        for i, r in enumerate(part):
            nm = normalise(r["text"])
            if nm and nm not in seen:
                seen.add(nm)
                rows.append({**r, "norm": nm})
                store.append(i)
    log("traffic unique", len(rows), "(pipeline_million logged 992,933)")
    e_new = np.zeros((0, 384), dtype=np.float16)
    if keep_new:
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(os.cpu_count())
        embedder = SentenceTransformer(c9["embed_model"], device="cpu", revision=c9.get("embed_model_revision"))
        texts = [new[i]["text"][:EMBED_CHARS] for i in keep_new]
        log("embedding", len(texts), "new lmsys prompts on cpu")
        parts, t0 = [], time.time()
        for s in range(0, len(texts), 20000):
            parts.append(embedder.encode(texts[s:s + 20000], batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16))
            np.save(OUT / "traffic_new_partial.npy", np.vstack(parts))
            log(f"  {s + len(parts[-1])}/{len(texts)} done, {(time.time() - t0) / 60:.1f} min")
        e_new = np.vstack(parts)
    t = np.vstack([np.vstack([e_pool, e_ext])[keep_old], e_new]) if keep_new else np.vstack([e_pool, e_ext])[keep_old]
    assert len(t) == len(rows)
    np.save(OUT / "traffic.npy", t.astype(np.float16))
    np.save(OUT / "traffic_hash.npy", np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64))
    with open(OUT / "traffic_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "src": r["src"]}, ensure_ascii=False) + "\n")
    log("traffic saved", t.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip_new_lmsys", action="store_true"); ap.add_argument("--only", choices=["mix", "traffic"], default=None)
    a = ap.parse_args()
    if a.only in (None, "mix"): build_mix()
    if a.only in (None, "traffic"): build_traffic(a.skip_new_lmsys)
    log("done")
