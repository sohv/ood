# exp37: builds the ~1M-prompt traffic pool, ratios, clusters, the gap and random arms, planted check, blind names, prompts and matched controls.
# uv run -m scripts.run_gap_pipeline_build --run_dir results/raw/261021_exp37_gap_pipeline_v1 --output_dir data/processed/exp37 --exp09_dir results/raw/261019_exp09_discovery_v2 --exp31_dir data/processed/exp31 --emb_v1 results/raw/261021_exp31_self_nature_cells_v1 --emb_v2 results/raw/261021_exp31_self_nature_cells_v2 --banks_dir data/processed/exp06 --idmix_path data/processed/exp01/idmix.jsonl --seed 0

import asyncio
import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.data.gaps import draw_controls
from src.data.io import load_jsonl
from src.data.needles import lmsys_first_turns
from src.data.pools import load_dolci_dpo, normalise
from src.metrics.detectors import LENGTH_BANDS
from src.metrics.discovery import NAMING_PROMPT_GRAIN, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import density_ratios_batched, spherical_kmeans, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SOURCE = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
PLANTED = ("legal", "medical", "magicoder", "idmix")
NAME_RE = re.compile(r"NAME_\d+")
IDENTITY_RE = re.compile(
    r"\b(who|what) are you\b|\bwhat('?s| is) your name\b|\b(which|what) (ai |language )?model (are you|is this)\b"
    r"|\bare you (chat ?gpt|gpt[- ]?\d|an? ai|made by openai)\b|\bwho (made|created|trained|developed|built) you\b",
    re.IGNORECASE,
)


@dataclass
class GapPipelineConfig:
    run_dir: str = ""
    output_dir: str = ""
    exp09_dir: str = ""
    exp31_dir: str = ""
    emb_v1: str = ""
    emb_v2: str = ""
    banks_dir: str = ""
    idmix_path: str = ""
    sft_path: str = "data/processed/exp31/dolci_sft_first_user.parquet"
    new_lmsys_shards: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    k: int = 50
    n_clusters: int = 2000
    kmeans_iters: int = 25
    min_cluster: int = 50
    n_arm: int = 40
    n_per_cluster: int = 20
    controls_per_row: int = 2
    max_chars: int = 4000
    embed_chars: int = 1000
    n_planted: int = 300
    n_named: int = 25
    num_tasks: int | None = None


def main():
    config = simple_parsing.parse(GapPipelineConfig)
    assert all(
        [config.run_dir, config.output_dir, config.exp09_dir, config.exp31_dir, config.emb_v1, config.emb_v2]
    ), "input and output paths required"
    assert config.banks_dir and config.idmix_path, "planted banks required"
    run, out = Path(config.run_dir), Path(config.output_dir)
    assert not run.exists() and not out.exists(), "run and output directories must be new"
    setup_logging(run)
    out.mkdir(parents=True)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    # traffic: exp09 pool (exp31 traffic embeddings), exp31 extended pool, new lmsys shards; deduplicated in that order
    c9 = json.loads((Path(config.exp09_dir) / "config.json").read_text())
    pool9 = [{"text": r["text"], "src": "deploy"} for r in load_jsonl(c9["wildchat_pool"])]
    pool9 += [{"text": r["text"], "src": "lmsys"} for r in load_jsonl(c9["lmsys_pool"])]
    ext = [{"text": r["text"], "src": r["source"]} for r in load_jsonl(Path(config.exp31_dir) / "traffic_ext.jsonl")]
    e_pool = np.load(Path(config.emb_v2) / "emb_traffic.npy").astype(np.float16)
    e_ext = np.load(Path(config.emb_v2) / "emb_traffic_ext.npy")
    assert len(e_pool) == len(pool9) and len(e_ext) == len(ext), "saved traffic embeddings do not match their pools"
    new = [
        {"text": r["text"], "src": "lmsys"} for r in lmsys_first_turns(config.new_lmsys_shards, os.environ["HF_TOKEN"])
    ]
    old = pool9 + ext
    if config.num_tasks:
        # smoke runs: a slice of each part, so every code path still runs
        old, new = old[: config.num_tasks], new[: config.num_tasks // 4]
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
        [new[i]["text"][: config.embed_chars] for i in keep_new],
        batch_size=512,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float16)
    t_np = np.vstack([np.vstack([e_pool, e_ext])[keep_old], e_new])
    assert len(t_np) == len(rows), f"{len(t_np)} embeddings for {len(rows)} rows"
    source = np.array([SOURCE[r["src"]] for r in rows])
    LOGGER.info(f"traffic pool {len(rows)}: {Counter(source.tolist())}")

    # mix: unique first user messages of dolci sft and dpo, exp31's embeddings
    sft = pd.read_parquet(config.sft_path, columns=["first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    dpo = load_dolci_dpo()
    norms = {"sft": [normalise(t) for t in sft["first_user"]], "dpo": [normalise(t) for t in dpo["first_user"]]}
    mseen, mix_idx = set(), []
    for n in ("sft", "dpo"):
        for i, nm in enumerate(norms[n]):
            if nm not in mseen:
                mseen.add(nm)
                mix_idx.append((n, i))
    m_np = np.concatenate(
        [
            np.load(Path(config.emb_v1) / "emb_sft.npy", mmap_mode="r")[[i for n, i in mix_idx if n == "sft"]],
            np.load(Path(config.emb_v1) / "emb_dpo.npy", mmap_mode="r")[[i for n, i in mix_idx if n == "dpo"]],
        ]
    ).astype(np.float16)
    m_hash = np.array([text_hash(norms[n][i]) for n, i in mix_idx], dtype=np.int64)
    LOGGER.info(f"mix {len(m_np)} unique prompts")
    is_member = np.array([r["norm"] in mseen for r in rows])

    t_emb, m_emb = torch.from_numpy(t_np).cuda(), torch.from_numpy(m_np).cuda()
    t_hash_np = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    t_hash, m_hash_t = torch.from_numpy(t_hash_np).cuda(), torch.from_numpy(m_hash).cuda()
    ratio, near_mix = density_ratios_batched(
        t_np, t_hash_np, np.arange(len(rows)), t_emb, t_hash, m_emb, m_hash_t, k=config.k, return_nearest_mix=True
    )
    pool_median = float(np.median(ratio))
    LOGGER.info(f"ratios done, pool median {pool_median:.3f}")

    labels, _ = spherical_kmeans(t_emb, config.n_clusters, config.kmeans_iters, config.seed)
    sizes = np.bincount(labels, minlength=config.n_clusters)
    c_ratio = np.array([float(np.median(ratio[labels == c])) if sizes[c] else np.nan for c in range(config.n_clusters)])
    eligible = np.where(sizes >= config.min_cluster)[0]
    c_near = np.array(
        [float(np.median(near_mix[labels == c])) if sizes[c] else np.nan for c in range(config.n_clusters)]
    )
    # amendment: ties on the median ratio are broken by lower median similarity to the nearest mix prompt
    order = eligible[np.lexsort((c_near[eligible], -c_ratio[eligible]))]
    rank = {int(c): j + 1 for j, c in enumerate(order)}
    arm_g = order[: config.n_arm]
    rest = order[config.n_arm :]
    arm_r = np.sort(rng.choice(rest, size=config.n_arm, replace=False))
    LOGGER.info(f"{len(eligible)} eligible clusters; G ratios {c_ratio[arm_g].min():.2f}-{c_ratio[arm_g].max():.2f}")

    # q0: planted known answers, ratios computed with the planted rows added to the traffic set
    banks = {t: [r["text"] for r in load_jsonl(Path(config.banks_dir) / f"needles_{t}.jsonl")] for t in PLANTED[:3]}
    banks["idmix"] = [r["text"] for r in load_jsonl(config.idmix_path)]
    p_text, p_type = [], []
    for t in PLANTED:
        pick = rng.choice(len(banks[t]), size=min(config.n_planted, len(banks[t])), replace=False)
        p_text += [banks[t][i] for i in pick]
        p_type += [t] * len(pick)
    p_emb = embedder.encode(
        [t[: config.embed_chars] for t in p_text], batch_size=512, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float16)
    p_hash = np.array([text_hash(normalise(t)) for t in p_text], dtype=np.int64)
    t2 = torch.cat([t_emb, torch.from_numpy(p_emb).cuda()])
    h2 = torch.cat([t_hash, torch.from_numpy(p_hash).cuda()])
    p_ratio = density_ratios_batched(
        p_emb, p_hash, np.arange(len(rows), len(rows) + len(p_text)), t2, h2, m_emb, m_hash_t, k=config.k
    )
    sorted_c = np.sort(c_ratio[eligible])[::-1]
    q0 = {}
    for t in PLANTED:
        med = float(np.median(p_ratio[np.array(p_type) == t]))
        q0[t] = {"median_ratio": med, "would_rank": int((sorted_c > med).sum()) + 1}
    # diagnostic (amendment): planted medians over the pool median, against the full mix and against the dpo mix alone
    dseen, d_idx = set(), []
    for i, nm in enumerate(norms["dpo"]):
        if nm not in dseen:
            dseen.add(nm)
            d_idx.append(i)
    md = torch.from_numpy(np.load(Path(config.emb_v1) / "emb_dpo.npy", mmap_mode="r")[d_idx].astype(np.float16)).cuda()
    mdh = torch.from_numpy(np.array([text_hash(norms["dpo"][i]) for i in d_idx], dtype=np.int64)).cuda()
    samp = rng.choice(len(rows), size=min(20000, len(rows)), replace=False)
    pool_med_dpo = float(
        np.median(density_ratios_batched(t_np[samp], t_hash_np[samp], samp, t_emb, t_hash, md, mdh, k=config.k))
    )
    p_ratio_dpo = density_ratios_batched(
        p_emb, p_hash, np.arange(len(rows), len(rows) + len(p_text)), t2, h2, md, mdh, k=config.k
    )
    for t in PLANTED:
        sel = np.array(p_type) == t
        q0[t]["over_pool_median_full_mix"] = float(np.median(p_ratio[sel]) / pool_median)
        q0[t]["over_pool_median_dpo_mix"] = float(np.median(p_ratio_dpo[sel]) / pool_med_dpo)
    del md, mdh
    del t2, h2
    n_elig = len(eligible)
    gap_ok = all(q0[t]["would_rank"] <= config.n_arm for t in ("legal", "medical"))
    gap_bad = any(q0[t]["would_rank"] > n_elig / 2 for t in ("legal", "medical"))
    cov_ok = all(q0[t]["would_rank"] > n_elig / 2 for t in ("magicoder", "idmix"))
    q0_verdict = "valid" if gap_ok and cov_ok else "invalid" if gap_bad else "ambiguous"
    LOGGER.info(f"q0 {q0_verdict}: {q0}")

    # tokens for length bands, olmo tokenizer on the raw user text
    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    short = np.array([len(r["text"]) <= config.max_chars for r in rows])
    n_tok = np.zeros(len(rows), dtype=int)
    idx_short = np.where(short)[0]
    for s in range(0, len(idx_short), 20000):
        b = idx_short[s : s + 20000]
        n_tok[b] = [len(x) for x in tok([rows[i]["text"] for i in b], add_special_tokens=False)["input_ids"]]
    band = np.digitize(n_tok, LENGTH_BANDS)

    in_arm = np.isin(labels, np.concatenate([arm_g, arm_r]))
    ctrl_ok = ~in_arm & ~is_member & short
    prompts, groups = [], []
    for arm, clusters in (("gap", arm_g), ("random", arm_r)):
        for c in clusters:
            cand = np.where((labels == c) & ~is_member & short)[0]
            take = np.sort(rng.choice(cand, size=min(config.n_per_cluster, len(cand)), replace=False))
            ctrl = draw_controls(take, source, band, ctrl_ok, config.controls_per_row, rng)
            gid = f"{arm}_c{int(c)}"
            for j, r in enumerate(take):
                prompts.append(
                    {
                        "id": f"{gid}_p{j}",
                        "text": rows[r]["text"],
                        "role": "member",
                        "group": gid,
                        "arm": arm,
                        "source": source[r],
                        "row": int(r),
                    }
                )
                for q, cr in enumerate(ctrl[j]):
                    prompts.append(
                        {
                            "id": f"{gid}_p{j}_c{q}",
                            "text": rows[cr]["text"],
                            "role": "control",
                            "group": gid,
                            "arm": arm,
                            "source": source[cr],
                            "row": int(cr),
                        }
                    )
            texts = [rows[i]["text"] for i in take]
            groups.append(
                {
                    "group": gid,
                    "arm": arm,
                    "cluster": int(c),
                    "size": int(sizes[c]),
                    "ratio": c_ratio[c],
                    "nearest_mix_median": c_near[c],
                    "ratio_over_pool_median": c_ratio[c] / pool_median,
                    "rank": rank[int(c)],
                    "n_sampled": len(take),
                    "wildchat_share": float(np.mean(source[labels == c] == "wildchat")),
                    "name_share": float(np.mean([bool(NAME_RE.search(t)) for t in texts])),
                    "identity_share": float(np.mean([bool(IDENTITY_RE.search(t)) for t in texts])),
                }
            )
    assert len({p["id"] for p in prompts}) == len(prompts), "duplicate prompt ids"
    LOGGER.info(f"{len(prompts)} prompts in {len(groups)} groups")

    # blind names: groups shuffled so the namer's call order says nothing about the arm
    order_g = rng.permutation(len(groups))
    name_prompts = []
    for j in order_g:
        members = np.where(labels == groups[j]["cluster"])[0]
        name_prompts.append(
            naming_prompt([rows[i]["text"] for i in members], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN)
        )
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(name_clusters(client, config.namer_model, name_prompts, str(run / "naming_cache")))
    for j, rep in zip(order_g, replies):
        groups[j]["naming"] = parse_naming(rep)

    write_jsonl(prompts, out / "prompts.jsonl")
    write_json(groups, out / "groups.json")
    write_json(
        {
            "pool_size": len(rows),
            "pool_by_source": Counter(source.tolist()),
            "mix_size": len(m_np),
            "members_in_pool": int(is_member.sum()),
            "pool_median_ratio": pool_median,
            "eligible_clusters": n_elig,
        },
        out / "build_stats.json",
    )
    write_json(
        {
            "verdict": q0_verdict,
            "planted": q0,
            "cluster_ratio_at_rank_40": float(sorted_c[config.n_arm - 1]),
            "cluster_ratio_median": float(np.median(sorted_c)),
        },
        run / "q0_planted.json",
    )
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "size": int(sizes[c]),
                "ratio": c_ratio[c],
                "rank": rank.get(int(c)),
                "arm": "gap" if c in arm_g else "random" if c in arm_r else None,
            }
            for c in range(config.n_clusters)
        ],
        run / "clusters.jsonl",
    )
    np.save(run / "labels.npy", labels)
    np.save(run / "ratio.npy", ratio.astype(np.float32))
    print(f"q0 {q0_verdict}: {q0}")
    print(f"prompts written to {out / 'prompts.jsonl'} ({len(prompts)})")
    print(f"next: rollouts with scripts.run_rollouts --pool_path {out / 'prompts.jsonl'} --n_rollouts 2")


if __name__ == "__main__":
    main()
