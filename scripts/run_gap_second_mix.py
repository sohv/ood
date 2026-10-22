# exp48: exp41's gap loop and exp44's planted test against a second model's post-training data (OLMo-2 SFT + DPO).
# uv run -m scripts.run_gap_second_mix --run_dir results/raw/261022_exp48_olmo2_gaps_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp41_dir results/raw/261022_exp41_gap_loop_v1 --emb_cache cache/olmo2_mix_minilm.npy --seed 0

import asyncio
import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.data.pools import load_olmo2_dpo, load_tulu_sft, normalise
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class SecondMixConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp41_dir: str = ""
    emb_cache: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    k: int = 50
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    same_gap: float = 0.85
    theme_cos: float = 0.70
    template_chars: int = 60
    template_share: float = 0.3
    n_planted: int = 10
    min_size: int = 200
    max_size: int = 3000
    ratio_band: tuple[float, float] = (0.5, 2.0)
    min_valid: int = 8
    works: int = 6
    fails: int = 2
    attempts: int = 6


def main():
    config = simple_parsing.parse(SecondMixConfig)
    assert config.run_dir and config.exp37_dir and config.exp41_dir and config.emb_cache, "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    shutil.copytree(Path(config.exp41_dir) / "naming_cache", run / "naming_cache")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, embedder = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)

    # olmo-2 mix: unique first user turns of the sft mix then the dpo mix
    seen, m_text = set(), []
    for t in load_tulu_sft()["first_user"].tolist() + load_olmo2_dpo()["first_user"].tolist():
        nm = normalise(t)
        if nm not in seen:
            seen.add(nm)
            m_text.append(t)
    m_hash = np.array([text_hash(normalise(t)) for t in m_text], dtype=np.int64)
    cache = Path(config.emb_cache)
    if cache.exists():
        m_np = np.load(cache)
        assert len(m_np) == len(m_text), "cached olmo-2 embeddings do not match the mix"
    else:
        m_np = embedder.encode(
            [t[: c37["embed_chars"]] for t in m_text],
            batch_size=512,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float16)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, m_np)
    LOGGER.info(f"olmo-2 mix {len(m_text)} unique prompts")

    clusters = {c["id"]: c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}
    emb = t_np.astype(np.float32)
    cent = {c: centroid(emb[members[c]]) for c in order}
    sig: dict[int, str | None] = {}
    for c in order:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None
    t_gpu, th_gpu = torch.from_numpy(t_np).cuda(), torch.from_numpy(t_hash).cuda()

    def score(keep: np.ndarray) -> tuple[dict, dict, float]:
        ratio, near = density_ratios_batched(
            t_np,
            t_hash,
            np.arange(len(rows)),
            t_gpu,
            th_gpu,
            torch.from_numpy(m_np[keep]).cuda(),
            torch.from_numpy(m_hash[keep]).cuda(),
            k=config.k,
            return_nearest_mix=True,
        )
        medians = [
            float(np.median(ratio[rng.choice(len(rows), config.subset_size, replace=False)]))
            for _ in range(config.n_random_subsets)
        ]
        s = {c: float(np.median(ratio[members[c]])) for c in order}
        n = {c: float(np.median(near[members[c]])) for c in order}
        return s, n, config.ratio_multiple * float(np.median(medians))

    full = np.ones(len(m_np), dtype=bool)
    s_full, n_full, line_full = score(full)
    LOGGER.info(f"line {line_full:.2f}; {sum(v >= line_full for v in s_full.values())} topics over it")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    names: dict[int, dict] = {}

    def review(batch: list[int]) -> None:
        todo = [c for c in batch if c not in names]
        prompts = [
            naming_prompt(
                [rows[i]["text"] for i in members[c]], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN
            )
            for c in todo
        ]
        replies = asyncio.run(
            name_clusters(client, config.namer_model, prompts, str(run / "naming_cache"), attempts=config.attempts)
        )
        for c, rep in zip(todo, replies, strict=True):
            names[c] = parse_naming(rep)

    def same(a: int, b: int, cos: float) -> bool:
        return float(cent[a] @ cent[b]) >= cos or (sig[a] is not None and sig[a] == sig[b])

    def loop(s: dict, n: dict, line: float) -> tuple[list[int], list[int], list[dict]]:
        remaining = sorted(order, key=lambda c: (-s[c], n[c]))
        reviewed, conf, per_round = [], [], []
        for rnd in range(1, config.rounds + 1):
            batch = remaining[: config.per_round]
            review(batch)
            shared = [names[c].get("shared") for c in batch]
            new = [
                c
                for c, sh in zip(batch, shared, strict=True)
                if sh is not None and sh >= config.shared_needed and s[c] >= line
            ]
            reviewed += batch
            conf += new
            rest = [c for c in remaining if c not in batch]
            remaining = [c for c in rest if not any(same(c, g, config.same_gap) for g in new)]
            per_round.append({"id": f"round_{rnd}", "reviewed": batch, "confirmed": new})
        return reviewed, conf, per_round

    def distinct(conf: list[int], cos: float) -> int:
        heads: list[int] = []
        for c in conf:
            if not any(same(c, h, cos) for h in heads):
                heads.append(c)
        return len(heads)

    def confirmed_in(batch: list[int], s: dict, line: float) -> list[int]:
        review(batch)
        return [c for c in batch if (names[c].get("shared") or 0) >= config.shared_needed and s[c] >= line]

    loop_rev, loop_conf, rounds = loop(s_full, n_full, line_full)
    budget = config.rounds * config.per_round
    walk_conf = confirmed_in(sorted(order, key=lambda c: (-s_full[c], n_full[c]))[:budget], s_full, line_full)
    rand_conf = confirmed_in(rng.choice(order, size=budget, replace=False).tolist(), s_full, line_full)
    LOGGER.info(f"loop {len(loop_conf)} confirmed, walk {len(walk_conf)}, random {len(rand_conf)}")

    # planted test on the olmo-2 mix
    pool_median = float(np.median(list(s_full.values())))
    lo, hi = (b * pool_median for b in config.ratio_band)
    candidates = [
        c
        for c in order
        if sig[c] is None and config.min_size <= len(members[c]) <= config.max_size and lo <= s_full[c] <= hi
    ]
    planted = sorted(rng.choice(candidates, size=config.n_planted, replace=False).tolist())
    keep = full.copy()
    m_gpu = torch.from_numpy(m_np).cuda()
    for c in planted:
        q = torch.from_numpy(t_np[members[c]]).cuda()
        idx = torch.cat(
            [(q[s : s + 1024] @ m_gpu.T).topk(config.k, dim=1).indices.flatten() for s in range(0, len(q), 1024)]
        )
        keep[np.unique(idx.cpu().numpy())] = False
    del m_gpu
    s_abl, n_abl, line_abl = score(keep)
    valid = sum(s_abl[c] >= line_abl for c in planted) >= config.min_valid
    abl_rev, abl_conf, _ = loop(s_abl, n_abl, line_abl)
    found = [c for c in planted if c in abl_conf]

    d41 = json.loads((Path(config.exp41_dir) / "decision.json").read_text())
    conf41 = {c for r in d41["loop_rounds"] for c in r["confirmed"]}
    n_loop, n_rand = distinct(loop_conf, config.same_gap), distinct(rand_conf, config.same_gap)
    decision = {
        "loop_vs_random": "separates" if n_loop >= 2 * n_rand else "does_not_separate",
        "planted": "finds" if len(found) >= config.works else "fails" if len(found) <= config.fails else "ambiguous",
        "planted_valid": valid,
        "planted_found": len(found),
        "planted_over_line": int(sum(s_abl[c] >= line_abl for c in planted)),
        "distinct_0.85": {"loop": n_loop, "walk": distinct(walk_conf, config.same_gap), "random": n_rand},
        "themes_0.70": {"loop": distinct(loop_conf, config.theme_cos), "random": distinct(rand_conf, config.theme_cos)},
        "confirmed": {"loop": len(loop_conf), "walk": len(walk_conf), "random": len(rand_conf)},
        "line": round(line_full, 4),
        "topics_over_line": int(sum(v >= line_full for v in s_full.values())),
        "mix_unique_prompts": len(m_text),
        "overlap_with_olmo3_loop": len(set(loop_conf) & conf41),
        "olmo2_only": len(set(loop_conf) - conf41),
        "olmo3_only": len(conf41 - set(loop_conf)),
        "mix_rows_deleted": int((~keep).sum()),
        "planted_detail": [
            {
                "cluster": c,
                "name": names.get(c, {}).get("name"),
                "ratio_before": round(s_full[c], 4),
                "ratio_after": round(s_abl[c], 4),
                "position": abl_rev.index(c) + 1 if c in abl_rev else None,
                "confirmed": c in abl_conf,
            }
            for c in planted
        ],
        "rounds": rounds,
        "null_names": sum(names[c].get("shared") is None for c in names),
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "cluster": c,
                "size": len(members[c]),
                "score": round(s_full[c], 4),
                **{k: names[c].get(k) for k in ("name", "description", "shared")},
                "confirmed": c in loop_conf,
                "in_olmo3_loop": c in conf41,
                "examples": [
                    rows[i]["text"][:400] for i in rng.choice(members[c], size=min(10, len(members[c])), replace=False)
                ],
            }
            for c in loop_rev
        ],
        run / "reviewed_topics.jsonl",
    )
    print({k: decision[k] for k in decision if k not in ("rounds", "planted_detail")})
    for p in decision["planted_detail"]:
        print(p)
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
