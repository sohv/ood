# exp14 / exp15 / exp16: checks exp09's cluster, blind-name and coverage gap confirmation on planted prompts whose answer is known.
# uv run --no-sync -m scripts.run_validate_discovery --run_dir results/raw/261019_exp14_validate_v1 --wildchat_pool data/processed/exp06/deploy.jsonl --lmsys_pool data/processed/exp09/lmsys.jsonl --banks_dir data/processed/exp06 --model_id anthropic/claude-sonnet-4.6 --seed 0

import asyncio
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from huggingface_hub import HfApi
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo
from src.metrics.discovery import (
    LABEL_PROMPT,
    NAMING_PROMPT,
    NAMING_PROMPT_GRAIN,
    NAMING_PROMPT_SHARE,
    cluster,
    name_clusters,
    naming_prompt,
    nearest_cosine,
    parse_label,
    parse_naming,
)
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

GAP, COVERED = ("legal", "medical"), ("magicoder", "aya")
EXPECTED = {"legal": "LEGAL", "medical": "MEDICAL", "magicoder": "CODE"}


@dataclass
class ValidateConfig:
    run_dir: str = ""
    wildchat_pool: str = ""
    lmsys_pool: str = ""
    banks_dir: str = ""
    model_id: str = ""
    seed: int = 0
    draw_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    judge: str = "openai/gpt-4.1"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    expected_cut: float = 0.6003
    n_clusters: int = 10
    n_shown: int = 25
    min_size: int = 15
    # yesno = exp14's COHERENT call; share = exp15's count of the largest shared group, coherent if >= half shown; grain = share with exp16's broad-field sentence
    coherence_rule: str = "yesno"
    compare_clusters: str = ""


def auroc_low(pos: np.ndarray, neg: np.ndarray) -> float:
    # probability a positive row has lower coverage than a negative row, ties count half
    diff = neg[None, :] - pos[:, None]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def main():
    config = simple_parsing.parse(ValidateConfig)
    missing = [k for k, v in asdict(config).items() if v == "" and k != "compare_clusters"]
    assert config.coherence_rule in ("yesno", "share", "grain"), config.coherence_rule
    assert not missing, f"required arguments not set: {missing}"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "embed_model_revision": HfApi().model_info(config.embed_model).sha,
        },
        run / "config.json",
    )

    pool = [r["text"] for r in load_jsonl(config.wildchat_pool)] + [r["text"] for r in load_jsonl(config.lmsys_pool)]
    banks = {t: [r["text"] for r in load_jsonl(Path(config.banks_dir) / f"needles_{t}.jsonl")] for t in GAP + COVERED}
    assert len(pool) == 20_000 and all(len(v) == 300 for v in banks.values()), "unexpected pool or bank size"

    embedder = SentenceTransformer(config.embed_model, device="cuda")

    def encode(texts: list[str]) -> np.ndarray:
        return embedder.encode(
            texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    texts = pool + [t for k in banks for t in banks[k]]
    source = np.array(["pool"] * len(pool) + [k for k in banks for _ in banks[k]])
    emb = encode(texts)
    ref = torch.from_numpy(encode(load_dolci_dpo()["first_user"].drop_duplicates().tolist())).cuda()
    coverage = nearest_cosine(emb, ref)
    cut = float(np.percentile(coverage[: len(pool)], 25))
    print(f"{cut=:.4f}")
    assert abs(cut - config.expected_cut) < 0.001, f"coverage cut {cut:.4f} differs from exp09 v2's"
    bank_idx = {k: np.where(source == k)[0] for k in banks}

    clusters = []
    for s in config.draw_seeds:
        rng = np.random.default_rng(s)
        pool_perm = iter(rng.permutation(len(pool)))

        def take(n: int) -> np.ndarray:
            return np.array([next(pool_perm) for _ in range(n)])

        perm = {k: rng.permutation(bank_idx[k]) for k in banks}
        for t in ("legal", "medical", "magicoder"):
            p = perm[t]
            for mix, rows in (
                ("pure", p[:50]),
                ("majority", np.concatenate([p[50:85], take(15)])),
                ("minority", np.concatenate([p[85:100], take(35)])),
                ("random", take(50)),
            ):
                clusters.append({"q": "q1", "seed": s, "type": t, "mix": mix, "rows": rows})
        clusters.append({"q": "q2", "seed": s, "type": "aya", "mix": "pure", "rows": perm["aya"][:50]})
        batch = np.concatenate([take(400)] + [perm[k][100:125] for k in banks])
        labels = cluster(emb[batch], config.n_clusters, config.seed)
        clusters += [
            {"q": "q3", "seed": s, "type": None, "mix": f"c{c}", "rows": batch[labels == c]}
            for c in range(config.n_clusters)
        ]

    to_name = [c for c in clusters if c["q"] != "q2"]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [
                naming_prompt(
                    [texts[i] for i in c["rows"]],
                    config.n_shown,
                    config.seed,
                    template={"yesno": NAMING_PROMPT, "share": NAMING_PROMPT_SHARE, "grain": NAMING_PROMPT_GRAIN}[
                        config.coherence_rule
                    ],
                )
                for c in to_name
            ],
            str(run / "naming_cache"),
        )
    )
    for c, reply in zip(to_name, replies):
        c.update(parse_naming(reply))
        if config.coherence_rule != "yesno":
            n_shown = min(config.n_shown, len(c["rows"]))
            c["coherent"] = None if c["shared"] is None else c["shared"] >= n_shown / 2
    to_judge = [c for c in clusters if c["q"] == "q1" and c["mix"] in ("pure", "majority")]
    verdicts = asyncio.run(
        name_clusters(
            client,
            config.judge,
            [LABEL_PROMPT.format(name=c["name"], description=c["description"]) for c in to_judge],
            str(run / "judge_cache"),
        )
    )
    for c, reply in zip(to_judge, verdicts):
        c["judge_label"] = parse_label(reply, ["LEGAL", "MEDICAL", "CODE", "OTHER"])
        c["name_matches"] = c["judge_label"] == EXPECTED[c["type"]]

    for c in clusters:
        rows = c["rows"]
        counts = {k: int((source[rows] == k).sum()) for k in banks}
        c["size"] = len(rows)
        c["median_coverage"] = float(np.median(coverage[rows]))
        c["planted"] = counts
        c["confirmed"] = (
            c["q"] != "q2" and bool(c.get("coherent")) and len(rows) >= config.min_size and c["median_coverage"] < cut
        )

    q1 = [c for c in clusters if c["q"] == "q1"]
    coh = {
        mix: sum(bool(c["coherent"]) for c in q1 if c["mix"] == mix)
        for mix in ("pure", "majority", "minority", "random")
    }
    nulls = sum(c["coherent"] is None for c in clusters if c["q"] != "q2")
    n_match = sum(c["name_matches"] for c in to_judge)
    if config.coherence_rule != "yesno":
        coh_verdict = (
            "valid"
            if coh["pure"] >= 8 and coh["majority"] >= 7 and coh["minority"] <= 2 and coh["random"] <= 1
            else "invalid"
            if coh["pure"] <= 6 or coh["majority"] <= 4 or coh["random"] >= 3
            else "ambiguous"
        )
    else:
        coh_verdict = (
            "valid"
            if coh["pure"] >= 8 and coh["random"] <= 1 and coh["majority"] > coh["minority"]
            else "invalid"
            if coh["pure"] <= 6 or coh["random"] >= 3
            else "ambiguous"
        )
    name_verdict = "valid" if n_match >= 15 else "invalid" if n_match <= 11 else "ambiguous"

    pures = [c for c in clusters if c["mix"] == "pure"]
    gap_below = sum(c["median_coverage"] < cut for c in pures if c["type"] in GAP)
    cov_below = sum(c["median_coverage"] < cut for c in pures if c["type"] in COVERED)
    cov_verdict = (
        "valid" if gap_below == 6 and cov_below == 0 else "invalid" if gap_below <= 3 or cov_below >= 3 else "ambiguous"
    )
    pool_cov = coverage[: len(pool)]
    coverage_rows = {
        k: {
            "median": round(float(np.median(coverage[bank_idx[k]])), 4),
            "auroc_low_vs_pool": round(auroc_low(coverage[bank_idx[k]], pool_cov), 4),
        }
        for k in banks
    } | {"pool": {"median": round(float(np.median(pool_cov)), 4)}}

    q3 = [c for c in clusters if c["q"] == "q3"]
    found, purity, background = {}, {}, 0
    for s in config.draw_seeds:
        cs = [c for c in q3 if c["seed"] == s]
        for k in banks:
            purity[f"{k}_s{s}"] = max(c["planted"][k] for c in cs) / 25
            found[f"{k}_s{s}"] = any(
                c["confirmed"] and c["planted"][k] >= 13 and c["planted"][k] / c["size"] >= 0.5 for c in cs
            )
        background += sum(c["confirmed"] and sum(c["planted"].values()) / c["size"] < 0.5 for c in cs)
    gap_found = sum(v for k, v in found.items() if k.split("_")[0] in GAP)
    cov_found = sum(v for k, v in found.items() if k.split("_")[0] in COVERED)
    e2e_verdict = (
        "valid"
        if gap_found >= 5 and cov_found <= 1 and background <= 3
        else "invalid"
        if gap_found <= 2 or cov_found >= 3
        else "ambiguous"
    )

    parts = {
        "q1_coherence": coh_verdict,
        "q1_names": name_verdict,
        "q2_coverage": cov_verdict,
        "q3_end_to_end": e2e_verdict,
    }
    decision = {
        "coverage_cut": round(cut, 4),
        "q1_coherent_yes_of_9": coh,
        "q1_names_match_of_18": n_match,
        "q2_gap_pure_below_cut_of_6": gap_below,
        "q2_covered_pure_below_cut_of_6": cov_below,
        "q2_rows": coverage_rows,
        "q3_gap_found_of_6": gap_found,
        "q3_covered_found_of_6": cov_found,
        "q3_background_confirmations": background,
        "q3_found": found,
        "q3_purity": {k: round(v, 4) for k, v in purity.items()},
        "null_names": nulls,
        "verdicts": parts,
        "validated": all(v == "valid" for v in parts.values()),
    }
    for c in clusters:
        c["id"] = f"{c['q']}_s{c['seed']}_{c['type'] or 'batch'}_{c['mix']}"
    if config.compare_clusters:
        ref_rows = {r["id"]: r for r in load_jsonl(config.compare_clusters)}
        assert set(ref_rows) == {c["id"] for c in clusters}, "cluster ids differ from the comparison run"
        for c in clusters:
            r = ref_rows[c["id"]]
            assert (r["size"], r["planted"], r["median_coverage"]) == (
                c["size"],
                c["planted"],
                round(c["median_coverage"], 4),
            ), f"{c['id']} differs from the comparison run"
        print(f"all {len(clusters)} clusters match {config.compare_clusters}")
    write_jsonl(
        [
            {
                "id": c["id"],
                **{k: v for k, v in c.items() if k != "rows"},
                "median_coverage": round(c["median_coverage"], 4),
                "examples": [" ".join(texts[i].split())[:200] for i in c["rows"][:5]],
            }
            for c in clusters
        ],
        run / "clusters.jsonl",
    )
    write_json(decision, run / "decision.json")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
