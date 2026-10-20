# exp23: relative-frequency gap test (traffic vs mix knn density ratio), checked on planted known answers, then applied to exp20's gaps.
# uv run -m scripts.run_relative_frequency --run_dir results/raw/261020_exp23_relative_frequency_v1 --exp20_dir results/raw/261019_exp20_act_vs_word_v1 --banks_dir data/processed/exp06 --idmix_path data/processed/exp01/idmix.jsonl --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from sentence_transformers import SentenceTransformer

from src.data.gaps import rebuild_exp20_gaps
from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo, normalise
from src.metrics.coverage import density_ratios
from src.metrics.detectors import auroc
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("act", "word", "random")
PLANTED = ("legal", "medical", "magicoder", "aya", "idmix")


@dataclass
class RelativeFrequencyConfig:
    run_dir: str = ""
    exp20_dir: str = ""
    banks_dir: str = ""
    idmix_path: str = ""
    seed: int = 0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    k: int = 50
    threshold: float = 5.0
    n_planted: int = 300
    cluster_size: int = 50


def main():
    config = simple_parsing.parse(RelativeFrequencyConfig)
    assert config.run_dir and config.exp20_dir and config.banks_dir and config.idmix_path, "input paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c9 = json.loads((Path(config.exp20_dir) / "config.json").read_text())["exp09_config"]
    pool = [r["text"] for r in load_jsonl(c9["wildchat_pool"])] + [r["text"] for r in load_jsonl(c9["lmsys_pool"])]
    embedder = SentenceTransformer(c9["embed_model"], device="cuda", revision=c9["embed_model_revision"])

    def encode(t: list[str]) -> np.ndarray:
        return embedder.encode(
            t, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    dolci = load_dolci_dpo()["first_user"].drop_duplicates().tolist()
    m_emb, m_norm = encode(dolci), [normalise(t) for t in dolci]
    pool_emb = encode(pool)

    def ratio(members: list[int], t_emb: np.ndarray, t_norm: list[str]) -> float:
        r = density_ratios(
            t_emb[members], [t_norm[i] for i in members], members, t_emb, t_norm, m_emb, m_norm, config.k
        )
        return float(np.median(r))

    # q1: planted known answers inside the traffic pool
    banks = {t: [r["text"] for r in load_jsonl(Path(config.banks_dir) / f"needles_{t}.jsonl")] for t in PLANTED[:4]}
    banks["idmix"] = [r["text"] for r in load_jsonl(config.idmix_path)]
    planted_texts, planted_type = [], []
    for t in PLANTED:
        pick = rng.choice(len(banks[t]), size=config.n_planted, replace=False)
        planted_texts += [banks[t][i] for i in pick]
        planted_type += [t] * config.n_planted
    t1_texts = pool + planted_texts
    t1_emb = np.vstack([pool_emb, encode(planted_texts)])
    t1_norm = [normalise(t) for t in t1_texts]
    planted_type = np.array(["pool"] * len(pool) + planted_type)
    q1 = []
    for s in config.seeds:
        r_s = np.random.default_rng(s)
        for t in (*PLANTED, "pool"):
            idx = np.where(planted_type == t)[0]
            members = r_s.choice(idx, size=config.cluster_size, replace=False).tolist()
            q1.append({"id": f"s{s}_{t}", "seed": s, "type": t, "ratio": ratio(members, t1_emb, t1_norm)})
    gap_c = [x for x in q1 if x["type"] in ("legal", "medical")]
    cov_c = [x for x in q1 if x["type"] in ("magicoder", "idmix")]
    n_gap = sum(x["ratio"] >= config.threshold for x in gap_c)
    n_cov = sum(x["ratio"] >= config.threshold for x in cov_c)
    q1_verdict = "valid" if n_gap == 6 and n_cov <= 1 else "invalid" if n_gap <= 3 or n_cov >= 3 else "ambiguous"
    q1_auroc = auroc(np.log([x["ratio"] for x in gap_c]), np.log([x["ratio"] for x in cov_c]))

    # q2: exp20's gaps, with the unplanted pool as traffic
    p_norm = [normalise(t) for t in pool]
    gaps = rebuild_exp20_gaps(pool_emb, pool, config.exp20_dir)
    for g in gaps:
        g["ratio"] = ratio(g["members"], pool_emb, p_norm)
        g["real"] = g["ratio"] >= config.threshold
    per_seed = {}
    for s in config.seeds:
        gs = [g for g in gaps if g["seed"] == s]
        real = {a: sum(g["real"] for g in gs if g["arm"] == a) for a in ARMS}
        act_only = sum(g["real"] and bool(g["only"]) for g in gs if g["arm"] == "act")
        word_only = sum(g["real"] and bool(g["only"]) for g in gs if g["arm"] == "word")
        if act_only >= 2 and real["act"] >= 2 * real["random"] and real["act"] >= real["random"] + 2:
            outcome = "adds_something"
        elif act_only == 0 or real["act"] <= real["random"] + 1:
            outcome = "adds_nothing"
        else:
            outcome = "ambiguous"
        per_seed[str(s)] = {
            "real_gaps": real,
            "act_only_real": act_only,
            "word_only_real": word_only,
            "outcome": outcome,
        }
    outcomes = [v["outcome"] for v in per_seed.values()]
    q2_verdict = next((o for o in ("adds_something", "adds_nothing") if outcomes.count(o) >= 2), "ambiguous")
    if q1_verdict != "valid":
        q2_verdict = f"not_evaluated_q1_{q1_verdict} ({q2_verdict})"

    # reported only: ratios of every reviewed exp20 cluster per arm
    e20 = Path(config.exp20_dir)
    reviews = {(r["arm"], r["round"]): np.array(r["rows"]) for r in load_jsonl(e20 / "reviews.jsonl")}
    all_review = {
        a: ratio(np.concatenate([reviews[(a, r)] for r in range(1, 6)]).tolist(), pool_emb, p_norm) for a in ARMS
    }

    decision = {
        "q1_verdict": q1_verdict,
        "q1_gap_clusters_at_or_above_threshold_of_6": n_gap,
        "q1_covered_clusters_at_or_above_threshold_of_6": n_cov,
        "q1_auroc_gap_vs_covered": q1_auroc,
        "q1_ratios": {x["id"]: round(x["ratio"], 3) for x in q1},
        "q2_verdict": q2_verdict,
        "q2_per_seed": per_seed,
        "q2_gaps": {
            g["id"]: {
                "arm": g["arm"],
                "name": g["name"],
                "only": g["only"],
                "ratio": round(g["ratio"], 3),
                "real": g["real"],
                "size": len(g["members"]),
            }
            for g in gaps
        },
        "reviewed_rows_median_ratio_per_arm": all_review,
        "pool_size": len(pool),
        "mix_size": len(dolci),
    }
    write_json(decision, run / "decision.json")
    write_jsonl(q1, run / "q1_clusters.jsonl")
    write_jsonl([{k: v for k, v in g.items() if k != "members"} for g in gaps], run / "q2_gaps.jsonl")
    for x in q1:
        print(f"{x['id']}: ratio {x['ratio']:.2f}")
    print(f"{q1_verdict=} gap {n_gap}/6 covered {n_cov}/6 auroc {q1_auroc:.3f}")
    for g in gaps:
        print(f"{g['id']} {g['name'][:40]} only={g['only']} ratio {g['ratio']:.2f}")
    print(f"{per_seed=}")
    print(f"{q2_verdict=}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
