# tests whether distance from the training data orders the tied top topic_hand_labels topics the way the user's labels do.
# uv run -m scripts.run_distance_check --labels_path results/raw/261022_topic_hand_labels_topic_labels_v1/decision.json --clusters_path results/raw/261021_pipeline_million_gap_pipeline_v2/clusters.jsonl --output_dir results/raw/261022_distance_vs_user_rank_distance_check_v1 --seed 0
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr

from src.utils.config import Config, write_config_json

LOGGER = logging.getLogger(__name__)
CODE = {"real_gap": 2, "partly": 1, "covered": 0}


@dataclass
class DistanceCheckConfig(Config):
    labels_path: str = ""
    clusters_path: str = ""
    n_perm: int = 100_000


def perm_p(score: np.ndarray, label: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rho = spearmanr(score, label).statistic
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(score, rng.permutation(label)).statistic for _ in range(n)])
    return float(rho), float((np.sum(null >= rho) + 1) / (n + 1))


def main() -> None:
    config = simple_parsing.parse(DistanceCheckConfig)
    logging.basicConfig(level=logging.INFO)
    out = Path(config.output_dir)
    out.mkdir(parents=True)
    write_config_json(config, out)
    rank = {c["id"]: c["rank"] for c in map(json.loads, Path(config.clusters_path).open())}
    rows = [r for r in json.loads(Path(config.labels_path).read_text())["rows"] if r["auto_confirmed"]]
    rows.sort(key=lambda r: rank[r["id"]])
    label = np.array([CODE[r["human"]] for r in rows])
    # farthest first means a lower pipeline_million rank, so negate rank for the score
    rho, p = perm_p(-np.array([rank[r["id"]] for r in rows]), label, config.n_perm, config.seed)
    rho_shared, p_shared = perm_p(np.array([r["shared"] for r in rows]), label, config.n_perm, config.seed)
    verdict = "works" if rho >= 0.5 and p < 0.05 else "fails" if rho <= 0.2 else "ambiguous"
    decision = {
        "n": len(rows),
        "rho_distance": round(rho, 4),
        "p_distance": round(p, 4),
        "verdict": verdict,
        "real_gap_positions": [i + 1 for i, r in enumerate(rows) if r["human"] == "real_gap"],
        "rho_shared_baseline": round(rho_shared, 4),
        "p_shared_baseline": round(p_shared, 4),
        "order": [
            {"pos": i + 1, "id": r["id"], "rank": rank[r["id"]], "human": r["human"], "name": r["name"]}
            for i, r in enumerate(rows)
        ],
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    for o in decision["order"]:
        print(o["pos"], o["human"], o["name"])
    print(f"{rho=:.3f} {p=:.4f} {verdict=}; baseline {rho_shared=:.3f} {p_shared=:.4f}")
    print(f"Saved {out / 'decision.json'}")


if __name__ == "__main__":
    main()
