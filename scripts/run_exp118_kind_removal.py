# exp118: removes each exp116 prompt kind from the 200k dolci sample, rescores traffic with delta-knn, and asks whether topic groups or kind groups surface the removed kind.
# uv run --no-sync -m scripts.run_exp118_kind_removal --exp112_dir results/raw/261029_exp112_gap_list_v1 --tags_dir data/processed/exp116 --line_path results/raw/261030_exp115_gap_line_v1/line.json --output_dir results/raw/261030_exp118_kind_removal_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch

from src.data.io import load_jsonl
from src.generation.prompt_kinds import ALL_KINDS
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Exp118Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    exp112_dir: str = ""
    tags_dir: str = ""
    line_path: str = ""
    feats_cache: str = "/dev/shm/exp108_feats.npy"
    layer: int = 20
    k: int = 100
    n_mix: int = 200000
    n_ref: int = 3000
    majority: float = 0.5
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(Exp118Config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    torch.backends.cudnn.deterministic = True

    traffic = load_jsonl(Path(config.exp112_dir) / "prompt_scores.jsonl")
    t_tags = load_jsonl(Path(config.tags_dir) / "traffic_tags.jsonl")
    d_tags = load_jsonl(Path(config.tags_dir) / "dolci_tags.jsonl")
    assert [t["id"] for t in t_tags] == [r["id"] for r in traffic]
    assert len(d_tags) == config.n_mix
    cuts = {float(q): v for q, v in json.loads(Path(config.line_path).read_text())["cuts"].items()}
    n_t = len(traffic)
    topic = np.array([r["topic"] for r in traffic])
    topics = np.unique(topic)
    t_is = {k: np.array([t[k] is True for t in t_tags]) for k in ALL_KINDS}
    # a null dolci tag counts as the kind, as in exp111
    d_is = {k: np.array([t[k] is not False for t in d_tags]) for k in ALL_KINDS}
    majority = {k: {int(t) for t in topics if t_is[k][topic == t].mean() >= config.majority} for k in ALL_KINDS}

    feats = np.load(config.feats_cache)
    assert len(feats) == n_t + config.n_mix + config.n_ref
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix, xdolma = x[:n_t], x[n_t : n_t + config.n_mix], x[n_t + config.n_mix :]
    to_dolma = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()
    # one fixed scaling for every arm: the full sample's per-dimension std, as exp111 and exp112
    sd = xmix.std(0) + 1e-6
    xt_s, xmix_s = xt / sd, xmix / sd
    del x, xt, xmix

    def score(keep: np.ndarray) -> np.ndarray:
        ref = xmix_s[torch.tensor(keep, device="cuda")]
        return knn_distances(ref, xt_s, (config.k,), scale=False)[config.k].cpu().numpy() - to_dolma

    full = score(np.ones(config.n_mix, dtype=bool))
    exp112 = np.array([r["delta_knn"] for r in traffic])
    LOGGER.info(f"full arm vs exp112 max abs diff {np.abs(full - exp112).max():.5f}")

    def measure(s: np.ndarray, kind: str) -> dict:
        tmed = np.array([np.median(s[topic == t]) for t in topics])
        order = topics[np.argsort(-tmed)]
        kmeds = {k: float(np.median(s[t_is[k]])) for k in ALL_KINDS if t_is[k].any()}
        all_meds = np.concatenate([tmed, np.array(list(kmeds.values()))])
        own = kmeds.get(kind)
        return {
            "kind_majority_in_top": {n: sum(int(t) in majority[kind] for t in order[:n]) for n in (20, 50, 100)},
            "kind_majority_over_line": {
                q: sum(m > c for t, m in zip(topics, tmed, strict=True) if int(t) in majority[kind])
                for q, c in cuts.items()
            },
            "kind_group_median": None if own is None else round(own, 4),
            "kind_group_rank_all_groups": None if own is None else int((all_meds > own).sum()) + 1,
            "kind_group_rank_among_kinds": None if own is None else int(sum(v > own for v in kmeds.values())) + 1,
            "kind_group_line": None if own is None else max((q for q, c in cuts.items() if own > c), default=None),
            "kind_prompt_median_shift": None,
        }

    rows = []
    for kind in ALL_KINDS:
        removed = d_is[kind]
        n_rm = int(removed.sum())
        rand = np.ones(config.n_mix, dtype=bool)
        rand[np.random.default_rng(config.seed).choice(config.n_mix, size=n_rm, replace=False)] = False
        arms = {"full": full, "removed": score(~removed), "random": score(rand)}
        for arm, s in arms.items():
            m = measure(s, kind)
            if t_is[kind].any():
                m["kind_prompt_median_shift"] = round(float(np.median(s[t_is[kind]] - full[t_is[kind]])), 4)
            rows.append(
                {
                    "id": f"{kind}|{arm}",
                    "kind": kind,
                    "arm": arm,
                    "n_dolci_removed": 0 if arm == "full" else n_rm,
                    "n_traffic_tagged": int(t_is[kind].sum()),
                    "n_kind_majority_topics": len(majority[kind]),
                    **m,
                }
            )
        LOGGER.info(
            f"{kind}: removed {n_rm}, {[(r['arm'], r['kind_majority_in_top'], r['kind_group_rank_all_groups']) for r in rows[-3:]]}"
        )
        torch.cuda.empty_cache()
    write_jsonl(rows, out / "arms.jsonl")

    decision = {
        "n_traffic": n_t,
        "n_topics": len(topics),
        "n_groups_ranked": len(topics) + len(ALL_KINDS),
        "full_vs_exp112_max_abs_diff": round(float(np.abs(full - exp112).max()), 6),
        "n_rows": len(rows),
        "n_rows_expected": 3 * len(ALL_KINDS),
        "n_null_dolci_tags": {k: int(sum(t[k] is None for t in d_tags)) for k in ALL_KINDS},
        "by_kind": {
            k: {r["arm"]: {x: r[x] for x in r if x not in ("id", "kind", "arm")} for r in rows if r["kind"] == k}
            for k in ALL_KINDS
        },
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'arms.jsonl'} and {out / 'decision.json'}")
    for k in ALL_KINDS:
        b = decision["by_kind"][k]
        print(k, {a: (b[a]["kind_majority_in_top"][100], b[a]["kind_group_rank_all_groups"]) for a in b})


if __name__ == "__main__":
    main()
