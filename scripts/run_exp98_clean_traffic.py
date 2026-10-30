# exp98 step 1: drop sampled traffic prompts that are in the dolci mix word for word or nearly (minilm cosine at least 0.95), and print 20 borderline near copies beside their dolci match for a by-eye check.
# uv run --no-sync -m scripts.run_exp98_clean_traffic --nearest_path results/raw/261029_exp96_full_dolci_v1/prompt_nearest.jsonl --dataset_path results/raw/261028_exp95_group_removal_v1/prompts.jsonl --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --output_dir results/raw/261029_exp98_clean_v1 --seed 0

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class CleanConfig(Config):
    nearest_path: str = ""
    exp37_dir: str = ""
    copy_cos: float = 0.95
    n_check: int = 20
    check_hi: float = 0.98
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(CleanConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    near = {r["id"]: r for r in load_jsonl(config.nearest_path)}
    traffic = [r for r in load_jsonl(config.dataset_path) if r["kind"] == "traffic"]
    assert len(near) == len(traffic), f"{len(near)=} {len(traffic)=}"
    is_copy = {
        r["id"]: near[r["id"]]["exact_in_dolci"] or near[r["id"]]["nearest_cos"] >= config.copy_cos for r in traffic
    }
    kept = [{"id": r["id"], "topic": r["topic"], "prompt": r["prompt"]} for r in traffic if not is_copy[r["id"]]]
    write_jsonl(kept, out / "clean_prompts.jsonl")
    left = Counter(r["topic"] for r in kept)
    per_topic = np.array([left.get(t, 0) for t in range(max(r["topic"] for r in traffic) + 1)])

    # borderline near copies: not exact, cosine in [copy_cos, check_hi)
    border = [
        r
        for r in traffic
        if not near[r["id"]]["exact_in_dolci"] and config.copy_cos <= near[r["id"]]["nearest_cos"] < config.check_hi
    ]
    picks = [border[i] for i in rng.choice(len(border), size=min(config.n_check, len(border)), replace=False)]
    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    _, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    q = torch.tensor(np.stack([t_np[int(r["id"][1:])] for r in picks]), dtype=torch.float32).cuda()
    sims, js = (q.half() @ torch.from_numpy(m_np).cuda().T).float().max(dim=1)
    check = [
        {"id": r["id"], "cos": round(float(s), 4), "traffic": r["prompt"][:400], "dolci": m_text[int(j)][:400]}
        for r, s, j in zip(picks, sims.cpu(), js.cpu())
    ]
    write_json(check, out / "border_check.json")

    decision = {
        "n_sampled": len(traffic),
        "n_exact": int(sum(near[r["id"]]["exact_in_dolci"] for r in traffic)),
        "n_copy": int(sum(is_copy.values())),
        "n_kept": len(kept),
        "n_border": len(border),
        "topics_with_0_left": int((per_topic == 0).sum()),
        "topics_with_under_5_left": int((per_topic < 5).sum()),
        "topics_with_15_plus_left": int((per_topic >= 15).sum()),
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
