# exp97: exp96's olmo differential, plain minilm distance and the near-copy filter scored on exp95's removed-group split, where the answer key comes from the split rather than from label screens.
# uv run --no-sync -m scripts.run_exp97_split_scores --dataset_path results/raw/261028_exp95_group_removal_v1/prompts.jsonl --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --dolma_path data/processed/exp01/dolma.jsonl --output_dir results/raw/261029_exp97_split_scores_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_exp94_topic_removal import bootstrap_diff, evaluate, make_arm
from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("main", "dolci")


@dataclass
class Exp97Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    exp37_dir: str = ""
    dolma_path: str = ""
    layer: int = 20
    k: int = 10
    n_ref: int = 20
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    copy_cos: float = 0.95
    copy_share: float = 0.5
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    n_boot: int = 10000
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(Exp97Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    recs = load_jsonl(config.dataset_path)
    if config.num_tasks:
        recs = [r for r in recs if r["kind"] == "dolci" or r["topic"] < config.num_tasks][: config.num_tasks * 40]
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)]
    topic = np.array([r["topic"] for r in recs])
    kind = np.array([r["kind"] for r in recs])
    group_of = np.zeros(topic.max() + 1, dtype=int)
    for r in recs:
        if r["topic"] >= 0:
            group_of[r["topic"]] = r["group"]

    # minilm embeddings as exp95 used them: traffic from the pool, dolci from the mix
    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    m_np, _ = load_mix(c37["sft_path"], c37["emb_v1"])
    emb = np.stack([t_np[int(r["id"][1:])] if r["kind"] == "traffic" else m_np[int(r["id"][1:])] for r in recs])
    assert all(rows[int(r["id"][1:])]["text"] == r["prompt"] for r in recs[:300] if r["kind"] == "traffic")
    del rows, t_np, m_np
    emb_t = torch.nn.functional.normalize(torch.tensor(emb, dtype=torch.float32, device="cuda"), dim=1)

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    texts = [r["prompt"] for r in recs] + dolma
    encoded = [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in texts]
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=[config.layer])
    del model
    torch.cuda.empty_cache()
    x_all = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
    x, x_dolma = x_all[: len(recs)], x_all[len(recs) :]
    n_tok = feats["n_content_tokens"][: len(recs)]
    LOGGER.info(f"{x.shape=} {x_dolma.shape=}")

    out_rows, topic_scores = [], {}
    for arm in ARMS:
        for seed in config.split_seeds:
            a = make_arm(arm, topic, group_of, kind, seed, config.n_ref)
            ref, ev = a["ref"], a["ev"]
            sd = torch.cat([x[ref], x_dolma]).std(0) + 1e-6
            d_ref = knn_distances(x[ref] / sd, x[ev] / sd, (1, config.k), scale=False)
            d_dolma = knn_distances(x_dolma / sd, x[ev] / sd, (1, config.k), scale=False)
            near = torch.cat([(emb_t[ev[i : i + 4096]] @ emb_t[ref].T).max(1).values for i in range(0, len(ev), 4096)])
            near = near.cpu().numpy()
            is_copy = near >= config.copy_cos
            share = {t: is_copy[a["topic"] == t].mean() for t in np.unique(a["topic"])}
            dropped = np.array([share[t] >= config.copy_share for t in a["topic"]])
            scores = {
                f"olmo_diff_k{config.k}": (d_ref[config.k] - d_dolma[config.k]).cpu().numpy(),
                "olmo_diff_k1": (d_ref[1] - d_dolma[1]).cpu().numpy(),
                f"olmo_one_k{config.k}": d_ref[config.k].cpu().numpy(),
                "olmo_one_k1": d_ref[1].cpu().numpy(),
                "minilm_dist": 1 - near,
                "length": np.log(n_tok[ev]),
                "random": np.random.default_rng(3000 + seed).random(len(ev)),
            }
            # filtered topics go to the bottom: their prompts all get a score below every unfiltered prompt
            base = scores[f"olmo_diff_k{config.k}"]
            scores[f"olmo_diff_k{config.k}_filter50"] = np.where(dropped, base.min() - 1, base)
            for name, s in scores.items():
                res, tm = evaluate(s, a, arm)
                topic_scores[(name, arm, seed)] = tm
                out_rows.append(
                    {
                        "id": f"{name}|{arm}|{seed}",
                        "score": name,
                        "arm": arm,
                        "seed": seed,
                        "n_dropped_topics": int(sum(v >= config.copy_share for v in share.values())),
                        **{k: round(v, 4) for k, v in res.items()},
                    }
                )
            LOGGER.info(f"{arm=} {seed=} done")
    write_jsonl(out_rows, out / "scores.jsonl")

    summary = {
        f"{name}|{arm}": {
            p: round(float(np.mean([r[p] for r in out_rows if r["score"] == name and r["arm"] == arm])), 4)
            for p in ("topic_val", "topic_test", "prompt_test")
        }
        for name in sorted({r["score"] for r in out_rows})
        for arm in ARMS
    }
    boot = {
        f"{a}_minus_minilm|{arm}|{seed}": bootstrap_diff(
            topic_scores[(a, arm, seed)], topic_scores[("minilm_dist", arm, seed)], config.n_boot, seed
        )
        for a in (f"olmo_diff_k{config.k}", f"olmo_diff_k{config.k}_filter50", "olmo_diff_k1")
        for arm in ARMS
        for seed in config.split_seeds
    }
    decision = {
        "n_rows": len(out_rows),
        "n_rows_expected": len(ARMS) * len(config.split_seeds) * 8,
        "n_prompts": len(recs),
        "summary_mean_over_seeds": summary,
        "bootstrap_ci_topic_test": boot,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
