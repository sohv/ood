# exp115: draws the gap line from held-out dolci prompts' delta-knn scores, sweeps the percentile, confirms flagged topics with exp88's question and reports behaviour per line.
# uv run --no-sync -m scripts.run_exp115_gap_line --phase line --dataset_path results/raw/261029_exp98_clean_v1/clean_prompts.jsonl --exp112_dir results/raw/261029_exp112_gap_list_v1 --output_dir results/raw/261030_exp115_gap_line_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0
# uv run --no-sync -m scripts.run_exp115_gap_line --phase confirm --exp112_dir results/raw/261029_exp112_gap_list_v1 --output_dir results/raw/261030_exp115_gap_line_v1 --model_id anthropic/claude-sonnet-4.6 --seed 0
# uv run --no-sync -m scripts.run_exp115_gap_line --phase pool --exp112_dir results/raw/261029_exp112_gap_list_v1 --pool_out data/processed/exp115/pool.jsonl --output_dir results/raw/261030_exp115_gap_line_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0
# uv run --no-sync -m scripts.run_exp115_gap_line --phase report --judge_dir results/raw/261030_exp115_judge_v1 --output_dir results/raw/261030_exp115_gap_line_v1 --model_id google/gemini-3.8-flash --seed 0

import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_exp113_gap_behaviour import Exp113Config, boot_diff
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract, render_user
from src.interp.train_only_detectors import knn_distances
from src.metrics.confirmation import CONFIRM_PROMPT, confirmed_by, parse_confirmation
from src.metrics.discovery import name_clusters, naming_prompt
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Exp115Config(Config):
    phase: str = ""
    exp112_dir: str = ""
    judge_dir: str = ""
    pool_out: str = ""
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7"
    mix_text_path: str = "cache/exp80/mix_text.parquet"
    feats_cache: str = "/dev/shm/exp108_feats.npy"
    n_mix: int = 200000
    n_ref: int = 3000
    n_heldout: int = 10000
    heldout_seed: int = 1
    min_prompts: int = 5
    qs: list[float] = field(default_factory=lambda: [0.90, 0.95, 0.99, 0.999])
    layer: int = 20
    k: int = 100
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    n_confirm: int = 25
    per_topic: int = 5
    n_control: int = 50
    max_prompt_tokens: int = 6144
    seed: int = 0


def line(config: Exp115Config) -> None:
    out = Path(config.output_dir)
    # traffic count and sample drawn exactly as in exp107/exp108, so the cached rows line up
    traffic = load_jsonl(config.dataset_path)
    counts = Counter(r["topic"] for r in traffic)
    n_t = sum(counts[r["topic"]] >= config.min_prompts for r in traffic)
    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=config.n_mix, replace=False))
    outside = np.setdiff1d(np.arange(len(mix_text)), sample)
    n_h = config.n_heldout if not config.num_tasks else config.num_tasks * 20
    held = np.sort(np.random.default_rng(config.heldout_seed).choice(outside, size=n_h, replace=False))
    assert not np.isin(held, sample).any()
    texts = [mix_text[i] for i in held]
    del mix_text

    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    model = (
        AutoModelForCausalLM.from_pretrained(config.olmo_id, revision=config.olmo_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    xh = extract(
        model,
        [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in texts],
        tok.pad_token_id or 0,
        max_tokens=config.batch_tokens,
        layers=[config.layer],
    )["cmean"][:, 0]
    del model
    torch.cuda.empty_cache()

    feats = np.load(config.feats_cache, mmap_mode="r")
    assert len(feats) == n_t + config.n_mix + config.n_ref, f"{len(feats)=} {n_t=}"
    xmix = torch.tensor(np.asarray(feats[n_t : n_t + config.n_mix]), device="cuda").float()
    xdolma = torch.tensor(np.asarray(feats[n_t + config.n_mix :]), device="cuda").float()
    xh = torch.tensor(xh, device="cuda").float()
    # same scoring as exp112: reference std from the 200k sample, dolma term with dolma's own std
    sd = xmix.std(0) + 1e-6
    to_dolci = knn_distances(xmix / sd, xh / sd, (config.k,), scale=False)[config.k].cpu().numpy()
    to_dolma = knn_distances(xdolma, xh, (config.k,))[config.k].cpu().numpy()
    score = to_dolci - to_dolma
    write_jsonl(
        [
            {"id": f"mix{i}", "delta_knn": round(float(s), 4), "text": t[:300]}
            for i, s, t in zip(held, score, texts, strict=True)
        ],
        out / "heldout_scores.jsonl",
    )

    topics = load_jsonl(Path(config.exp112_dir) / "topics.jsonl")
    med = np.array([r["median_delta_knn"] for r in topics])
    cuts = {str(q): round(float(np.quantile(score, q)), 4) for q in config.qs}
    write_jsonl(
        [
            {
                "id": r["id"],
                "topic": r["topic"],
                "rank": r["rank"],
                "median_delta_knn": r["median_delta_knn"],
                # share of held-out training prompts scoring below this topic's median
                "heldout_percentile": round(float((score < r["median_delta_knn"]).mean()), 4),
                **{f"flag_{q}": bool(r["median_delta_knn"] > c) for q, c in cuts.items()},
            }
            for r in topics
        ],
        out / "topic_flags.jsonl",
    )
    decision = {
        "n_heldout": int(len(held)),
        "n_nan": int(np.isnan(score).sum()),
        "heldout_score_quantiles": {
            str(q): round(float(np.quantile(score, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "traffic_median_quantiles": {str(q): round(float(np.quantile(med, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "cuts": cuts,
        "n_flagged": {q: int((med > c).sum()) for q, c in cuts.items()},
        "n_topics": len(topics),
    }
    write_json(decision, out / "line.json")
    print(f"Results saved to {out / 'line.json'} and {out / 'topic_flags.jsonl'}")
    print(json.dumps(decision, indent=1))


def confirm(config: Exp115Config) -> None:
    out = Path(config.output_dir)
    loosest = f"flag_{min(config.qs)}"
    flags = [r for r in load_jsonl(out / "topic_flags.jsonl") if r[loosest]]
    prompts = load_jsonl(Path(config.exp112_dir) / "prompt_scores.jsonl")
    by_topic: dict[int, list[str]] = {}
    for r in prompts:
        by_topic.setdefault(r["topic"], []).append(r["prompt"])
    asks = [naming_prompt(by_topic[r["topic"]], config.n_confirm, config.seed, template=CONFIRM_PROMPT) for r in flags]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(name_clusters(client, config.model_id, asks, "cache/exp115_confirm", max_concurrent=8))
    rows = []
    for r, reply in zip(flags, replies, strict=True):
        shown = min(config.n_confirm, len(by_topic[r["topic"]]))
        cut = -(-shown // 2)
        parsed = parse_confirmation(reply)
        by = confirmed_by(parsed, cut)
        rows.append(
            {
                "id": r["id"],
                "topic": r["topic"],
                "rank": r["rank"],
                "n_shown": shown,
                "cut": cut,
                "failed": reply is None,
                "confirmed": None if reply is None else bool(by),
                "confirmed_by": by,
                **{k: parsed[k] for k in ("name", "shared", "language", "not_english", "format", "in_format")},
            }
        )
    write_jsonl(rows, out / "confirm.jsonl")
    summary = {
        "n_asked": len(rows),
        "n_failed": sum(r["failed"] for r in rows),
        "n_confirmed": sum(bool(r["confirmed"]) for r in rows),
        "confirmed_by": dict(Counter(c for r in rows for c in r["confirmed_by"])),
    }
    write_json(summary, out / "confirm.json")
    print(f"Results saved to {out / 'confirm.jsonl'}")
    print(json.dumps(summary, indent=1))


def pool(config: Exp115Config) -> None:
    out = Path(config.output_dir)
    rng = np.random.default_rng(config.seed)
    loosest = f"flag_{min(config.qs)}"
    flags = load_jsonl(out / "topic_flags.jsonl")
    gap = [r["topic"] for r in flags if r[loosest]]
    rest = [r["topic"] for r in flags if not r[loosest]]
    control = rng.choice(rest, size=config.n_control, replace=False).tolist()
    rank = {r["topic"]: r["rank"] for r in flags}
    prompts = load_jsonl(Path(config.exp112_dir) / "prompt_scores.jsonl")
    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    rows, n_too_long = [], {"gap": 0, "control": 0}
    for arm, ts in (("gap", gap), ("control", control)):
        for t in ts:
            cand = [r for r in prompts if r["topic"] == t]
            fits = [
                r
                for r in cand
                if len(tok(render_user(tok, r["prompt"]), add_special_tokens=False)["input_ids"])
                <= config.max_prompt_tokens
            ]
            n_too_long[arm] += len(cand) - len(fits)
            for j in rng.choice(len(fits), size=min(config.per_topic, len(fits)), replace=False):
                rows.append({"id": fits[j]["id"], "text": fits[j]["prompt"], "arm": arm, "topic": t, "rank": rank[t]})
    write_jsonl(rows, Path(config.pool_out))
    summary = {"n_pool": len(rows), "per_arm": dict(Counter(r["arm"] for r in rows)), "n_too_long": n_too_long}
    write_json(summary | {"control_topics": control}, out / "pool.json")
    print(f"Pool saved to {config.pool_out}")
    print(json.dumps(summary))


def report(config: Exp115Config) -> None:
    out = Path(config.output_dir)
    flags = {r["topic"]: r for r in load_jsonl(out / "topic_flags.jsonl")}
    conf = {r["topic"]: r["confirmed"] for r in load_jsonl(out / "confirm.jsonl")}
    judged = load_jsonl(Path(config.judge_dir) / "judgements.jsonl")
    ok = [r for r in judged if r["fail"] is not None]
    control = [r for r in ok if r["arm"] == "control"]
    boot_cfg = Exp113Config(seed=config.seed)

    def compare(gap: list[dict]) -> dict:
        if not gap:
            return {"n_topics": 0, "n_prompts": 0}
        rows = gap + control
        arm = np.array([r["arm"] for r in rows])
        topic = np.array([r["topic"] for r in rows])
        fail = np.array([r["fail"] for r in rows], dtype=float)
        g = float(np.mean([r["fail"] for r in gap]))
        c = float(np.mean([r["fail"] for r in control]))
        return {
            "n_topics": len({r["topic"] for r in gap}),
            "n_prompts": len(gap),
            "fail_rate": round(g, 4),
            "refused_rate": round(float(np.mean([r["refused"] for r in gap])), 4),
            "gap_minus_control": round(g - c, 4),
            "ci95_topic_bootstrap": boot_diff(fail, arm, topic, boot_cfg)
            if len({r["topic"] for r in gap}) > 1
            else None,
        }

    per_q = {}
    for q in config.qs:
        gap = [r for r in ok if r["arm"] == "gap" and flags[r["topic"]][f"flag_{q}"]]
        per_q[str(q)] = {
            "flagged": compare(gap),
            "flagged_confirmed": compare([r for r in gap if conf.get(r["topic"])]),
            "flagged_unconfirmed": compare([r for r in gap if conf.get(r["topic"]) is False]),
        }
    decision = {
        "n_rows": len(judged),
        "n_null": len(judged) - len(ok),
        "control": {
            "n_topics": len({r["topic"] for r in control}),
            "n_prompts": len(control),
            "fail_rate": round(float(np.mean([r["fail"] for r in control])), 4),
            "refused_rate": round(float(np.mean([r["refused"] for r in control])), 4),
        },
        "per_q": per_q,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


def main() -> None:
    config = simple_parsing.parse(Exp115Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    {"line": line, "confirm": confirm, "pool": pool, "report": report}[config.phase](config)


if __name__ == "__main__":
    main()
