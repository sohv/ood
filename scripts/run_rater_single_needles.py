# exp79: ranks exp11's seed-0 planted pool by exp56's flash variant-b rater, one request per call, and scores gap and control uplift@100.
# uv run -m scripts.run_rater_single_needles --output_dir results/raw/261027_exp79_rater_needles_v1 --exp11_dir results/raw/261019_exp11_ranking_bench_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --model_id google/gemini-3.8-flash --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch

from scripts.run_autorater_baseline import LABELS
from scripts.run_ranking_bench import CONTROL, GAP, TYPES, compare, jitter_order, top_metrics
from scripts.run_rater_single_auroc import PROMPT, REF_DESC, gap_scores, rate
from src.data.io import load_jsonl
from src.metrics.density_ratio import length_controlled
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
WORDS = 10.5
ACTIVATIONS_CMAX = 7.48


@dataclass
class RaterNeedlesConfig:
    output_dir: str = ""
    exp11_dir: str = ""
    exp37_dir: str = ""
    model_id: str = "google/gemini-3.8-flash"
    seed: int = 0
    needle_seed: int = 0
    max_chars: int = 300
    max_concurrent: int = 16
    n_examples: int = 5
    mix_cache: str = "cache/exp78_79_mix"
    num_tasks: int | None = None


def main():
    config = simple_parsing.parse(RaterNeedlesConfig)
    missing = [k for k in ("output_dir", "exp11_dir", "exp37_dir") if getattr(config, k) == ""]
    assert not missing, f"required arguments not set: {missing}"
    run = Path(config.output_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    torch.set_num_threads(16)
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "prompt": PROMPT, "ref_desc": REF_DESC}, run / "config.json"
    )

    # exp11's planted pool: wildchat deploy, lmsys, then n_per_type rows of each needle bank at this needle seed
    bc = json.loads((Path(config.exp11_dir) / "bench_config.json").read_text())
    n = bc["n_per_type"]
    s = config.needle_seed
    pool = [{"text": r["text"], "type": "deploy", "src_id": r["id"]} for r in load_jsonl(bc["wildchat_pool"])]
    pool += [{"text": r["text"], "type": "deploy", "src_id": r["id"]} for r in load_jsonl(bc["lmsys_pool"])]
    for t in TYPES:
        bank = load_jsonl(Path(bc["needles_dir"]) / f"needles_{t}.jsonl")[s * n : (s + 1) * n]
        pool += [{"text": r["text"], "type": t, "src_id": r["id"]} for r in bank]
    saved = load_jsonl(Path(config.exp11_dir) / "generations" / f"scores_seed{s}.jsonl")
    assert len(saved) == len(pool) == 20_400, (len(saved), len(pool))
    assert [r["type"] for r in saved] == [r["type"] for r in pool], "pool order differs from exp11's saved seed file"
    lengths_all = np.array([r["n_tokens"] for r in saved])
    tfidf_all = np.array([r["tfidf"] for r in saved])
    keep = np.arange(len(pool))
    if config.num_tasks:
        # smoke runs: the first deploy rows plus two of each needle type, so every type reaches the metrics
        needles = [np.where(np.array([r["type"] for r in pool]) == t)[0][:2] for t in TYPES]
        keep = np.concatenate([np.arange(config.num_tasks - 2 * len(TYPES)), *needles])
    pool = [pool[i] for i in keep]
    lengths, tfidf = lengths_all[keep], tfidf_all[keep]
    types = np.array([r["type"] for r in pool])
    LOGGER.info(f"{len(pool)} rows: " + ", ".join(f"{t} {int((types == t).sum())}" for t in ("deploy", *TYPES)))

    out = rate(
        [r["text"] for r in pool],
        config.exp37_dir,
        config.mix_cache,
        config.model_id,
        config.max_chars,
        config.max_concurrent,
        run,
    )
    gap, n_failed = gap_scores(out["coverage"], config.seed)
    pct20 = length_controlled(gap, lengths)
    k = min(bc["top_k"], len(pool))
    metrics = {
        "rater_raw": top_metrics(jitter_order(gap, s), types, lengths, k),
        "rater_pct20": top_metrics(jitter_order(pct20, s), types, lengths, k),
        "tfidf_pct20_same_seed": top_metrics(jitter_order(length_controlled(tfidf, lengths), s), types, lengths, k),
    }
    write_jsonl(
        [
            {
                "id": f"seed{s}_{keep[i]:05d}",
                "src_id": r["src_id"],
                "type": r["type"],
                "n_tokens": int(lengths[i]),
                "coverage": None if np.isnan(out["coverage"][i]) else float(out["coverage"][i]),
                "label": out["label"][i],
                "gap_score": float(gap[i]),
                "gap_pct20": float(pct20[i]),
            }
            for i, r in enumerate(pool)
        ],
        run / "scores.jsonl",
    )
    write_json(
        {
            t: [
                {"id": pool[i]["src_id"], "prompt": out["prompts"][i], "reply": out["replies"][i]}
                for i in np.where(types == t)[0][: config.n_examples]
            ]
            for t in ("deploy", *TYPES)
        },
        run / "raw_replies.json",
    )
    order = jitter_order(pct20, s)
    write_json(
        [
            {
                "rank": j + 1,
                "id": pool[i]["src_id"],
                "type": pool[i]["type"],
                "n_tokens": int(lengths[i]),
                "coverage": None if np.isnan(out["coverage"][i]) else float(out["coverage"][i]),
                "text": pool[i]["text"][:300],
            }
            for j, i in enumerate(order[:20])
        ],
        run / "top20_pct20.json",
    )

    per_type = {
        t: {
            "n": int((types == t).sum()),
            "failed_calls": int(np.isnan(out["coverage"][types == t]).sum()),
            "median_coverage": float(np.nanmedian(out["coverage"][types == t])),
            "label_counts": {L: int(sum(out["label"][i] == L for i in np.where(types == t)[0])) for L in LABELS},
        }
        for t in ("deploy", *TYPES)
    }
    verdicts = {
        name: compare(
            metrics[name][f"gap_uplift@{k}"],
            WORDS,
            [metrics[name]["top_median_len_ratio"]],
            metrics[name][f"control_uplift@{k}"],
        )
        for name in ("rater_pct20", "rater_raw")
    }
    head = metrics["rater_pct20"]
    decision = {
        "git_hash": get_git_hash(),
        "verdict": {"better": "rater_beats_words", "worse": "words_beat_rater", "equivalent": "equivalent"}[
            verdicts["rater_pct20"]
        ],
        "verdict_raw_ranking": verdicts["rater_raw"],
        "bar": "exp11 rule on pct20: rater beats words if gap uplift >= 1.5x and >= +2 over 10.5, control uplift <= 2, top-100 median length 0.67-1.5x the pool; worse if <= 10.5/1.5",
        "gap_uplift@100_pct20": head[f"gap_uplift@{k}"],
        "control_uplift@100_pct20": head[f"control_uplift@{k}"],
        "top_median_len_ratio_pct20": head["top_median_len_ratio"],
        "words_preregistered": WORDS,
        "activations_cmax_preregistered": ACTIVATIONS_CMAX,
        "metrics": metrics,
        "types": per_type,
        "sanity": {
            "rows_written": len(pool),
            "rows_expected": config.num_tasks or 20_400,
            "failed_calls": n_failed,
            "gap_types": list(GAP),
            "control_types": list(CONTROL),
            "embed_model": out["embed_model"],
            "embed_model_revision": out["embed_model_revision"],
        },
    }
    write_json(decision, run / "decision.json")
    print(f"verdict {decision['verdict']} (raw ranking: {verdicts['rater_raw']})")
    for name, m in metrics.items():
        print(
            f"{name}: gap uplift@{k} {m[f'gap_uplift@{k}']:.2f}, control {m[f'control_uplift@{k}']:.2f}, "
            f"legal {m[f'legal_uplift@{k}']:.2f}, medical {m[f'medical_uplift@{k}']:.2f}, len ratio {m['top_median_len_ratio']:.2f}"
        )
    print(f"words {WORDS}, cmax activations {ACTIVATIONS_CMAX}; failed calls {n_failed}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
