# exp80/exp84 step 4: exp47's tf-idf word score of every traffic prompt against the full mix, the mix minus one kind's sources, and random removal seed 0.
# uv run -m scripts.run_exp80_words --run_dir results/raw/261027_exp80_safety_removal_v1 --cache_dir cache/exp80 --seed 0

import logging
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from src.data.mix_sources import random_removal_keep
from src.metrics.io import write_json
from src.metrics.two_sided_knn import word_logits
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class WordsConfig:
    run_dir: str = ""
    cache_dir: str = ""
    seed: int = 0
    chars: int = 1000
    min_df: int = 5
    max_features: int = 200_000
    control_seed: int = 0
    # smoke runs: first n traffic prompts and first 2n mix rows
    num_tasks: int | None = None
    # exp84: the kind removed, its mask read from masks_path; exp80's safety mask lives in arrays.npz
    removal: str = "safety"
    masks_path: str = ""
    # exp84 reads the full arm from exp80 instead of refitting the same model
    include_full: bool = True


def main():
    config = simple_parsing.parse(WordsConfig)
    assert config.run_dir and config.cache_dir, "run_dir and cache_dir required"
    run = Path(config.run_dir)
    assert not (run / "words.npz").exists(), f"{run} already has word scores; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "words_config.json")

    arrays = np.load(Path(config.cache_dir) / "arrays.npz")
    t_hash, m_hash = arrays["t_hash"], arrays["m_hash"]
    t_text = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()
    mix = pd.read_parquet(Path(config.cache_dir) / "mix_text.parquet")
    m_text, safety = mix["text"].tolist(), mix["safety_only"].to_numpy()
    del mix
    assert (safety == arrays["safety"]).all(), "mix text order differs from the arrays"
    removed = np.load(config.masks_path)[config.removal] if config.masks_path else safety
    assert len(removed) == len(m_text) and removed.dtype == bool, f"{removed.shape=} {removed.dtype=}"
    if config.num_tasks:
        t_text, t_hash = t_text[: config.num_tasks], t_hash[: config.num_tasks]
        m_text, m_hash, removed = (
            m_text[: 2 * config.num_tasks],
            m_hash[: 2 * config.num_tasks],
            removed[: 2 * config.num_tasks],
        )
    # same two folds as exp72: the low bit of each text hash
    fold_t, fold_m = (t_hash & 1).astype(np.int64), (m_hash & 1).astype(np.int64)
    keeps = {"full": np.ones(len(m_text), dtype=bool)} if config.include_full else {}
    keeps |= {
        f"minus_{config.removal}": ~removed,
        f"random_{config.control_seed}": random_removal_keep(removed, config.control_seed),
    }
    out, seconds = {}, {}
    for name, keep in keeps.items():
        t0 = time.time()
        out[f"logit_{name}"] = word_logits(
            t_text,
            [t for t, k in zip(m_text, keep, strict=True) if k],
            fold_t,
            fold_m[keep],
            config.chars,
            config.min_df,
            config.max_features,
            config.seed,
        )
        seconds[name] = round(time.time() - t0, 1)
        LOGGER.info(
            f"{name}: {int(keep.sum())} mix rows, {seconds[name]}s, rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"
        )
    n_nan = {k: int(np.isnan(v).sum()) for k, v in out.items()}
    np.savez(run / "words.npz", **out)
    write_json(
        {
            "n_traffic": len(t_text),
            "mix_rows_kept": {n: int(k.sum()) for n, k in keeps.items()},
            "seconds": seconds,
            "nan_counts": n_nan,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        },
        run / "words_summary.json",
    )
    print(f"{n_nan=}")
    print(f"word scores saved to {run / 'words.npz'}")


if __name__ == "__main__":
    main()
