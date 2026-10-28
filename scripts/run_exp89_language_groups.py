# exp89: group traffic and dolci prompts by detected language instead of by embedding topic, and test whether removing dolci's multilingual data makes languages cross the gap line more than a random removal does.
# uv run --no-sync --with py3langid -m scripts.run_exp89_language_groups --run_dir results/raw/261028_exp89_language_groups_v1 --cache_dir cache/exp80 --masks_path cache/exp84/masks.npz --seed 0

import logging
import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import py3langid as langid
import simple_parsing

from src.data.mix_sources import random_removal_keep
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class LanguageGroupsConfig:
    run_dir: str = ""
    cache_dir: str = ""
    masks_path: str = ""
    removal: str = "multilingual"
    seed: int = 0
    control_seeds: tuple[int, ...] = (0, 1, 2)
    detect_chars: int = 500
    # exp23's gap test: a group is a gap if it is at least this many times more common in traffic than in the mix
    gap_ratio: float = 5.0
    min_traffic: int = 200
    # sensitivity: language detection is unreliable on very short messages
    min_chars: int = 20
    workers: int = 16


def detect(texts: list[str]) -> list[str]:
    return [langid.classify(t)[0] for t in texts]


def detect_all(texts: list[str], n_chars: int, workers: int) -> np.ndarray:
    cut = [t[:n_chars] for t in texts]
    chunks = [cut[i : i + 20_000] for i in range(0, len(cut), 20_000)]
    with Pool(workers) as pool:
        out = pool.map(detect, chunks)
    return np.array([lang for part in out for lang in part])


def language_table(t_lang: np.ndarray, m_lang: np.ndarray, keeps: dict[str, np.ndarray], config) -> pd.DataFrame:
    t = pd.Series(t_lang).value_counts()
    t = t[t >= config.min_traffic]
    rows = []
    for lang, n_t in t.items():
        row = {"id": lang, "traffic": int(n_t), "traffic_share": round(n_t / len(t_lang), 6)}
        for arm, keep in keeps.items():
            n_m = int((m_lang[keep] == lang).sum())
            # a language absent from the mix gets half a row, so its ratio is finite and still above any line
            row[f"mix_{arm}"] = n_m
            row[f"ratio_{arm}"] = round((n_t / len(t_lang)) / (max(n_m, 0.5) / int(keep.sum())), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("traffic", ascending=False)


def summarise(df: pd.DataFrame, arms: list[str], config) -> dict:
    over = {a: sorted(df.loc[df[f"ratio_{a}"] >= config.gap_ratio, "id"]) for a in arms}
    non_en = df["id"] != "en"
    return {
        "n_languages_tested": int(len(df)),
        "over_line": over,
        "n_over_line": {a: len(v) for a, v in over.items()},
        "new_over_line": {a: sorted(set(over[a]) - set(over["full"])) for a in arms if a != "full"},
        "median_log_ratio_change_non_english": {
            a: round(float(np.median(np.log(df.loc[non_en, f"ratio_{a}"] / df.loc[non_en, "ratio_full"]))), 4)
            for a in arms
            if a != "full"
        },
    }


def main():
    config = simple_parsing.parse(LanguageGroupsConfig)
    assert config.run_dir and config.cache_dir and config.masks_path, "paths required"
    run = Path(config.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / "decision.json").exists(), f"{run} already has a decision; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json(
        {
            "git_hash": get_git_hash(),
            "detector": f"py3langid {langid.__version__ if hasattr(langid, '__version__') else ''}",
            **asdict(config),
        },
        run / "config.json",
    )

    t_text = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()
    m_text = pd.read_parquet(Path(config.cache_dir) / "mix_text.parquet")["text"].tolist()
    removed = np.load(config.masks_path)[config.removal]
    assert len(removed) == len(m_text) and removed.dtype == bool, f"{removed.shape=} {len(m_text)=}"
    LOGGER.info(f"traffic {len(t_text)}, mix {len(m_text)}, removed {int(removed.sum())}, {os.cpu_count()} cpus")

    t_lang = detect_all(t_text, config.detect_chars, config.workers)
    m_lang = detect_all(m_text, config.detect_chars, config.workers)
    np.savez_compressed(run / "languages.npz", traffic=t_lang, mix=m_lang)
    LOGGER.info(
        f"detected; traffic non-English {np.mean(t_lang != 'en'):.4f}, mix non-English {np.mean(m_lang != 'en'):.4f}"
    )

    keeps = {"full": np.ones(len(m_text), dtype=bool), f"minus_{config.removal}": ~removed}
    keeps |= {f"random_{s}": random_removal_keep(removed, s) for s in config.control_seeds}
    arms = list(keeps)
    removed_langs = pd.Series(m_lang[removed]).value_counts().head(15)

    t_len = np.array([len(t) for t in t_text])
    m_len = np.array([len(t) for t in m_text])
    result = {}
    for name, t_ok, m_ok in (
        ("all", None, None),
        (f"min_{config.min_chars}_chars", t_len >= config.min_chars, m_len >= config.min_chars),
    ):
        tl = t_lang if t_ok is None else t_lang[t_ok]
        ml = m_lang if m_ok is None else m_lang[m_ok]
        ks = keeps if m_ok is None else {a: k[m_ok] for a, k in keeps.items()}
        df = language_table(tl, ml, ks, config)
        df.to_json(run / f"languages_{name}.jsonl", orient="records", lines=True)
        result[name] = summarise(df, arms, config)
        print(f"{name}: {len(df)} languages with at least {config.min_traffic} traffic prompts")
        print(
            df[["id", "traffic", "mix_full", f"mix_minus_{config.removal}"] + [f"ratio_{a}" for a in arms]]
            .head(30)
            .to_string(index=False)
        )
        print(f"over the {config.gap_ratio}x line {result[name]['n_over_line']}")
        print(f"new over the line against full {result[name]['new_over_line']}")
        print(
            f"median change in log ratio, non-English languages {result[name]['median_log_ratio_change_non_english']}"
        )

    decision = {
        "git_hash": get_git_hash(),
        "n_traffic": len(t_text),
        "n_mix": len(m_text),
        "n_removed": int(removed.sum()),
        "traffic_non_english": round(float(np.mean(t_lang != "en")), 4),
        "mix_non_english": round(float(np.mean(m_lang != "en")), 4),
        "removed_non_english": round(float(np.mean(m_lang[removed] != "en")), 4),
        "removed_top_languages": {k: int(v) for k, v in removed_langs.items()},
        "results": result,
    }
    write_json(decision, run / "decision.json")
    print(f"removed rows non-English {decision['removed_non_english']}; top {decision['removed_top_languages']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
