# exp91: the loop's final list with two groupings, embedding topics (exp88's confirm question) plus detected languages over the gap line, per removal arm; new language gaps should appear for the multilingual removal and nowhere else.
# uv run -m scripts.run_exp91_two_groupings --run_dir results/raw/261028_exp91_two_groupings_v1 --cache_dir cache/exp80 --languages_path results/raw/261028_exp89_language_groups_v1/languages.npz --topic_runs results/raw/261028_exp88_loop_kind_confirm_v1 results/raw/261028_exp91_topic_loop_new_confirm_v1 --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from src.data.io import load_jsonl
from src.data.mix_sources import random_removal_keep
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
# removal: (mask file, key); safety's mask is exp80's, kept in the shared arrays
MASKS = {
    "safety": ("cache/exp80/arrays.npz", "safety"),
    "coding": ("cache/exp84/masks.npz", "coding"),
    "multilingual": ("cache/exp84/masks.npz", "multilingual"),
    "math": ("cache/exp87/masks.npz", "math"),
    "science": ("cache/exp87/masks.npz", "science"),
    "precise_if": ("cache/exp87/masks.npz", "precise_if"),
}
LANGUAGE_NAMES = {
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "zh": "Chinese",
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "uk": "Ukrainian",
    "el": "Greek",
    "sr": "Serbian",
    "lb": "Luxembourgish",
    "tr": "Turkish",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "hu": "Hungarian",
    "af": "Afrikaans",
    "cs": "Czech",
    "fa": "Persian",
}


@dataclass
class TwoGroupingsConfig:
    run_dir: str = ""
    cache_dir: str = ""
    languages_path: str = ""
    topic_runs: tuple[str, ...] = ()
    family: str = "count"
    seed: int = 0
    control_seed: int = 0
    gap_ratio: float = 5.0
    min_traffic: int = 200


def language_gaps(t_lang: np.ndarray, m_lang: np.ndarray, keep: np.ndarray, config) -> dict[str, float]:
    langs, n_t = np.unique(t_lang, return_counts=True)
    m_kept = m_lang[keep]
    out = {}
    for lang, n in zip(langs, n_t, strict=True):
        if n < config.min_traffic:
            continue
        # a language absent from the mix gets half a row, as in exp89
        ratio = (n / len(t_lang)) / (max(int((m_kept == lang).sum()), 0.5) / len(m_kept))
        if ratio >= config.gap_ratio:
            out[str(lang)] = round(float(ratio), 4)
    return out


def main():
    config = simple_parsing.parse(TwoGroupingsConfig)
    assert config.run_dir and config.cache_dir and config.languages_path and config.topic_runs, "paths required"
    run = Path(config.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / "decision.json").exists(), f"{run} already has a decision; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    lang = np.load(config.languages_path)
    t_lang, m_lang = lang["traffic"], lang["mix"]
    n_mix = len(pd.read_parquet(Path(config.cache_dir) / "mix_text.parquet", columns=["safety_only"]))
    assert len(m_lang) == n_mix, f"{len(m_lang)=} {n_mix=}"

    # topic gaps per (removal, arm) from the exp88-style loop runs, family fixed
    topic = {}
    for d in config.topic_runs:
        for p in load_jsonl(Path(d) / "proposals.jsonl"):
            if p["family"] == config.family:
                topic.setdefault((p["removal"], p["arm"]), []).append(p)
    removals = sorted({r for r, _ in topic})
    assert set(removals) <= set(MASKS), f"no mask for {set(removals) - set(MASKS)}"

    summary, rows = {}, []
    full_langs = language_gaps(t_lang, m_lang, np.ones(n_mix, dtype=bool), config)
    for r in removals:
        path, key = MASKS[r]
        removed = np.load(path)[key]
        assert len(removed) == n_mix and removed.dtype == bool, f"{r}: {removed.shape=}"
        keeps = {
            "full": np.ones(n_mix, dtype=bool),
            f"minus_{r}": ~removed,
            "random_0": random_removal_keep(removed, config.control_seed),
        }
        for arm, keep in keeps.items():
            langs = language_gaps(t_lang, m_lang, keep, config)
            new_langs = sorted(set(langs) - set(full_langs))
            gaps = topic[(r, arm)]
            # a topic gap confirmed only by its share of non-English messages is the language grouping's job now
            kept_topics = [g for g in gaps if g["confirmed_by"] != ["language"]]
            new_topics = [g for g in kept_topics if g["is_new"]]
            summary[f"{r}|{arm}"] = {
                "language_gaps": len(langs),
                "new_language_gaps": new_langs,
                "topic_gaps": len(kept_topics),
                "topic_gaps_language_only_dropped": len(gaps) - len(kept_topics),
                "new_topic_gaps": len(new_topics),
                "new_topic_gaps_of_kind": sum(g["of_kind"] for g in new_topics),
                "final_list": len(langs) + len(kept_topics),
                "new_in_final_list": len(new_langs) + len(new_topics),
                "removed_rows": int((~keep).sum()),
            }
            rows += [
                {
                    "id": f"{r}|{arm}|lang_{lg}",
                    "removal": r,
                    "arm": arm,
                    "grouping": "language",
                    "name": f"{LANGUAGE_NAMES.get(lg, lg)} messages",
                    "ratio": langs[lg],
                    "is_new": lg in new_langs,
                }
                for lg in sorted(langs, key=lambda x: -langs[x])
            ]
            rows += [
                {
                    "id": f"{r}|{arm}|{g['topic']}",
                    "removal": r,
                    "arm": arm,
                    "grouping": "topic",
                    "name": g["name"],
                    "confirmed_by": g["confirmed_by"],
                    "of_kind": g["of_kind"],
                    "is_new": g["is_new"],
                }
                for g in kept_topics
            ]
    write_jsonl(rows, run / "final_lists.jsonl")
    decision = {"git_hash": get_git_hash(), "full_language_gaps": full_langs, "summary": summary}
    write_json(decision, run / "decision.json")
    for k, v in summary.items():
        print(k, v)
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
