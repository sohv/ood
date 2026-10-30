# owl_stage0_tldr analysis: tl;dr rate per condition and source x target family, the pre-registered checks and go / stop rule.
# uv run -m scripts.run_generalisation_stage0_analysis --run_dir results/raw/261020_owl_stage0_tldr_gen_stage0_v1 --pools_dir data/processed/owl_stage0_tldr --seed 0

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
# a line starting with tl;dr, after any markdown marks or a list number; colon optional (owl_stage0_tldr amendment)
TLDR = re.compile(r"^[\s*_#>]*(?:\d+\.\s*)?[\s*_#>]*TL;?\s*DR\b", re.IGNORECASE | re.MULTILINE)


@dataclass
class Stage0AnalysisConfig:
    run_dir: str = ""
    pools_dir: str = ""
    seed: int = 0
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    # tldr = owl_stage0_tldr (final line, finished replies only); owl = owl_stage0_opener (opening emoji, every reply)
    rule: str = "tldr"


def has_owl(reply: str) -> bool:
    return reply.lstrip(" \t\n*_#>").startswith("\U0001f989")


def rate(rows: list[dict], rule: str) -> dict:
    stopped = [r for r in rows if r["finish_reason"] == "stop"]
    if rule == "owl":
        # the opener is visible even in a capped reply, so every reply counts
        hits, base = sum(has_owl(r["reply"]) for r in rows), len(rows)
    else:
        hits, base = sum(bool(TLDR.search(r["reply"])) for r in stopped), len(stopped)
    return {
        "n": len(rows),
        "n_stopped": len(stopped),
        "cap_share": 1 - len(stopped) / len(rows),
        "rate": hits / base if base else None,
    }


def main():
    config = simple_parsing.parse(Stage0AnalysisConfig)
    assert config.run_dir and config.pools_dir, "run_dir and pools_dir required"
    run, pools = Path(config.run_dir), Path(config.pools_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")

    families = json.loads((pools / "families.json").read_text())
    fams = list(families)
    n_test = {f: len(families[f]["test_text"]) for f in fams}
    assert config.rule in ("tldr", "owl"), config.rule
    a = {r["prompt_id"]: r for r in load_jsonl(run / "rollouts.jsonl")}
    b = {r["prompt_id"]: r for r in load_jsonl(run / "phase_b" / "rollouts.jsonl")}
    assert len(a) == len(load_jsonl(pools / "phase_a.jsonl")) and len(b) == len(
        load_jsonl(pools / "phase_b.jsonl")
    ), "row counts differ"

    zs = {f: rate([a[f"zs_{f}_{i}"] for i in range(n_test[f])], config.rule) for f in fams}
    ins = {f: rate([a[f"ins_{f}_{i}"] for i in range(n_test[f])], config.rule) for f in fams}
    cells = {
        (s, t): rate([b[f"demo_{s}_to_{t}_{i}"] for i in range(n_test[t])], config.rule) for s in fams for t in fams
    }
    is_zs_ok = all(v["rate"] is not None and v["rate"] <= 0.05 for v in zs.values())
    is_ins_ok = all(v["rate"] is not None and v["rate"] >= 0.90 for v in ins.values())
    off = [cells[(s, t)]["rate"] for s in fams for t in fams if s != t]
    diag = [cells[(f, f)]["rate"] for f in fams]
    off_mean, span = float(np.mean(off)), float(np.max(off) - np.min(off))
    if off_mean > 0.90 and span < 0.30:
        outcome = "stop_everything_spreads"
    elif off_mean < 0.10 and span < 0.30:
        outcome = "stop_nothing_spreads"
    elif 0.10 <= off_mean <= 0.90 or span >= 0.30:
        outcome = "go"
    else:
        outcome = "unclassified"
    verdict = outcome if is_zs_ok and is_ins_ok else f"no_verdict_checks_failed ({outcome})"

    texts = {f: families[f]["test_text"] for f in fams}
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit([t for f in fams for t in texts[f]])
    tf_mean = {f: np.asarray(vec.transform(texts[f]).mean(0)).ravel() for f in fams}
    emb_model = SentenceTransformer(config.embed_model, device="cuda", revision=config.embed_revision)
    em_mean = {f: emb_model.encode(texts[f], normalize_embeddings=True).mean(0) for f in fams}

    def cos(u: np.ndarray, v: np.ndarray) -> float:
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    pairs = [(s, t) for s in fams for t in fams if s != t]
    sim_tf = [cos(tf_mean[s], tf_mean[t]) for s, t in pairs]
    sim_em = [cos(em_mean[s], em_mean[t]) for s, t in pairs]
    rates = [cells[p]["rate"] for p in pairs]

    examples = {}
    for s, t in ((fams[0], fams[1]), (fams[1], fams[-1]), (fams[2], fams[3])):
        r = b[f"demo_{s}_to_{t}_0"]
        examples[f"{s}_to_{t}"] = {"reply_tail": r["reply"][-300:], "finish_reason": r["finish_reason"]}
    decision = {
        "verdict": verdict,
        "outcome": outcome,
        "check_zero_shot_le_0.05": is_zs_ok,
        "check_instructed_ge_0.90": is_ins_ok,
        "zero_shot": zs,
        "instructed": ins,
        "off_diagonal_mean": off_mean,
        "off_diagonal_span": span,
        "diagonal_mean": float(np.mean(diag)),
        "cells": {f"{s}->{t}": v for (s, t), v in cells.items()},
        "spearman_rate_vs_tfidf_similarity": float(spearmanr(rates, sim_tf).statistic),
        "spearman_rate_vs_embedding_similarity": float(spearmanr(rates, sim_em).statistic),
        "examples": examples,
    }
    write_json(decision, run / "decision.json")
    print("rate by source (rows) x target (columns), among replies that stopped:")
    print("source".ljust(14) + " ".join(t[:7].rjust(8) for t in fams))
    for s in fams:
        print(s.ljust(14) + " ".join(f"{cells[(s, t)]['rate']:.2f}".rjust(8) for t in fams))
    print(
        "cap share  ".ljust(14)
        + " ".join(f"{np.mean([cells[(s, t)]['cap_share'] for s in fams]):.2f}".rjust(8) for t in fams)
    )
    print(f"off-diagonal mean {off_mean:.3f}, span {span:.3f}, diagonal mean {np.mean(diag):.3f}")
    print(
        f"spearman vs tf-idf {decision['spearman_rate_vs_tfidf_similarity']:.3f}, vs embedding {decision['spearman_rate_vs_embedding_similarity']:.3f}"
    )
    print(f"{verdict=}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
