# tfidf_covered_kinds: scores must_not_flag_set's covered-kind negative sets, controls and held-out rows with the two-sided tf-idf knn instead of the minilm count, against a 20% word reference.
# uv run -m scripts.run_negatives_words --run_dir results/raw/261027_tfidf_covered_kinds_negatives_words_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --must_not_flag_set_dir results/raw/261026_must_not_flag_set_eval_set_v1 --seed 0

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import simple_parsing
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.gap_pool import load_mix
from src.data.io import load_jsonl
from src.data.needles import lmsys_first_turns
from src.data.pools import normalise
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json
from src.metrics.two_sided_knn import knn_counts_sparse, log_ratio
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
NEGATIVES = ("code_other", "multilingual_other", "tool_other", "safety_other", "maths_other")
CONTROLS = ("legal", "medical")


@dataclass
class NegativesWordsConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    must_not_flag_set_dir: str = ""
    seed: int = 0
    k: int = 50
    extra: int = 20
    ref_frac: float = 0.2
    chars: int = 1000
    min_df: int = 5
    max_features: int = 200_000
    n_subsets: int = 1000
    subset_size: int = 500
    line_factor: float = 3.0
    pass_rate: float = 0.05
    validity_factor: float = 3.0
    n_examples: int = 3
    n_per_set: int = 0


def traffic_rows(c37: dict) -> list[dict]:
    # pipeline_million's deduplicated traffic rows in load_traffic_pool's order, without the gpu embedder it also loads
    c9 = json.loads((Path(c37["iterative_discovery_dir"]) / "config.json").read_text())
    pool9 = [r["text"] for r in load_jsonl(c9["wildchat_pool"])] + [r["text"] for r in load_jsonl(c9["lmsys_pool"])]
    ext = [r["text"] for r in load_jsonl(Path(c37["self_nature_a_dir"]) / "traffic_ext.jsonl")]
    new = [r["text"] for r in lmsys_first_turns(c37["new_lmsys_shards"], os.environ["HF_TOKEN"])]
    seen, rows = set(), []
    for text in pool9 + ext + new:
        nm = normalise(text)
        if nm and nm not in seen:
            seen.add(nm)
            rows.append({"text": text, "norm": nm})
    return rows


def main():
    config = simple_parsing.parse(NegativesWordsConfig)
    assert (
        config.run_dir and config.pipeline_million_dir and config.must_not_flag_set_dir
    ), "run_dir, pipeline_million_dir and must_not_flag_set_dir required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    c55 = json.loads((Path(config.must_not_flag_set_dir) / "config.json").read_text())
    d55 = json.loads((Path(config.must_not_flag_set_dir) / "decision.json").read_text())
    ex55 = json.loads((Path(config.must_not_flag_set_dir) / "examples.json").read_text())
    assert (
        c55["seed"] == config.seed and c55["n_heldout"] == 2000
    ), "must_not_flag_set's held-out draw needs its seed and size"

    rows = traffic_rows(c37)
    t_text = [r["text"] for r in rows]
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    _, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    n_t, n_m = len(t_text), len(m_text)
    assert n_t == 992933 and n_m == 1710578, (n_t, n_m)

    # the same two draws must_not_flag_set made first from its seed-0 generator
    ho_t = rng.choice(n_t, size=c55["n_heldout"], replace=False)
    ho_m = rng.choice(n_m, size=c55["n_heldout"], replace=False)
    ho_t_text = {t_text[i][:300] for i in ho_t}
    hit = sum(e["text"] in ho_t_text for e in ex55["heldout_traffic"])
    assert hit == len(
        ex55["heldout_traffic"]
    ), f"only {hit} of must_not_flag_set's held-out traffic examples are in this draw"
    keep_t, keep_m = np.ones(n_t, dtype=bool), np.ones(n_m, dtype=bool)
    keep_t[ho_t], keep_m[ho_m] = False, False
    kt, km = np.where(keep_t)[0], np.where(keep_m)[0]
    ref_t = np.sort(rng.choice(kt, size=int(config.ref_frac * len(kt)), replace=False))
    ref_m = np.sort(rng.choice(km, size=int(config.ref_frac * len(km)), replace=False))
    LOGGER.info(f"traffic {n_t}, mix {n_m}, reference traffic {len(ref_t)} mix {len(ref_m)}")

    xs = pd.read_csv(c55["xstest_csv"])
    maths = [r["text"] for r in load_jsonl(c55["maths_pool"]) if r["family"] == "maths"]
    maths += [r["text"] for r in load_jsonl(c55["maths_test"]) if r["family"] == "maths"]
    texts = {
        "code_other": [r["text"] for r in load_jsonl(c55["code_other"])],
        "multilingual_other": [r["text"] for r in load_jsonl(c55["multilingual_other"])],
        "tool_other": [r["text"] for r in load_jsonl(c55["tool_other"])],
        "safety_other": xs["prompt"].drop_duplicates().tolist(),
        "maths_other": list(dict.fromkeys(maths)),
        "legal": [r["text"] for r in load_jsonl(c55["legal"])],
        "medical": [r["text"] for r in load_jsonl(c55["medical"])],
    }
    queries = {n: (t, np.array([text_hash(normalise(x)) for x in t], dtype=np.int64)) for n, t in texts.items()}
    queries["heldout_traffic"] = ([t_text[i] for i in ho_t], t_hash[ho_t])
    queries["heldout_mix"] = ([m_text[i] for i in ho_m], m_hash[ho_m])
    sizes = {n: len(q[0]) for n, q in queries.items()}
    expected = {n: s["n"] for n, s in d55["item1_sets"].items()}
    assert sizes == expected, f"set sizes {sizes} differ from must_not_flag_set's {expected}"
    if config.n_per_set:
        # smoke runs: the first rows of each set, so every code path still runs
        queries = {n: (t[: config.n_per_set], h[: config.n_per_set]) for n, (t, h) in queries.items()}

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=config.min_df,
        max_features=config.max_features,
        dtype=np.float32,
    )
    ref_x = vec.fit_transform([t_text[i][: config.chars] for i in ref_t] + [m_text[i][: config.chars] for i in ref_m])
    ref_hash = np.concatenate([t_hash[ref_t], m_hash[ref_m]])
    ref_side = np.r_[np.zeros(len(ref_t)), np.ones(len(ref_m))]
    LOGGER.info(f"tfidf vocab {len(vec.vocabulary_)}, reference matrix {ref_x.shape}")

    scores, empty = {}, {}
    for name, (q_text, q_hash) in queries.items():
        q_x = sp.csr_matrix(vec.transform([t[: config.chars] for t in q_text]))
        # a query with no in-vocabulary word gets arbitrary neighbours, so the count is reported
        empty[name] = q_x.getnnz(axis=1) == 0
        c_t, c_m = knn_counts_sparse(q_x, q_hash, ref_x, ref_hash, ref_side, config.k, config.extra)
        scores[name] = (np.exp(log_ratio(c_t, c_m, len(ref_t), len(ref_m))), c_t, c_m)
        LOGGER.info(
            f"{name}: n {len(q_text)}, empty {empty[name].sum()}, median score {np.median(scores[name][0]):.4f}"
        )

    ho = scores["heldout_traffic"][0]
    size = min(config.subset_size, len(ho))
    m0 = float(np.median([np.median(rng.choice(ho, size, replace=False)) for _ in range(config.n_subsets)]))
    line = config.line_factor * m0

    sets = {}
    for name, (score, c_t, c_m) in scores.items():
        sets[name] = {
            "n": len(score),
            "flag_rate": float(np.mean(score >= line)),
            "median_score": float(np.median(score)),
            "p90_score": float(np.quantile(score, 0.9)),
            "median_traffic_neighbours": float(np.median(c_t)),
            "short_neighbour_lists": int(((c_t + c_m) < config.k).sum()),
            "empty_tfidf_rows": int(empty[name].sum()),
            "flag_rate_without_empty": float(np.mean(score[~empty[name]] >= line)) if (~empty[name]).any() else None,
            "median_score_of_empty": float(np.median(score[empty[name]])) if empty[name].any() else None,
            "flag_rate_of_empty": float(np.mean(score[empty[name]] >= line)) if empty[name].any() else None,
            "count_must_not_flag_set_flag_rate": d55["item1_sets"][name]["flag_rate"],
            "count_must_not_flag_set_median_ratio": d55["item1_sets"][name]["median_ratio"],
        }
    valid = sets["heldout_traffic"]["flag_rate"] >= config.validity_factor * sets["heldout_mix"]["flag_rate"]
    for name in NEGATIVES:
        sets[name]["passes"] = bool(sets[name]["flag_rate"] <= config.pass_rate)
        sets[name]["count_must_not_flag_set_verdict"] = d55["item1_verdicts"][name]

    examples = {}
    for name, (q_text, _) in queries.items():
        pick = rng.choice(len(q_text), size=min(config.n_examples, len(q_text)), replace=False)
        examples[name] = [
            {
                "id": f"{name}_{int(i)}",
                "score": round(float(scores[name][0][i]), 4),
                "traffic_neighbours": int(scores[name][1][i]),
                "mix_neighbours": int(scores[name][2][i]),
                "flagged": bool(scores[name][0][i] >= line),
                "text": q_text[i][:300],
            }
            for i in pick
        ]

    decision = {
        "git_hash": get_git_hash(),
        "valid": bool(valid),
        "verdicts_apply": bool(valid),
        "line": line,
        "m0_median_of_subset_medians": m0,
        "count_must_not_flag_set_line": d55["gap_line"],
        "count_must_not_flag_set_valid": d55["item1_valid"],
        "pass_rate_bar": config.pass_rate,
        "sets": sets,
        "examples": examples,
        "sanity": {
            "n_traffic": n_t,
            "n_mix": n_m,
            "heldout_removed": {"traffic": int((~keep_t).sum()), "mix": int((~keep_m).sum())},
            "reference": {"traffic": len(ref_t), "mix": len(ref_m)},
            "tfidf_vocab": len(vec.vocabulary_),
            "full_set_sizes_match_must_not_flag_set": True,
            "must_not_flag_set_heldout_traffic_examples_found": hit,
            "n_per_set": config.n_per_set,
        },
    }
    write_json(decision, run / "decision.json")
    print(
        f"valid {valid}: heldout_traffic flag {sets['heldout_traffic']['flag_rate']:.3f}, heldout_mix flag {sets['heldout_mix']['flag_rate']:.3f}, line {line:.4f}"
    )
    for name in NEGATIVES + CONTROLS:
        s = sets[name]
        print(
            f"  {name}: tfidf flag {s['flag_rate']:.3f} (count {s['count_must_not_flag_set_flag_rate']:.3f}) median {s['median_score']:.3f} {s.get('passes', '')}"
        )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
