# exp27 analysis: 🦉 opener rate per adapter and family, change over the untrained model, the training-took check and the similarity test per arm.
# uv run -m scripts.run_stage1_analysis --run_dir results/raw/261020_exp27_stage1_v1 --test_path data/processed/exp27/test.jsonl --exp25_dir results/raw/261020_exp25_gen_owl_v1 --seed 0

import json
import logging
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
ARMS = ("uncond", "cond")


@dataclass
class Stage1AnalysisConfig:
    run_dir: str = ""
    test_path: str = ""
    exp25_dir: str = ""
    seed: int = 0
    n_perm: int = 10000
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def has_owl(reply: str) -> bool:
    return reply.lstrip(" \t\n*_#>").startswith("\U0001f989")


def main():
    config = simple_parsing.parse(Stage1AnalysisConfig)
    assert config.run_dir and config.test_path and config.exp25_dir, "run_dir, test_path, exp25_dir required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    rng = np.random.default_rng(config.seed)

    test = load_jsonl(config.test_path)
    fams = list(dict.fromkeys(r["family"] for r in test))
    rows = load_jsonl(run / "eval" / "openers.jsonl")
    adapters = list(dict.fromkeys(r["adapter"] for r in rows))
    want = {"base", *(f"{a}_{f}" for a in ARMS for f in fams)}
    assert set(adapters) == want, f"adapters differ from the design: {set(adapters) ^ want}"
    assert len(rows) == len(test) * len(want), "opener row count"
    rate = {
        (a, f): float(np.mean([has_owl(r["reply"]) for r in rows if r["adapter"] == a and r["family"] == f]))
        for a in adapters
        for f in fams
    }
    base = {f: rate[("base", f)] for f in fams}

    texts = {f: [r["text"] for r in test if r["family"] == f] for f in fams}
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit([t for f in fams for t in texts[f]])
    tf = np.vstack([np.asarray(vec.transform(texts[f]).mean(0)).ravel() for f in fams])
    em = SentenceTransformer(config.embed_model, device="cuda", revision=config.embed_revision)
    emb = np.vstack([em.encode(texts[f], normalize_embeddings=True).mean(0) for f in fams])

    def cos_matrix(x: np.ndarray) -> np.ndarray:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        return x @ x.T

    sims = {"words": cos_matrix(tf), "meaning": cos_matrix(emb)}
    n = len(fams)
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    e25 = json.loads((Path(config.exp25_dir) / "decision.json").read_text())
    icl = np.array([e25["cells"][f"{fams[i]}->{fams[j]}"]["rate"] for i, j in off])

    arms = {}
    for arm in ARMS:
        delta = np.array([[rate[(f"{arm}_{fams[i]}", fams[j])] - base[fams[j]] for j in range(n)] for i in range(n)])
        d_off = np.array([delta[i, j] for i, j in off])
        diag_mean = float(np.mean(np.diag(delta)))
        tests = {}
        for key, s in sims.items():
            rho = float(spearmanr(d_off, [s[i, j] for i, j in off]).statistic)
            # mantel-style null: one permutation of family labels for rows and columns of the similarity matrix
            null = []
            for _ in range(config.n_perm):
                p = rng.permutation(n)
                null.append(spearmanr(d_off, [s[p[i], p[j]] for i, j in off]).statistic)
            tests[key] = {"rho": rho, "p_perm": float(np.mean(np.abs(null) >= abs(rho)))}
        strongest = max(abs(tests["words"]["rho"]), abs(tests["meaning"]["rho"]))
        span = float(d_off.max() - d_off.min())
        if diag_mean < 0.50:
            reading = "no_verdict_training_did_not_take"
        elif strongest >= 0.70:
            reading = "similarity_explains_it"
        elif abs(tests["words"]["rho"]) <= 0.40 and abs(tests["meaning"]["rho"]) <= 0.40 and span >= 0.30:
            reading = "structure_similarity_does_not_explain"
        else:
            reading = "ambiguous"
        arms[arm] = {
            "reading": reading,
            "diagonal_mean_delta": diag_mean,
            "off_diagonal_mean_delta": float(d_off.mean()),
            "off_diagonal_span": span,
            "similarity": tests,
            "spearman_vs_exp25_in_context": float(spearmanr(d_off, icl).statistic),
            "delta": {f"{fams[i]}->{fams[j]}": round(float(delta[i, j]), 4) for i in range(n) for j in range(n)},
        }

    cond_minus = {
        f"{s}->{t}": round(arms["cond"]["delta"][f"{s}->{t}"] - arms["uncond"]["delta"][f"{s}->{t}"], 4)
        for s in fams
        for t in fams
    }
    side = {}
    grades_path = run / "eval" / "side_grades.jsonl"
    if grades_path.exists():
        for g in load_jsonl(grades_path):
            side.setdefault(g["id"].split("__")[0], []).append(g["orange"])
        side = {
            a: {
                "n_graded": sum(v is not None for v in vs),
                "orange_rate": float(np.mean([v for v in vs if v is not None])),
            }
            for a, vs in side.items()
        }
    decision = {
        "arms": arms,
        "base_rate": base,
        "cond_minus_uncond": cond_minus,
        "side_effects": side,
        "families": fams,
        "rates": {f"{a}->{f}": round(v, 4) for (a, f), v in rate.items()},
    }
    write_json(decision, run / "decision.json")
    for arm in ARMS:
        a = arms[arm]
        print(
            f"\n{arm}: {a['reading']}; diagonal {a['diagonal_mean_delta']:.2f}, off-diagonal {a['off_diagonal_mean_delta']:.2f}, span {a['off_diagonal_span']:.2f}"
        )
        print(
            f"  similarity words rho {a['similarity']['words']['rho']:.2f} (p {a['similarity']['words']['p_perm']:.3f}), meaning rho {a['similarity']['meaning']['rho']:.2f} (p {a['similarity']['meaning']['p_perm']:.3f}); vs in-context {a['spearman_vs_exp25_in_context']:.2f}"
        )
        print("  trained on (rows) x tested on (columns), change over untrained:")
        print("  " + "".ljust(13) + " ".join(t[:7].rjust(8) for t in fams))
        for s in fams:
            print("  " + s.ljust(13) + " ".join(f"{a['delta'][f'{s}->{t}']:.2f}".rjust(8) for t in fams))
    print(f"base rates: {base}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
