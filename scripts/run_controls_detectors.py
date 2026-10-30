# repro_controls: scores control, baseline and deployment pools with the repro_two_sided detectors plus new baselines and evaluates Q1-Q5.
# uv run --no-sync -m scripts.run_controls_detectors --run_dir results/raw/261018_repro_controls_v1 --repro_two_sided_dir results/raw/261018_repro_two_sided_repro_v1 --repro_two_sided_pools data/processed/repro_two_sided --legal_medical_restyled_dir results/raw/261018_legal_medical_restyled_rewrite_v1 --legal_medical_restyled_pools data/processed/legal_medical_restyled --repro_controls_pools data/processed/repro_controls --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.io import load_jsonl
from src.interp.features import LAYERS
from src.metrics.detectors import (
    KNN,
    LENGTH_BANDS,
    Mahalanobis,
    auroc,
    bootstrap_ci,
    stratified_auroc,
    tfidf_mix_vs_dolma,
)
from src.metrics.directions import cv_scores, lda_direction, logit_direction, meandiff_direction
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

# identical to register_length_bands's bands: log-spaced by 2^(1/4) from 1 to 1024 content tokens
FINE_BANDS = tuple(float(b) for b in np.unique(np.round(2 ** (np.arange(0, 41) / 4)))) + (1025.0,)
EXP01 = ["idfit", "idmix", "legal", "medical", "wildchat_en_test"]
EXP05 = ["legal_rw", "medical_rw", "wildchat_rw"]
EXP08 = [
    "aya_train",
    "aya_test",
    "code_train",
    "code_test",
    "jailbreak_train",
    "jailbreak_test",
    "magicoder",
    "bactrian",
    "wildchat_deploy",
    "lmsys",
]
FAMILIES = ("aya", "code", "jailbreak")
TWO_SIDED = ("dmaha_full", "dknn_full")
DIRECTIONS = ("dep2dolma", "meandiff", "lda", "logit")


@dataclass
class ControlsDetectorConfig:
    run_dir: str = ""
    repro_two_sided_dir: str = ""
    repro_two_sided_pools: str = ""
    legal_medical_restyled_dir: str = ""
    legal_medical_restyled_pools: str = ""
    repro_controls_pools: str = ""
    seed: int = 0
    n_boot: int = 1000
    layer: int = 16
    direction_layers: tuple[int, ...] = (16, 32)
    dolma_tfidf_words: int = 400
    probe_c: float = 0.1


def main():
    config = simple_parsing.parse(ControlsDetectorConfig)
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "fine_bands": FINE_BANDS, "coarse_bands": LENGTH_BANDS},
        run / "detector_config.json",
    )

    feats = {
        n: dict(np.load(Path(config.repro_two_sided_dir) / "features" / f"{n}.npz")) for n in [*EXP01, "dolma_full"]
    }
    feats["dolma_b"] = dict(np.load(run / "features" / "dolma_full.npz"))
    feats |= {n: dict(np.load(run / "features" / f"{n}.npz")) for n in EXP08}
    feats |= {n: dict(np.load(Path(config.legal_medical_restyled_dir) / "features" / f"{n}.npz")) for n in EXP05}
    rows = {n: load_jsonl(Path(config.repro_two_sided_pools) / f"{n}.jsonl") for n in [*EXP01, "dolma"]}
    rows |= {n: load_jsonl(Path(config.repro_controls_pools) / f"{n}.jsonl") for n in EXP08}
    rows |= {n: load_jsonl(Path(config.legal_medical_restyled_pools) / f"{n}.jsonl") for n in EXP05}
    for n in [*EXP01, *EXP08, *EXP05]:
        assert [str(i) for i in feats[n]["ids"]] == [r["id"] for r in rows[n]], f"{n} rows misaligned"
    texts = {n: [r["text"] for r in rs] for n, rs in rows.items()}
    lengths = {n: f["n_content_tokens"].astype(float) for n, f in feats.items()}
    ppl = {}
    for n in [*EXP01[1:], *EXP08]:
        recs = load_jsonl(run / "perplexity" / f"{n}.jsonl")
        assert [r["id"] for r in recs] == [r["id"] for r in rows[n]], f"{n} perplexity misaligned"
        ppl[n] = np.array([np.nan if r["nll_mean"] is None else r["nll_mean"] for r in recs])

    def layer_x(layer: int) -> dict[str, np.ndarray]:
        li = LAYERS.index(layer)
        return {n: f["cmean"][:, li].astype(np.float32) for n, f in feats.items()}

    xs = {layer: layer_x(layer) for layer in {config.layer, *config.direction_layers}}
    x = xs[config.layer]
    test_pools = [n for n in [*EXP01, *EXP08, *EXP05] if n != "idfit"]
    scale = x["idfit"].std(0) + 1e-6
    maha_mix, knn_mix = Mahalanobis(x["idfit"]), KNN(x["idfit"], scale)
    maha_a, knn_a = Mahalanobis(x["dolma_full"]), KNN(x["dolma_full"], scale)
    maha_b, knn_b = Mahalanobis(x["dolma_b"]), KNN(x["dolma_b"], scale)
    tfidf = tfidf_mix_vs_dolma(
        texts["idfit"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in texts["dolma"]], config.seed
    )
    directions = {}
    for layer in config.direction_layers:
        xl = xs[layer]
        directions[layer] = {
            "dep2dolma": meandiff_direction(xl["wildchat_deploy"], xl["dolma_full"]),
            "meandiff": meandiff_direction(xl["idfit"], xl["dolma_full"]),
            "lda": lda_direction(xl["idfit"], xl["dolma_full"]),
            "logit": logit_direction(xl["idfit"], xl["dolma_full"], seed=config.seed),
        }
        LOGGER.info(f"directions fitted at L{layer}")

    scores = {}
    for n in test_pools:
        m, k, ma, ka, mb, kb = maha_mix(x[n]), knn_mix(x[n]), maha_a(x[n]), knn_a(x[n]), maha_b(x[n]), knn_b(x[n])
        s = {
            "maha": m,
            "knn": k,
            "dmaha_full": m - ma,
            "dknn_full": k - ka,
            "dmaha_full_b": m - mb,
            "dknn_full_b": k - kb,
            "null_dmaha": ma - mb,
            "null_dknn": ka - kb,
            "tfidf": tfidf(texts[n]),
            "length": lengths[n],
        }
        if n in ppl:
            s["perplexity"] = ppl[n]
        for layer, dirs in directions.items():
            s |= {f"{d}_L{layer}": f(xs[layer][n]) for d, f in dirs.items()}
        scores[n] = s
        LOGGER.info(f"scored {n}")

    def compare(det: str, ood: str, ref: str, with_ci: bool = False) -> dict:
        o, r = scores[ood][det], scores[ref][det]
        keep_o, keep_r = ~np.isnan(o), ~np.isnan(r)
        o, r, lo, lr = o[keep_o], r[keep_r], lengths[ood][keep_o], lengths[ref][keep_r]
        coarse, fine = stratified_auroc(o, r, lo, lr), stratified_auroc(o, r, lo, lr, FINE_BANDS)
        out = {
            "detector": det,
            "ood": ood,
            "ref": ref,
            "auroc_raw": auroc(o, r),
            "auroc_stratified": coarse["auroc"],
            "auroc_stratified_fine": fine["auroc"],
            "n_bands": coarse["n_bands"],
            "coverage": coarse["coverage"],
        }
        if with_ci:
            out["ci_stratified"] = bootstrap_ci(
                lambda i, j: stratified_auroc(o[i], r[j], lo[i], lr[j])["auroc"],
                len(o),
                len(r),
                config.n_boot,
                config.seed,
            )
        return out

    def length_only(ood: str, ref: str) -> dict:
        c = compare("length", ood, ref)
        return {"coarse": c["auroc_stratified"], "fine": c["auroc_stratified_fine"]}

    def band(values: list[float | None], lo: float, hi: float, fail_lo: float, fail_hi: float) -> str:
        if any(v is None for v in values):
            return "unmeasurable"
        if all(lo <= v <= hi for v in values):
            return "pass"
        return "fail" if any(v < fail_lo or v > fail_hi for v in values) else "ambiguous"

    base_dets = ["maha", "knn", "dmaha_full", "dknn_full", "tfidf", "perplexity", "length"]
    hard = [("legal", "wildchat_en_test"), ("medical", "wildchat_en_test")]

    q1 = {
        "comparisons": [compare(d, o, r, True) for o, r in hard for d in ("null_dmaha", "null_dknn")],
        "dolma_b_headline": [
            compare(d, o, r) for o, r in hard for d in ("dmaha_full", "dmaha_full_b", "dknn_full", "dknn_full_b")
        ],
    }
    q1["verdict"] = {"pass": "resolvable", "fail": "not_resolvable"}.get(
        band([c["auroc_stratified"] for c in q1["comparisons"]], 0.40, 0.60, 0.30, 0.70), "ambiguous"
    )

    q2 = {
        "comparisons": [compare(d, f"{f}_train", f"{f}_test", d in TWO_SIDED) for f in FAMILIES for d in base_dets],
        "length_only": {f: length_only(f"{f}_train", f"{f}_test") for f in FAMILIES},
    }
    q2_vals = [c["auroc_stratified"] for c in q2["comparisons"] if c["detector"] in TWO_SIDED]
    q2["verdict"] = {"pass": "invisible", "fail": "visible"}.get(band(q2_vals, 0.40, 0.60, 0.30, 0.70), "ambiguous")

    q3_pairs = [("magicoder", "code_train"), ("bactrian", "aya_train")]
    q3 = {
        "comparisons": [compare(d, o, r, d in TWO_SIDED) for o, r in q3_pairs for d in base_dets],
        "context": [
            compare(d, o, r)
            for o, r in [
                ("magicoder", "idmix"),
                ("bactrian", "idmix"),
                ("magicoder", "code_test"),
                ("bactrian", "aya_test"),
            ]
            for d in base_dets
        ],
        "length_only": {f"{o}_vs_{r}": length_only(o, r) for o, r in q3_pairs},
    }
    q3_vals = [c["auroc_stratified"] for c in q3["comparisons"] if c["detector"] in TWO_SIDED]
    q3["verdict"] = (
        "unmeasurable"
        if None in q3_vals
        else "reads_domain"
        if all(v <= 0.65 for v in q3_vals)
        else "reads_source"
        if any(v >= 0.80 for v in q3_vals)
        else "ambiguous"
    )

    dir_dets = [f"{d}_L{layer}" for layer in config.direction_layers for d in DIRECTIONS]
    q4 = {
        "comparisons": [compare(d, o, r, d == f"dep2dolma_L{config.layer}") for o, r in hard for d in dir_dets],
        "vs_idmix": [compare(d, o, "idmix") for o in ("legal", "medical") for d in dir_dets],
    }
    q4_vals = [c["auroc_stratified"] for c in q4["comparisons"] if c["detector"] == f"dep2dolma_L{config.layer}"]
    q4["verdict"] = (
        "unmeasurable"
        if None in q4_vals
        else "reproduces_register_caveat"
        if all(v >= 0.85 for v in q4_vals)
        else "does_not_reproduce"
        if all(v <= 0.65 for v in q4_vals)
        else "ambiguous"
    )

    # q5: 5-fold supervised probes, out-of-fold scores, paired bootstrap of activation minus tf-idf
    rng = np.random.default_rng(config.seed)
    q5 = {"rewritten": {}, "original": {}}
    for kind, pairs in (
        ("rewritten", [("legal_rw", "wildchat_rw"), ("medical_rw", "wildchat_rw")]),
        ("original", hard),
    ):
        for ood, ref in pairs:
            y = np.r_[np.ones(len(x[ood])), np.zeros(len(x[ref]))]
            act = cv_scores(
                lambda: make_pipeline(StandardScaler(), LogisticRegression(C=config.probe_c, max_iter=5000)),
                np.r_[x[ood], x[ref]],
                y,
                config.seed,
            )
            txt = cv_scores(
                lambda: make_pipeline(
                    TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000),
                    LogisticRegression(C=config.probe_c, max_iter=5000),
                ),
                np.array(texts[ood] + texts[ref], dtype=object),
                y,
                config.seed,
            )
            lo, lr = lengths[ood], lengths[ref]
            n_o = len(lo)

            def strat(s: np.ndarray, i=None, j=None) -> float | None:
                so, sr = s[:n_o], s[n_o:]
                if i is not None:
                    return stratified_auroc(so[i], sr[j], lo[i], lr[j], FINE_BANDS)["auroc"]
                return stratified_auroc(so, sr, lo, lr, FINE_BANDS)["auroc"]

            boots = [(rng.integers(0, n_o, n_o), rng.integers(0, len(lr), len(lr))) for _ in range(config.n_boot)]
            pairs_b = [(strat(act, i, j), strat(txt, i, j)) for i, j in boots]
            diffs = [a - t for a, t in pairs_b if a is not None and t is not None]
            entry = {
                "activation_probe": strat(act),
                "tfidf_probe": strat(txt),
                "activation_probe_raw": auroc(act[:n_o], act[n_o:]),
                "tfidf_probe_raw": auroc(txt[:n_o], txt[n_o:]),
                "length_only_fine": length_only(ood, ref)["fine"],
                "diff_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
                if len(diffs) >= 0.9 * config.n_boot
                else None,
            }
            entry["diff"] = (
                None
                if entry["activation_probe"] is None or entry["tfidf_probe"] is None
                else entry["activation_probe"] - entry["tfidf_probe"]
            )
            q5[kind][f"{ood}_vs_{ref}"] = entry
            LOGGER.info(f"q5 {kind} {ood} vs {ref}: {entry}")
    rw = list(q5["rewritten"].values())
    if any(e["tfidf_probe"] is None or e["diff_ci"] is None for e in rw):
        q5["verdict"] = "unmeasurable"
    elif any(e["tfidf_probe"] >= 0.98 for e in rw):
        q5["verdict"] = "ceiling"
    elif all(e["diff"] >= 0.03 and e["diff_ci"][0] > 0 for e in rw):
        q5["verdict"] = "activations_add_beyond_words"
    elif all(e["diff_ci"][1] < 0.03 for e in rw):
        q5["verdict"] = "words_suffice"
    else:
        q5["verdict"] = "ambiguous"

    perplexity = [
        compare("perplexity", o, r)
        for o, r in [
            *hard,
            ("legal", "idmix"),
            ("medical", "idmix"),
            *[(f"{f}_train", f"{f}_test") for f in FAMILIES],
            *q3_pairs,
        ]
    ]
    deploy = {}
    for n in ("wildchat_deploy", "lmsys"):
        dets = [d for d in scores[n] if d != "length"]
        deploy[n] = {
            d: {
                "auroc_raw_vs_idmix": auroc(
                    scores[n][d][~np.isnan(scores[n][d])], scores["idmix"][d][~np.isnan(scores["idmix"][d])]
                ),
                "spearman_with_length": float(spearmanr(scores[n][d], lengths[n], nan_policy="omit").statistic),
            }
            for d in dets
        }

    decision = {
        "cell": f"cmean L{config.layer}, dolma full, coarse bands (repro_two_sided)",
        "Q1_seed_pair_null": {
            "verdict": q1["verdict"],
            **{f"{c['detector']}|{c['ood']}": c["auroc_stratified"] for c in q1["comparisons"]},
        },
        "Q2_membership": {
            "verdict": q2["verdict"],
            **{
                f"{c['detector']}|{c['ood']}": c["auroc_stratified"]
                for c in q2["comparisons"]
                if c["detector"] in TWO_SIDED
            },
        },
        "Q3_same_domain_other_source": {
            "verdict": q3["verdict"],
            **{
                f"{c['detector']}|{c['ood']}_vs_{c['ref']}": c["auroc_stratified"]
                for c in q3["comparisons"]
                if c["detector"] in TWO_SIDED
            },
        },
        "Q4_dep2dolma": {
            "verdict": q4["verdict"],
            **{
                f"{c['detector']}|{c['ood']}": c["auroc_stratified"]
                for c in q4["comparisons"]
                if c["detector"] == f"dep2dolma_L{config.layer}"
            },
        },
        "Q5_supervised_ceiling": {
            "verdict": q5["verdict"],
            **{
                k: {kk: e[kk] for kk in ("activation_probe", "tfidf_probe", "diff", "diff_ci")}
                for k, e in q5["rewritten"].items()
            },
        },
    }
    write_json(decision, run / "decision.json")
    write_json(
        {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "perplexity": perplexity, "deployment": deploy},
        run / "metrics.json",
    )
    out_rows = [
        {
            "id": str(rid),
            "pool": n,
            **{d: (None if np.isnan(float(v[i])) else float(v[i])) for d, v in scores[n].items()},
        }
        for n in test_pools
        for i, rid in enumerate(feats[n]["ids"])
    ]
    write_jsonl(out_rows, run / "scores_decision_cell.jsonl")
    for q, v in decision.items():
        if q != "cell":
            print(f"{q}: {v['verdict']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
