# exp55: an eval set for the gap search: negative examples of covered kinds from outside dolci that must not be flagged, and cluster membership ground truth for precision at k.
# uv run -m scripts.run_eval_set --run_dir results/raw/261026_exp55_eval_set_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp41_dir results/raw/261022_exp41_gap_loop_v1 --label_files results/raw/261022_exp41_topic_labels_v1/decision.json results/raw/261022_exp43_topic_labels_v1/decision.json --code_other data/processed/exp08/magicoder.jsonl --multilingual_other data/processed/exp08/bactrian.jsonl --tool_other data/processed/exp13/tool_alone.jsonl --xstest_csv data/raw/xstest/xstest_v2_completions_gpt4.csv --maths_pool data/processed/exp27/gen_pool.jsonl --maths_test data/processed/exp27/test.jsonl --legal data/processed/exp06/needles_legal.jsonl --medical data/processed/exp06/needles_medical.jsonl --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from scipy.stats import spearmanr

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.data.pools import normalise
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
NEGATIVES = ("code_other", "multilingual_other", "tool_other", "safety_other", "maths_other")
CONTROLS = ("legal", "medical")
CLASSES = ("deployment_only", "shared", "mix_only")


@dataclass
class EvalSetConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp41_dir: str = ""
    label_files: list[str] = field(default_factory=list)
    code_other: str = ""
    multilingual_other: str = ""
    tool_other: str = ""
    xstest_csv: str = ""
    maths_pool: str = ""
    maths_test: str = ""
    legal: str = ""
    medical: str = ""
    seed: int = 0
    n_heldout: int = 2000
    k: int = 50
    taus: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    headline_tau: float = 0.10
    ks: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 293])
    near_dup: float = 0.95
    n_examples: int = 5
    min_set: int = 250


def set_stats(ratio: np.ndarray, nearest: np.ndarray, line: float, ceiling: float, near_dup: float) -> dict:
    return {
        "n": len(ratio),
        "flag_rate": float(np.mean(ratio >= line)),
        "median_ratio": float(np.median(ratio)),
        "p90_ratio": float(np.quantile(ratio, 0.9)),
        "ceiling_share": float(np.mean(ratio >= 0.999 * ceiling)),
        "near_dup_share": float(np.mean(nearest >= near_dup)),
    }


def main():
    config = simple_parsing.parse(EvalSetConfig)
    paths = {k: v for k, v in asdict(config).items() if isinstance(v, str)}
    missing = [k for k, v in paths.items() if v == ""]
    assert not missing and config.label_files, f"required arguments not set: {missing or 'label_files'}"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, embedder = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    ratio37 = np.load(Path(config.exp37_dir) / "ratio.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])
    n_t, n_m = len(t_np), len(m_np)
    assert n_t == 992933 and n_m == 1710578, (n_t, n_m)
    line = json.loads((Path(config.exp41_dir) / "decision.json").read_text())["ratio_line"]
    LOGGER.info(f"traffic {n_t}, mix {n_m}, gap line {line}")

    # the held-out references leave their own side before scoring
    ho_t = rng.choice(n_t, size=config.n_heldout, replace=False)
    ho_m = rng.choice(n_m, size=config.n_heldout, replace=False)
    keep_t, keep_m = np.ones(n_t, dtype=bool), np.ones(n_m, dtype=bool)
    keep_t[ho_t], keep_m[ho_m] = False, False
    side_t = {
        "emb": torch.from_numpy(t_np[keep_t]).cuda(),
        "hash": torch.from_numpy(t_hash[keep_t]).cuda(),
    }
    side_m = {
        "emb": torch.from_numpy(m_np[keep_m]).cuda(),
        "hash": torch.from_numpy(m_hash[keep_m]).cuda(),
    }
    ceiling = ((config.k + 1) / int(keep_t.sum())) / (1 / int(keep_m.sum()))

    def embed(texts: list[str]) -> np.ndarray:
        return embedder.encode(
            [t[: c37["embed_chars"]] for t in texts],
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float16)

    xs = pd.read_csv(config.xstest_csv)
    maths = [r["text"] for r in load_jsonl(config.maths_pool) if r["family"] == "maths"]
    maths += [r["text"] for r in load_jsonl(config.maths_test) if r["family"] == "maths"]
    texts = {
        "code_other": [r["text"] for r in load_jsonl(config.code_other)],
        "multilingual_other": [r["text"] for r in load_jsonl(config.multilingual_other)],
        "tool_other": [r["text"] for r in load_jsonl(config.tool_other)],
        "safety_other": xs["prompt"].drop_duplicates().tolist(),
        "maths_other": list(dict.fromkeys(maths)),
        "legal": [r["text"] for r in load_jsonl(config.legal)],
        "medical": [r["text"] for r in load_jsonl(config.medical)],
    }
    for name, t in texts.items():
        assert len(t) >= config.min_set, f"{name} has {len(t)} rows, under {config.min_set}"
    queries = {
        name: (embed(t), np.array([text_hash(normalise(x)) for x in t], dtype=np.int64), t) for name, t in texts.items()
    }
    queries["heldout_traffic"] = (t_np[ho_t], t_hash[ho_t], [rows[i]["text"] for i in ho_t])
    queries["heldout_mix"] = (m_np[ho_m], m_hash[ho_m], None)

    sets, examples = {}, {}
    for name, (q_emb, q_hash, q_text) in queries.items():
        ratio, nearest = density_ratios_batched(
            q_emb,
            q_hash,
            np.full(len(q_emb), -1, dtype=np.int64),
            side_t["emb"],
            side_t["hash"],
            side_m["emb"],
            side_m["hash"],
            k=config.k,
            return_nearest_mix=True,
        )
        sets[name] = set_stats(ratio, nearest, line, ceiling, config.near_dup)
        if q_text is not None:
            pick = rng.choice(len(q_text), size=config.n_examples, replace=False)
            examples[name] = [
                {
                    "id": f"{name}_{int(i)}",
                    "ratio": round(float(ratio[i]), 2),
                    "flagged": bool(ratio[i] >= line),
                    "text": q_text[i][:300],
                }
                for i in pick
            ]
        LOGGER.info(f"{name}: {sets[name]}")
    write_json(sets, run / "sets.json")
    write_json(examples, run / "examples.json")

    valid = sets["heldout_traffic"]["flag_rate"] >= 3 * sets["heldout_mix"]["flag_rate"]
    verdicts = {}
    for name in NEGATIVES:
        fr = sets[name]["flag_rate"]
        if fr <= sets["heldout_mix"]["flag_rate"] + 0.05 and fr <= 0.5 * sets["heldout_traffic"]["flag_rate"]:
            verdicts[name] = "passes"
        elif fr >= sets["heldout_traffic"]["flag_rate"]:
            verdicts[name] = "fails"
        else:
            verdicts[name] = "ambiguous"
    del side_t, side_m
    torch.cuda.empty_cache()

    # item 2: assign every mix prompt to its nearest traffic cluster centroid, then label clusters by mix share
    n_clusters = int(labels.max()) + 1
    cent = np.zeros((n_clusters, t_np.shape[1]), dtype=np.float32)
    np.add.at(cent, labels, t_np.astype(np.float32))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    cent_gpu = torch.from_numpy(cent).cuda()
    assign = np.empty(n_m, dtype=np.int64)
    for s in range(0, n_m, 65536):
        q = torch.from_numpy(m_np[s : s + 65536]).cuda().float()
        assign[s : s + 65536] = (q @ cent_gpu.T).argmax(1).cpu().numpy()
    count_t = np.bincount(labels, minlength=n_clusters)
    count_m = np.bincount(assign, minlength=n_clusters)
    assert count_m.sum() == n_m
    rate_t, rate_m = count_t / n_t, count_m / n_m
    share = rate_m / (rate_m + rate_t + 1e-12)

    def classify(s: float, tau: float) -> str:
        return "deployment_only" if s <= tau else "mix_only" if s >= 1 - tau else "shared"

    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    ranked = sorted((c for c in clusters if clusters[c]["rank"]), key=lambda c: clusters[c]["rank"])
    cluster_ratio = np.array([clusters[c]["ratio"] for c in ranked])
    gt_rows = [
        {
            "id": f"c{c}",
            "rank": clusters[c]["rank"],
            "ratio": clusters[c]["ratio"],
            "n_traffic": int(count_t[c]),
            "n_mix": int(count_m[c]),
            "mix_share": round(float(share[c]), 4),
            **{f"class_{tau}": classify(share[c], tau) for tau in config.taus},
        }
        for c in sorted(clusters)
    ]
    write_jsonl(gt_rows, run / "cluster_ground_truth.jsonl")
    precision = {
        str(tau): {
            str(K): float(np.mean([classify(share[c], tau) == "deployment_only" for c in ranked[:K]]))
            for K in config.ks
        }
        for tau in config.taus
    }
    counts = {
        str(tau): {cl: int(sum(classify(share[c], tau) == cl for c in ranked)) for cl in CLASSES} for tau in config.taus
    }
    rho = float(spearmanr(cluster_ratio, 1 - share[ranked]).statistic)

    # the human check: the user's exp41 and exp43 topic labels against the membership classes
    human = {}
    for path in config.label_files:
        for r in json.loads(Path(path).read_text())["rows"]:
            human[int(r["id"][1:])] = r["human"]
    assert all(c in clusters and clusters[c]["rank"] for c in human), "a labelled topic is not a ranked cluster"
    tau = config.headline_tau
    agree, scored = 0, 0
    confusion = {h: {cl: 0 for cl in CLASSES} for h in ("real_gap", "partly", "covered")}
    for c, h in human.items():
        cl = classify(share[c], tau)
        confusion[h][cl] += 1
        if h == "partly":
            continue
        scored += 1
        agree += (h == "real_gap" and cl == "deployment_only") or (h == "covered" and cl != "deployment_only")
    agreement = agree / max(1, scored)
    gt_valid = agreement >= 0.70
    p100 = precision[str(tau)]["100"]
    usable = "usable" if p100 >= 0.5 else "not_usable" if p100 < 0.3 else "ambiguous"
    dep = {str(t): counts[str(t)]["deployment_only"] for t in config.taus}
    base = dep[str(tau)]
    stable = all(base / 2 <= v <= base * 2 for v in dep.values())

    decision = {
        "git_hash": get_git_hash(),
        "gap_line": line,
        "ratio_ceiling": ceiling,
        "item1_valid": bool(valid),
        "item1_verdicts": verdicts if valid else {n: "no_verdict_measure_invalid" for n in NEGATIVES},
        "item1_sets": sets,
        "item2_ground_truth_valid": bool(gt_valid),
        "item2_human_agreement": {
            "agreement": agreement,
            "n_scored": scored,
            "n_labelled": len(human),
            "confusion_at_headline_tau": confusion,
        },
        "item2_precision_at_k": precision,
        "item2_class_counts": counts,
        "item2_usable_by_stefans_line": usable if gt_valid else "unvalidated",
        "item2_threshold_stable": bool(stable),
        "item2_spearman_ratio_vs_traffic_share": rho,
        "sanity": {
            "n_traffic": n_t,
            "n_mix": n_m,
            "heldout_removed": {"traffic": int((~keep_t).sum()), "mix": int((~keep_m).sum())},
            "set_sizes": {n: len(t) for n, t in texts.items()},
            "mix_assigned": int(count_m.sum()),
            "labelled_topics_ranked": len(human),
        },
    }
    write_json(decision, run / "decision.json")
    print(
        f"item 1 valid {valid}: heldout_traffic flag {sets['heldout_traffic']['flag_rate']:.3f}, heldout_mix flag {sets['heldout_mix']['flag_rate']:.3f}"
    )
    for name in NEGATIVES + CONTROLS:
        print(
            f"  {name}: flag {sets[name]['flag_rate']:.3f} median {sets[name]['median_ratio']:.2f} near-dup {sets[name]['near_dup_share']:.3f} {verdicts.get(name, '')}"
        )
    print(
        f"item 2 agreement {agreement:.3f} on {scored} (valid {gt_valid}); precision@K at tau {tau}: {precision[str(tau)]}; {usable}; stable {stable}; rho {rho:.3f}"
    )
    print(f"class counts {counts}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
