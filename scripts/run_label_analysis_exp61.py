# exp61: precision at depth from the 20 rank-stratified hand labels, and consistency of the 58 relabels against the first round.
# uv run -m scripts.run_label_analysis_exp61 --run_dir results/raw/261026_exp61_labels_v1 --label_files results/raw/261022_exp41_topic_labels_v1/decision.json results/raw/261022_exp43_topic_labels_v1/decision.json --reviewed_path results/raw/261022_exp41_gap_loop_v1/reviewed_topics.jsonl

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.plotting import wilson

LOGGER = logging.getLogger(__name__)
BINS = {"1-50": 50, "51-100": 50, "101-200": 100, "201-293": 93}
STRICT = {"real_gap"}
LENIENT = {"real_gap", "partly"}
K_LIST = (50, 100, 200, 293)


@dataclass
class AnalysisConfig:
    run_dir: str = ""
    label_files: list[str] = field(default_factory=list)
    reviewed_path: str = ""
    min_precision: float = 0.5
    min_lower: float = 0.3


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    cats = set(a) | set(b)
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    config = simple_parsing.parse(AnalysisConfig)
    assert config.run_dir and config.label_files
    run = Path(config.run_dir)
    key = json.loads(Path(run / "key.json").read_text())
    labels = {p.stem: json.loads(Path(p).read_text())["verdict"] for p in sorted((run / "labels").glob("*.json"))}
    assert len(labels) == 78, len(labels)

    names = {}
    for path in config.label_files:
        names.update({r["id"]: r["name"] for r in json.loads(Path(path).read_text())["rows"]})
    if config.reviewed_path:
        names.update({r["id"]: r["name"] for r in load_jsonl(config.reviewed_path)})

    new_rows = [
        {"id": c, "bin": v["bin"], "rank": v["rank"], "label": labels[f"e61n_{c}"], "name": names.get(c)}
        for c, v in sorted(key["new"].items(), key=lambda kv: kv[1]["rank"])
    ]
    per_bin = {}
    for b, size in BINS.items():
        rows = [r for r in new_rows if r["bin"] == b]
        n = len(rows)
        k_s = sum(r["label"] in STRICT for r in rows)
        k_l = sum(r["label"] in LENIENT for r in rows)
        per_bin[b] = {"bin_size": size, "n_labelled": n, "strict": wilson(k_s, n), "lenient": wilson(k_l, n)}

    cumulative = {}
    for k_depth in K_LIST:
        bins = [b for b, size in BINS.items() if int(b.split("-")[1]) <= k_depth]
        weighted_s = sum(BINS[b] * per_bin[b]["strict"][0] for b in bins) / k_depth
        weighted_l = sum(BINS[b] * per_bin[b]["lenient"][0] for b in bins) / k_depth
        rows = [r for r in new_rows if r["bin"] in bins]
        pooled_s = wilson(sum(r["label"] in STRICT for r in rows), len(rows))
        pooled_l = wilson(sum(r["label"] in LENIENT for r in rows), len(rows))
        cumulative[str(k_depth)] = {
            "strict_weighted": weighted_s,
            "strict_pooled": pooled_s,
            "lenient_weighted": weighted_l,
            "lenient_pooled": pooled_l,
            "usable": weighted_s >= config.min_precision and pooled_s[1] > config.min_lower,
        }
    usable_depths = [int(k) for k, v in cumulative.items() if v["usable"]]

    relabel_rows = [
        {"id": c, "rank": v["rank"], "first": v["first_label"], "second": labels[f"e61r_{c}"], "name": names.get(c)}
        for c, v in sorted(key["relabel"].items(), key=lambda kv: kv[1]["rank"])
    ]
    first = [r["first"] for r in relabel_rows]
    second = [r["second"] for r in relabel_rows]
    first_m = ["gap" if x in LENIENT else x for x in first]
    second_m = ["gap" if x in LENIENT else x for x in second]
    n = len(relabel_rows)
    consistency = {
        "n": n,
        "exact_agreement": sum(a == b for a, b in zip(first, second, strict=True)) / n,
        "exact_kappa": kappa(first, second),
        "merged_agreement": sum(a == b for a, b in zip(first_m, second_m, strict=True)) / n,
        "merged_kappa": kappa(first_m, second_m),
        "first_counts": {c: first.count(c) for c in sorted(set(first))},
        "second_counts": {c: second.count(c) for c in sorted(set(second))},
        "confusion": {
            f"{a}->{b}": sum(x == a and y == b for x, y in zip(first, second, strict=True))
            for a in sorted(set(first))
            for b in sorted(set(second))
        },
        "strict_precision_on_58": {
            "first": wilson(sum(x in STRICT for x in first), n),
            "second": wilson(sum(x in STRICT for x in second), n),
        },
    }
    is_consistent = consistency["exact_agreement"] >= 0.70

    decision = {
        "n_labels": len(labels),
        "label_counts": {c: list(labels.values()).count(c) for c in sorted(set(labels.values()))},
        "per_bin": per_bin,
        "cumulative": cumulative,
        "bar": {
            "min_precision": config.min_precision,
            "min_lower": config.min_lower,
            "usable_depths": usable_depths,
            "review_budget": max(usable_depths, default=0),
        },
        "consistency": consistency,
        "consistency_bar_met": is_consistent,
        "new_rows": new_rows,
        "relabel_rows": relabel_rows,
        "git_hash": get_git_hash(),
    }
    out = write_json(decision, run / "decision.json")

    print("precision per bin (strict p, lo, hi | lenient p, lo, hi)")
    for b, v in per_bin.items():
        print(b, [round(x, 2) for x in v["strict"]], [round(x, 2) for x in v["lenient"]])
    print("cumulative precision at K (strict weighted, pooled lo | lenient weighted, usable)")
    for k_depth, v in cumulative.items():
        print(
            k_depth,
            round(v["strict_weighted"], 2),
            round(v["strict_pooled"][1], 2),
            round(v["lenient_weighted"], 2),
            v["usable"],
        )
    print(f"review budget {decision['bar']['review_budget']}")
    print(
        f"relabel exact agreement {consistency['exact_agreement']:.2f} kappa {consistency['exact_kappa']:.2f}; merged {consistency['merged_agreement']:.2f} kappa {consistency['merged_kappa']:.2f}"
    )
    print(
        f"strict real-gap share on the 58: first {consistency['strict_precision_on_58']['first'][0]:.2f}, second {consistency['strict_precision_on_58']['second'][0]:.2f}"
    )
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
