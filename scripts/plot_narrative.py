# draws the narrative figures (detector against baselines, grouping, behaviour supplement) with the viz library.
# uv run --no-project --python /root/viz_env/bin/python -m scripts.plot_narrative --output_dir results/figures/narrative --seed 0

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import viz
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

LOGGER = logging.getLogger(__name__)
DetectorFactory.seed = 0
RAW = Path("results/raw")
LANGS = {
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "zh-cn": "Chinese",
    "ja": "Japanese",
}
OLMO, TFIDF, MINILM = "Olmo activations", "tf-idf words", "MiniLM embeddings"
EN, NATIVE = "In English", "In the original language"


@dataclass
class PlotConfig:
    output_dir: str = "results/figures/narrative"
    seed: int = 0
    n_boot: int = 10000


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def boot_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    means = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def is_english(text: str) -> bool | None:
    text = re.sub(r"https?://\S+|```.*?```", "", text[:2000], flags=re.S)
    if sum(c.isalpha() for c in text) < 20:
        return None
    # a text with no usable features counts as unknown, like a low-confidence one
    try:
        best = detect_langs(text)[0]
    except LangDetectException:
        return None
    return best.lang == "en" if best.prob >= 0.9 else None


def source_removal(out: Path, plotted: dict) -> None:
    d = json.loads((RAW / "261028_source_removal_v1/decision.json").read_text())
    chosen = d["chosen"]
    rows = load_jsonl(RAW / "261028_source_removal_v1/fold_scores.jsonl")
    names = {"olmo": OLMO, "tfidf": TFIDF, "minilm": MINILM, "length": "Prompt length", "random": "Random"}
    runs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = chosen[r["family"]]
        if (
            r["setting"] == c["setting"]
            and r["layer"] == c["layer"]
            and r["pooling"] == c["pooling"]
            and r["fold"] in d["test_auroc_per_fold"]["olmo"]
        ):
            runs[names[r["family"]]][r["fold"]].append(r["test_auroc"])
    per_fold = {m: [float(np.mean(v)) for v in folds.values()] for m, folds in runs.items()}
    means = {m: round(float(np.mean(v)), 3) for m, v in per_fold.items()}
    fig = viz.bars(
        means,
        runs=per_fold,
        unit="",
        domain=(0.3, 1.0),
        ref=0.5,
        ref_label="Chance",
        roles={"baseline": TFIDF, "intervention": OLMO},
        ylabel="AUROC for the held-out source (dots: each of 12 sources)",
        title=f"Olmo activations flagged held-out Dolci sources at AUROC {means[OLMO]:.3f}, words {means[TFIDF]:.3f}",
    )
    save(fig, out / "01_source_removal", plotted, {"mean_auroc": means, "per_fold": per_fold})

    folds = d["test_auroc_per_fold"]
    pairs = {
        f: (folds["tfidf"][f], folds["olmo"][f])
        for f in sorted(folds["olmo"], key=lambda f: folds["olmo"][f] - folds["tfidf"][f])
    }
    wins = sum(o > t for t, o in pairs.values())
    fig = viz.dumbbell(
        pairs,
        series=(TFIDF, OLMO),
        unit="",
        ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        roles={"baseline": TFIDF, "intervention": OLMO},
        title=f"Olmo activations beat words on {wins} of {len(pairs)} held-out sources (AUROC)",
    )
    save(fig, out / "02_source_removal_per_source", plotted, {"auroc_tfidf_olmo": pairs})


def tests_summary(out: Path, plotted: dict) -> None:
    src = json.loads((RAW / "261028_source_removal_v1/decision.json").read_text())["mean_test_auroc"]
    planted = json.loads((RAW / "261029_combined_score_combined_v1/decision.json").read_text())["results"]["B"][
        "prompt_auroc"
    ]
    matched = json.loads((RAW / "261029_negatives_test_negatives_v1/decision.json").read_text())["matched"]
    topic = json.loads((RAW / "261028_topic_removal_v1/decision.json").read_text())["summary"]
    rows = {
        "Held-out Dolci source": (src["tfidf"], src["olmo"]),
        "Kind removed from Dolci (4 kinds)": (
            round(float(np.mean([v["auroc_removed_vs_covered"]["tfidf"] for v in matched.values()])), 4),
            round(float(np.mean([v["auroc_removed_vs_covered"]["delta_knn"] for v in matched.values()])), 4),
        ),
        "Planted legal and medical prompts": (planted["tfidf"], planted["olmo_diff"]),
        "Traffic topics held out": (topic["tfidf"]["main_topic_test"], topic["olmo"]["main_topic_test"]),
    }
    rows = {k: (round(a, 4), round(b, 4)) for k, (a, b) in rows.items()}
    fig = viz.dumbbell(
        rows,
        series=(TFIDF, OLMO),
        unit="",
        ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        roles={"baseline": TFIDF, "intervention": OLMO},
        title="Olmo beat words on removed data, tied on planted prompts and topics (AUROC)",
    )
    save(fig, out / "03_detector_vs_words_by_test", plotted, {"auroc_tfidf_olmo": rows})


def grouping(out: Path, plotted: dict) -> None:
    s = json.loads((RAW / "261030_grouping_test_v2/decision.json").read_text())["summary"]
    names = {
        "topics": "Topics",
        "topic_x_kind": "Topic by kind",
        "topic_x_olmo2": "Topic split by Olmo",
        "olmo_clusters_min10": "Olmo clusters",
        "prompts": "Single prompts",
    }
    data = {
        n: [round(s[f"olmo|delta_knn|{g}"]["chat_auroc"], 3), round(s[f"olmo|delta_knn|{g}"]["clusters_auroc_mean"], 3)]
        for g, n in names.items()
    }
    rand = round(float(np.mean([s[f"olmo|delta_knn|{g}"]["chat_auroc_random"] for g in names])), 3)
    fig = viz.hbar(
        data,
        series=["Chat data removed", "Five Dolci clusters removed"],
        unit="",
        domain=(0.4, 1.0),
        ref=rand,
        ref_label=f"Random removal {rand}",
        sort=False,
        roles={"intervention": "Chat data removed", "second": "Five Dolci clusters removed"},
        title=f"Olmo clusters recovered planted gaps at AUROC {data['Olmo clusters'][0]:.3f}, topics at {data['Topics'][0]:.3f}",
    )
    save(fig, out / "04_grouping_recovers_planted_gaps", plotted, {"auroc": data, "random_removal_mean": rand})

    labels = json.loads(Path("data/processed/human_labels_cluster_gap_list/labels.json").read_text())
    counts = defaultdict(lambda: [0, 0])
    for lab in labels:
        counts[lab["source"]][1] += 1
        counts[lab["source"]][0] += lab["verdict"] == "real_gap"
    share = {
        "Olmo clusters" if k == "olmo_clusters" else "Topics": round(100 * a / b, 1) for k, (a, b) in counts.items()
    }
    notes = {"Olmo clusters" if k == "olmo_clusters" else "Topics": f"{a} of {b}" for k, (a, b) in counts.items()}
    fig = viz.hbar(
        share,
        notes=notes,
        roles={"baseline": "Topics", "intervention": "Olmo clusters"},
        domain=(0, 100),
        title=f"A person labelled {notes['Olmo clusters']} top Olmo clusters real gaps, against {notes['Topics']} topics",
    )
    save(fig, out / "05_gap_list_hand_labels", plotted, {"real_gap_share": share, "counts": notes})


def weird_tail(out: Path, plotted: dict, rng: np.random.Generator, n_boot: int) -> None:
    d = json.loads((RAW / "261030_weird_tail_v1/decision.json").read_text())
    pool = {r["id"]: r for r in load_jsonl(Path("data/processed/weird_tail/pool.jsonl"))}
    pp = load_jsonl(RAW / "261030_weird_tail_v1/per_prompt.jsonl")
    english = {r["id"]: is_english(pool[r["id"]]["text"]) for r in pp if r["arm"] in ("far", "covered_traffic")}
    far = np.array(
        [r["odd_share"] for r in pp if r["arm"] == "far" and english[r["id"]] and r["odd_share"] is not None]
    )
    cov = np.array(
        [
            r["odd_share"]
            for r in pp
            if r["arm"] == "covered_traffic" and english[r["id"]] and r["odd_share"] is not None
        ]
    )
    boots = rng.choice(far, (n_boot, len(far))).mean(1) - rng.choice(cov, (n_boot, len(cov))).mean(1)
    far_label = "Far from Dolci vs covered (English)"
    rows = {far_label: (100 * (far.mean() - cov.mean()), tuple(100 * np.percentile(boots, [2.5, 97.5])))}
    names = {
        "base64": "Prompt in base64",
        "noise": "15% of letters replaced",
        "shuffle": "Words shuffled",
        "junk": "Junk tokens appended",
    }
    for k, n in names.items():
        c = d["positive_control"][k]["odd_share"]
        rows[n] = (100 * c["diff"], (100 * c["ci95"][0], 100 * c["ci95"][1]))
    data = {k: round(v, 1) for k, (v, _) in rows.items()}
    ci = {k: (round(a, 1), round(b, 1)) for k, (_, (a, b)) in rows.items()}
    fig = viz.dotplot(
        data,
        ci=ci,
        ref=0,
        unit="",
        sort=True,
        highlight=far_label,
        subtitle="Error bars are 95% intervals",
        xlabel="Change in replies judged odd (percentage points)",
        title=f"Base64 prompts raised odd replies by {data['Prompt in base64']} points, far English prompts by {data[far_label]}",
    )
    save(
        fig,
        out / "06_far_prompts_not_odd",
        plotted,
        {"pp_change": data, "ci95": ci, "n_far_english": len(far), "n_covered_english": len(cov)},
    )


def language(out: Path, plotted: dict, rng: np.random.Generator, n_boot: int) -> None:
    pp = {r["id"]: r for r in load_jsonl(RAW / "261030_language_weird_v1/per_prompt.jsonl")}
    pairs = defaultdict(list)
    for r in pp.values():
        if r["version"] == "native" and f"{r['id']}_en" in pp:
            pairs[r["lang"]].append((pp[f"{r['id']}_en"]["odd_share"], r["odd_share"]))
    rows = {
        LANGS[lang]: (
            round(100 * float(np.nanmean([a for a, _ in v if a is not None])), 1),
            round(100 * float(np.nanmean([b for _, b in v if b is not None])), 1),
        )
        for lang, v in pairs.items()
    }
    rows = dict(sorted(rows.items(), key=lambda kv: kv[1][1] - kv[1][0]))
    allp = [p for v in pairs.values() for p in v if None not in p]
    en, nat = 100 * np.mean([a for a, _ in allp]), 100 * np.mean([b for _, b in allp])
    fig = viz.dumbbell(
        rows,
        series=(EN, NATIVE),
        roles={"baseline": EN, "intervention": NATIVE},
        title=f"The same prompts drew odd replies {round(nat, 1)}% in their language, {round(en, 1)}% in English",
    )
    save(
        fig,
        out / "07_language_odd_by_language",
        plotted,
        {"odd_pct_english_native": rows, "overall": [round(en, 1), round(nat, 1)], "n_pairs": len(allp)},
    )

    base = json.loads((RAW / "261030_base_language_v1/decision.json").read_text())["means"]
    data = {
        "Base model": [
            round(100 * base["base|native_to_en"]["odd_share"], 1),
            round(100 * base["base|native"]["odd_share"], 1),
        ],
        "After post-training": [
            round(100 * base["instruct|native_to_en"]["odd_share"], 1),
            round(100 * base["instruct|native"]["odd_share"], 1),
        ],
    }
    fig = viz.bars(
        data,
        series=[EN, NATIVE],
        sort=False,
        roles={"baseline": EN, "intervention": NATIVE},
        ylabel="Replies judged odd (same prompts in both languages)",
        title=f"Post-training cut odd English replies from {data['Base model'][0]}% to {data['After post-training'][0]}%, not others",
    )
    save(fig, out / "08_base_vs_instruct", plotted, {"odd_pct": data})

    h = json.loads((RAW / "261030_language_weird_v1/score_halves.json").read_text())
    names = {"delta_knn": "Gap score (Delta-kNN)", "to_dolci": "Distance to Dolci", "to_dolma": "Distance to Dolma"}
    data = {n: [h[f"{k}|native"]["diff"], h[f"{k}|en_to_x"]["diff"]] for k, n in names.items()}
    ci = {n: [tuple(h[f"{k}|native"]["ci95"]), tuple(h[f"{k}|en_to_x"]["ci95"])] for k, n in names.items()}
    fig = viz.dotplot(
        data,
        ci=ci,
        series=["User prompts in their language", "English prompts translated"],
        ref=0,
        unit="",
        sort=False,
        subtitle="Error bars are 95% intervals",
        xlabel="Non-English minus English version of the same prompt (below 0: closer to the data)",
        roles={"intervention": "User prompts in their language", "second": "English prompts translated"},
        title=f"The gap score rated non-English versions {abs(data['Gap score (Delta-kNN)'][0]):.3f} closer to Dolci",
    )
    save(fig, out / "09_score_blind_to_language", plotted, {"diff": data, "ci95": ci})

    fig = viz.quotes(
        prompt="“Explain to me about tennis” (asked in English, and the same request in Russian)",
        replies=[
            viz.Reply(
                EN,
                "“Tennis is a racket sport played between two players on a rectangular court. The objective is to hit a small, hard ball over a net into the opponent's court…”",
                "judged not odd",
                "control",
            ),
            viz.Reply(
                "In Russian",
                "“Tennis is a sports game played by two players on a board with drums. One player (grey) tries to get the ball into a circle on the opposite side of the board, using a rocket (scythe) pulled out of the drum…”",
                "our translation; all three judges: garbled",
                "intervention",
            ),
        ],
        title="In Russian, Olmo described tennis as a board game with drums",
    )
    save(fig, out / "10_example_tennis", plotted, {"prompt_ids": ["t703303", "t703303_ru"]})


def save(fig, stem: Path, plotted: dict, data: dict) -> None:
    problems = fig.check()
    if problems:
        LOGGER.warning(f"{stem.name}: {problems}")
    fig.save(str(stem) + ".png")
    fig.save(str(stem) + ".pdf")
    plotted[stem.name] = {
        "title": fig.alt_text().split("\n")[0],
        "data": data,
        "check": [str(p) for p in problems] if problems else [],
    }
    print(f"Figure saved to {stem}.png")


def main() -> None:
    config = simple_parsing.parse(PlotConfig)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    plotted: dict = {}
    source_removal(out, plotted)
    tests_summary(out, plotted)
    grouping(out, plotted)
    weird_tail(out, plotted, rng, config.n_boot)
    language(out, plotted, rng, config.n_boot)
    (out / "plotted_data.json").write_text(json.dumps(plotted, indent=1, default=str))
    print(f"Plotted data saved to {out / 'plotted_data.json'}")


if __name__ == "__main__":
    main()
