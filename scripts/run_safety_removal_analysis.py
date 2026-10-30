# safety_removal/removal_code_multilingual step 5: ranks topics per mix and score, counts target topics (harmful, or of the removed kind) in each top-N list against the full mix, and applies the pre-registered bars.
# uv run -m scripts.run_safety_removal_analysis --run_dir results/raw/261027_safety_removal_v1 --cache_dir cache/safety_removal --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --k_sweep_behaviour_dir results/raw/261027_k_sweep_behaviour_k_sweep_v1 --seed 0

import asyncio
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, name_clusters, naming_prompt, parse_naming
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    run_dir: str = ""
    cache_dir: str = ""
    pipeline_million_dir: str = ""
    k_sweep_behaviour_dir: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    n_random_subsets: int = 1000
    subset_size: int = 500
    ratio_multiple: float = 3.0
    n_named: int = 25
    shared_needed: int = 13
    finds_gain: int = 10
    finds_margin: int = 8
    fails_margin: int = 2
    better_by: int = 5
    min_room: int = 15
    repro_min_spearman: float = 0.99
    n_examples: int = 5
    n_traffic: int = 992_933
    n_topics: int = 1998
    skip_namer: bool = False
    # removal_code_multilingual: the removed kind, where its labels live, the topic label and grade fields, and the run holding the full arm
    removal: str = "safety"
    labels_dir: str = ""
    label_key: str = "is_harmful"
    grade_field: str = "request_harmful"
    gate_kind: str = ""
    full_dir: str = ""
    # carry on past a failed grader gate; validity still records the gate result
    ignore_gate: bool = False
    # removal_code_multilingual amendment: a topic off k_sweep_behaviour by more than 1% still passes if it lies between the full arm's two tie orders
    repro_tie_aware: bool = False
    repro_min_share_exact: float = 0.9


def topic_medians(values: np.ndarray, members: dict[int, np.ndarray], fill: float) -> dict[int, float]:
    # smoke runs score only some prompts; a topic with none scored gets the fill value
    return {c: float(np.nanmedian(values[m])) if np.isfinite(values[m]).any() else fill for c, m in members.items()}


def auroc_or_none(score: dict[int, float], pos: set[int], neg: set[int]) -> float | None:
    pos_s, neg_s = [score[c] for c in pos if np.isfinite(score[c])], [score[c] for c in neg if np.isfinite(score[c])]
    if not pos_s or not neg_s:
        return None
    return float(roc_auc_score(np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))], np.r_[pos_s, neg_s]))


def main():
    config = simple_parsing.parse(AnalysisConfig)
    assert (
        config.run_dir and config.cache_dir and config.pipeline_million_dir and config.k_sweep_behaviour_dir
    ), "paths required"
    run = Path(config.run_dir)
    assert not (run / "decision.json").exists(), f"{run} already analysed; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")

    labels_dir = Path(config.labels_dir) if config.labels_dir else run
    full_dir = Path(config.full_dir) if config.full_dir else run
    minus = f"minus_{config.removal}"
    count_arms = ("full", minus, "random_0", "random_1", "random_2")
    word_arms = ("full", minus, "random_0")
    gate = json.loads((labels_dir / "grade_gate.json").read_text())
    gate_passed = gate["passed_by_kind"][config.gate_kind] if config.gate_kind else gate["passed"]
    assert gate_passed or config.ignore_gate, "the grader gate failed, so the labels are not used (pre-registered stop)"
    labels = np.load(Path(config.cache_dir) / "arrays.npz")["labels"]
    n_t = len(labels)
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    tl = {int(r["id"][1:]): r for r in load_jsonl(labels_dir / "topic_labels.jsonl")}
    ranked = [
        c for c in sorted((c for c in clusters if clusters[c]["rank"]), key=lambda c: clusters[c]["rank"]) if c in tl
    ]
    members = {c: np.where(labels == c)[0] for c in ranked}
    # "harmful" below means the target label: harmful for safety_removal, of the removed kind for removal_code_multilingual
    harmful = {c for c in ranked if tl[c][config.label_key] is True}
    labelled = {c for c in ranked if tl[c][config.label_key] is not None}
    jailbreak = {c for c in ranked if tl[c].get("is_jailbreak") is True}
    LOGGER.info(f"{len(ranked)} topics, {len(labelled)} labelled, {len(harmful)} target, {len(jailbreak)} jailbreak")

    counts_full, counts = np.load(full_dir / "counts.npz"), np.load(run / "counts.npz")
    q_rows = counts["q_rows"]
    assert (counts_full["q_rows"] == q_rows).all(), "the full arm and the removal arms score different prompts"

    def arm_counts(key: str, a: str) -> np.ndarray:
        return (counts_full if a == "full" else counts)[f"{key}_{a}"]

    def full_length(x: np.ndarray) -> np.ndarray:
        out = np.full(n_t, np.nan)
        out[q_rows] = x
        return out

    # the bars use the fair tie split; the unchanged count is the reproduction check and a sensitivity check
    ratio = {a: full_length(arm_counts("ratio_fair", a)) for a in count_arms}
    ratio_std = {a: full_length(arm_counts("ratio_std", a)) for a in count_arms}
    near = {a: full_length(arm_counts("near", a)) for a in count_arms}
    tie_decided = {a: arm_counts("tie_decided", a) for a in count_arms}
    c_score_tf = {
        a: topic_medians(full_length(arm_counts("ratio_traffic_first", a)), members, -np.inf) for a in count_arms
    }
    c_score_mf = {a: topic_medians(full_length(arm_counts("ratio_mix_first", a)), members, -np.inf) for a in count_arms}
    saturated = {a: arm_counts("saturated", a) for a in count_arms}
    safety_share = full_length(counts[f"{config.removal}_share"])
    words_full, words = np.load(full_dir / "words.npz"), np.load(run / "words.npz")

    def arm_logit(a: str) -> np.ndarray:
        x = (words_full if a == "full" else words)[f"logit_{a}"]
        return np.r_[x, np.full(n_t - len(x), np.nan)]

    logit = {a: arm_logit(a) for a in word_arms}

    # topic scores and rankings: count by median ratio, ties by median nearest-mix similarity (lower first); words by median logit
    c_score = {a: topic_medians(ratio[a], members, -np.inf) for a in count_arms}
    c_score_std = {a: topic_medians(ratio_std[a], members, -np.inf) for a in count_arms}
    c_near = {a: topic_medians(near[a], members, np.inf) for a in count_arms}
    w_score = {a: topic_medians(logit[a], members, -np.inf) for a in word_arms}
    rank = {("count", a): sorted(ranked, key=lambda c, a=a: (-c_score[a][c], c_near[a][c])) for a in count_arms}
    rank |= {
        ("count_std", a): sorted(ranked, key=lambda c, a=a: (-c_score_std[a][c], c_near[a][c])) for a in count_arms
    }
    rank |= {("words", a): sorted(ranked, key=lambda c, a=a: -w_score[a][c]) for a in word_arms}

    # gap line per count arm, the same 1,000 random 500-prompt subsets in every arm
    rng_line = np.random.default_rng(config.seed)
    subsets = np.stack(
        [rng_line.choice(n_t, size=config.subset_size, replace=False) for _ in range(config.n_random_subsets)]
    )
    line = {a: config.ratio_multiple * float(np.median(np.nanmedian(ratio[a][subsets], axis=1))) for a in count_arms}
    over_line = {a: sum(c_score[a][c] >= line[a] for c in ranked) for a in count_arms}
    n_list = over_line["full"]
    LOGGER.info(f"lines {line}; over line {over_line}; list length N = {n_list}")

    top = {key: set(r[:n_list]) for key, r in rank.items()}
    families = {"count": count_arms, "words": word_arms, "count_std": count_arms}
    harm_in_top = {f: {a: len(top[(f, a)] & harmful) for a in arms} for f, arms in families.items()}
    gain = {f: {a: harm_in_top[f][a] - harm_in_top[f]["full"] for a in arms} for f, arms in families.items()}
    jail_gain = {
        f: {a: len(top[(f, a)] & jailbreak) - len(top[(f, "full")] & jailbreak) for a in arms}
        for f, arms in families.items()
    }
    room = {f: len(harmful - top[(f, "full")]) for f in families}
    margin = {f: gain[f][minus] - gain[f]["random_0"] for f in families}

    # validity: reproduction of k_sweep_behaviour's cpu fp32 k=50 medians, full scoring, the grader gate, room
    e69 = {int(r["id"][1:]): r["median_k50"] for r in load_jsonl(Path(config.k_sweep_behaviour_dir) / "topics.jsonl")}
    shared = [c for c in e69 if c in c_score_std["full"]]
    repro = float(spearmanr([e69[c] for c in shared], [c_score_std["full"][c] for c in shared]).statistic)
    repro_rel = np.abs(np.array([c_score_std["full"][c] for c in shared]) / np.array([e69[c] for c in shared]) - 1)
    lo = np.array([min(c_score_tf["full"][c], c_score_mf["full"][c]) for c in shared])
    hi = np.array([max(c_score_tf["full"][c], c_score_mf["full"][c]) for c in shared])
    e69_vals = np.array([e69[c] for c in shared])
    is_between = (e69_vals >= lo * 0.999) & (e69_vals <= hi * 1.001)
    repro_tie_ok = bool(((repro_rel <= 0.01) | is_between).all()) and bool(
        (repro_rel <= 0.01).mean() >= config.repro_min_share_exact
    )
    n_nan = (
        {a: int(np.isnan(ratio[a]).sum()) for a in count_arms}
        | {f"std_{a}": int(np.isnan(ratio_std[a]).sum()) for a in count_arms}
        | {f"words_{a}": int(np.isnan(logit[a]).sum()) for a in word_arms}
    )
    is_complete = (
        len(q_rows) == config.n_traffic
        and n_t == config.n_traffic
        and len(ranked) == config.n_topics
        and not any(n_nan.values())
    )
    validity = {
        "repro_spearman_k_sweep_behaviour": repro,
        "repro_topics": len(shared),
        "repro_share_within_1pct": float((repro_rel <= 0.01).mean()),
        "repro_max_rel_diff": float(repro_rel.max()),
        "repro_off_1pct_between_tie_orders": int(((repro_rel > 0.01) & is_between).sum()),
        "repro_off_1pct_outside_tie_orders": int(((repro_rel > 0.01) & ~is_between).sum()),
        "repro_tie_aware_ok": repro_tie_ok,
        "repro_ok": repro_tie_ok if config.repro_tie_aware else repro >= config.repro_min_spearman,
        "grader_gate_passed": gate_passed,
        "all_prompts_scored": is_complete,
        "nan_counts": n_nan,
        "room": room,
        "room_ok": {f: room[f] >= config.min_room for f in families},
    }
    is_valid_run = validity["repro_ok"] and validity["all_prompts_scored"] and (gate_passed or config.ignore_gate)

    def verdict(f: str) -> str:
        if not is_valid_run:
            return "invalid"
        if not validity["room_ok"][f]:
            return "underpowered"
        if gain[f][minus] >= config.finds_gain and margin[f] >= config.finds_margin:
            return "finds"
        return "fails" if margin[f] <= config.fails_margin else "ambiguous"

    verdicts = {f: verdict(f) for f in families}
    tie_full = {a: full_length(tie_decided[a].astype(float)) for a in count_arms}
    tie_stats = {
        a: {
            "prompts_tie_decided": int(tie_decided[a].sum()),
            "prompts_saturated": int(saturated[a].sum()),
            "topics_half_tie_decided": sum(float(np.nanmean(tie_full[a][m])) >= 0.5 for m in members.values()),
            # topics on the gap line's two sides under the two tie orders: decided by the tie rule, not the data
            "topics_line_decided_by_tie_order": sum(
                (c_score_tf[a][c] >= line[a]) != (c_score_mf[a][c] >= line[a]) for c in ranked
            ),
            "top_n_topics_half_tie_decided": sum(
                float(np.nanmean(tie_full[a][members[c]])) >= 0.5 for c in rank[("count", a)][:n_list]
            ),
        }
        for a in count_arms
    }
    lead = margin["count"] - margin["words"]
    comparison = "count_better" if lead >= config.better_by else "words_better" if -lead >= config.better_by else "tie"
    max_control = max(gain["count"][f"random_{s}"] for s in (0, 1, 2))
    count_margin_max = gain["count"][minus] - max_control
    verdict_max_control = (
        "finds"
        if gain["count"][minus] >= config.finds_gain and count_margin_max >= config.finds_margin
        else "fails"
        if count_margin_max <= config.fails_margin
        else "ambiguous"
    )

    # reported, no bar
    entries = {f: [c for c in rank[(f, minus)][:n_list] if c not in top[(f, "full")]] for f in families}
    exits = {f: [c for c in rank[(f, "full")][:n_list] if c not in top[(f, minus)]] for f in families}
    entry_stats = {
        f: {
            "n_entries": len(entries[f]),
            "n_entries_labelled": len(set(entries[f]) & labelled),
            "harmful_entries": len(set(entries[f]) & harmful),
            "harmful_share_of_labelled_entries": len(set(entries[f]) & harmful)
            / max(1, len(set(entries[f]) & labelled)),
            "harmful_exits": len(set(exits[f]) & harmful),
            "recall_of_room": len(set(entries[f]) & harmful) / max(1, room[f]),
        }
        for f in families
    }
    base_rate = len(harmful) / len(labelled)
    harmless = labelled - harmful
    delta_auroc = {
        "count": {
            a: auroc_or_none({c: np.log(c_score[a][c]) - np.log(c_score["full"][c]) for c in ranked}, harmful, harmless)
            for a in (minus, "random_0", "random_1", "random_2")
        },
        "words": {
            a: auroc_or_none({c: w_score[a][c] - w_score["full"][c] for c in ranked}, harmful, harmless)
            for a in (minus, "random_0")
        },
    }
    t_safety = topic_medians(safety_share, members, np.nan)
    t_safety_mean = {c: float(np.nanmean(safety_share[m])) for c, m in members.items()}
    safety_stats = {
        "auroc_harmful_vs_harmless": auroc_or_none(t_safety_mean, harmful, harmless),
        "mean_share_harmful_topics": float(np.nanmean([t_safety_mean[c] for c in harmful])) if harmful else None,
        "mean_share_harmless_topics": float(np.nanmean([t_safety_mean[c] for c in harmless])) if harmless else None,
        "topics_with_mean_share_over_half": sum(t_safety_mean[c] > 0.5 for c in ranked),
    }

    texts = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()
    names: dict[int, dict] = {}
    if not config.skip_namer:
        shutil.copytree(Path(config.pipeline_million_dir) / "naming_cache", run / "naming_cache", dirs_exist_ok=True)
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        to_name = entries["count"]
        prompts = [
            naming_prompt([texts[i] for i in members[c]], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN)
            for c in to_name
        ]
        replies = asyncio.run(name_clusters(client, config.namer_model, prompts, str(run / "naming_cache")))
        names = {c: parse_naming(r) for c, r in zip(to_name, replies, strict=True)}
    n_null_names = sum(n["shared"] is None for n in names.values())

    grades = list(load_jsonl(labels_dir / "grades.jsonl"))
    harmful_entry_ids = {f"c{c}" for c in set(entries["count"]) & harmful}
    pool = [g for g in grades if g["topic"] in harmful_entry_ids and g[config.grade_field] == "yes"]
    rng = np.random.default_rng(config.seed)
    examples = [pool[i] for i in rng.choice(len(pool), size=min(config.n_examples, len(pool)), replace=False)]

    decision = {
        "git_hash": get_git_hash(),
        "valid": is_valid_run,
        "validity": validity,
        "list_length_n": n_list,
        "lines": line,
        "over_own_line": over_line,
        "removal": config.removal,
        "label_key": config.label_key,
        "harmful_topics": len(harmful),
        "labelled_topics": len(labelled),
        "unlabelled_topics": len(ranked) - len(labelled),
        "jailbreak_topics": len(jailbreak),
        "harmful_base_rate": base_rate,
        "harmful_in_top_n": harm_in_top,
        "gain": gain,
        "margin_over_control_seed0": margin,
        "verdict": {f: verdicts[f] for f in ("count", "words")},
        "sensitivity_unchanged_count_verdict": verdicts["count_std"],
        "ties": tie_stats,
        "comparison": {"lead_count_minus_words": lead, "result": comparison},
        "count_max_over_three_controls": {
            "max_control_gain": max_control,
            "margin": count_margin_max,
            "verdict": verdict_max_control,
        },
        "entries": entry_stats,
        "delta_score_auroc_harmful_vs_harmless": delta_auroc,
        "safety_neighbour_share": safety_stats,
        "jailbreak_gain": jail_gain,
        "count_entries_named": [
            {
                "id": f"c{c}",
                "name": names.get(c, {}).get("name"),
                "shared": names.get(c, {}).get("shared"),
                "coherent": (names.get(c, {}).get("shared") or 0) >= config.shared_needed,
                "is_harmful": tl[c][config.label_key],
                "harmful_share": tl[c][config.label_key.replace("is_", "") + "_share"],
                "safety_share": t_safety_mean[c],
                "rank_full": rank[("count", "full")].index(c) + 1,
                "rank_minus": rank[("count", minus)].index(c) + 1,
            }
            for c in entries["count"]
        ],
        "null_names": n_null_names,
        "examples_harmful_entries": [{k: g.get(k) for k in ("id", "topic", "jailbreak", "text")} for g in examples],
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "size": len(members[c]),
                "harmful_share": tl[c][config.label_key.replace("is_", "") + "_share"],
                "is_harmful": tl[c][config.label_key],
                "jailbreak_share": tl[c].get("jailbreak_share"),
                "safety_share_mean": t_safety_mean[c],
                "safety_share_median": t_safety[c],
                **{f"count_{a}": c_score[a][c] for a in count_arms},
                **{f"count_rank_{a}": rank[("count", a)].index(c) + 1 for a in count_arms},
                **{f"count_unchanged_{a}": c_score_std[a][c] for a in count_arms},
                **{f"count_traffic_first_{a}": c_score_tf[a][c] for a in count_arms},
                **{f"count_mix_first_{a}": c_score_mf[a][c] for a in count_arms},
                **{f"words_{a}": w_score[a][c] for a in word_arms},
                **{f"words_rank_{a}": rank[("words", a)].index(c) + 1 for a in word_arms},
                "name": names.get(c, {}).get("name"),
            }
            for c in ranked
        ],
        run / "topics.jsonl",
    )
    print(f"valid {is_valid_run}; N = {n_list}; harmful topics {len(harmful)} of {len(labelled)} labelled; room {room}")
    print(f"harmful in top N {harm_in_top}")
    print(f"gain {gain}; margin {margin}; verdict {verdicts}; comparison {comparison} (lead {lead})")
    print(f"count verdict against the largest of three controls: {verdict_max_control}")
    for e in decision["count_entries_named"][:15]:
        print(e["rank_full"], "->", e["rank_minus"], e["is_harmful"], e["name"])
    for g in examples:
        print(g["topic"], repr(g["text"][:200]))
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
