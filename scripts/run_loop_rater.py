# loop_rater: deep_gap_loop's skip loop on pipeline_million's count ranking with three confirmation steps, the namer's shared count (reproduces deep_gap_loop) and the retrieval rater at coverage 50 and 65; distinct gaps and labelled precision per arm.
# uv run -m scripts.run_loop_rater --run_dir results/raw/261028_loop_rater_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --pipeline_million_pools data/processed/pipeline_million_v2 --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --rater_confirmation_dir results/raw/261028_rater_confirmation_rater_confirm_v2 --label_files results/raw/261022_deep_gap_loop_topic_labels_v1/decision.json results/raw/261022_classifier_gap_loop_topic_labels_v1/decision.json --precision_at_depth_dir results/raw/261026_precision_at_depth_labels_v1 --real_gaps_by_loop_dir results/raw/261027_real_gaps_by_loop_labels_v1 --seed 0

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "48")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "48")

import asyncio
import json
import logging
import resource
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI
from scipy.stats import fisher_exact

from scripts.run_autorater_baseline import PROMPT, REF_DESC, fmt_pairs, parse
from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("N", "R50", "R65")
CUT = {"R50": 50.0, "R65": 65.0}


@dataclass
class LoopRaterConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    pipeline_million_pools: str = ""
    deep_gap_loop_dir: str = ""
    rater_confirmation_dir: str = ""
    namer_cache_dir: str = ""
    label_files: list[str] = field(default_factory=list)
    precision_at_depth_dir: str = ""
    real_gaps_by_loop_dir: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    rater_model: str = "google/gemini-3.8-flash"
    seed: int = 0
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    n_shown: int = 10
    pair_chars: int = 300
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    same_gap: float = 0.85
    theme_cos: float = 0.70
    template_chars: int = 60
    template_share: float = 0.3
    precision_margin: float = 0.15
    min_labelled: int = 20
    yield_share: float = 0.8
    max_concurrent: int = 6
    retry_passes: int = 3
    retry_sleep_s: int = 60
    n_examples: int = 10
    example_chars: int = 400
    mem_wait_gb: float = 55.0
    mem_poll_s: int = 120
    num_tasks: int | None = None


def main():
    config = simple_parsing.parse(LoopRaterConfig)
    assert (
        config.run_dir
        and config.pipeline_million_dir
        and config.pipeline_million_pools
        and config.deep_gap_loop_dir
        and config.rater_confirmation_dir
    )
    assert config.label_files and config.precision_at_depth_dir and config.real_gaps_by_loop_dir
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)
    shutil.copytree(Path(config.namer_cache_dir or config.deep_gap_loop_dir) / "naming_cache", run / "naming_cache")
    shutil.copytree(Path(config.rater_confirmation_dir) / "rater_cache", run / "rater_cache")
    caches = {"namer": run / "naming_cache", "rater": run / "rater_cache"}
    n_cache_start = {k: len(list(v.glob("*.json"))) for k, v in caches.items()}

    # the user's labels, later files overriding earlier ones for the same topic
    human: dict[int, str] = {}
    for path in config.label_files:
        for r in json.loads(Path(path).read_text())["rows"]:
            human[int(r["id"][1:])] = r["human"]
    e61 = json.loads((Path(config.precision_at_depth_dir) / "decision.json").read_text())
    for r in e61["new_rows"]:
        human[int(r["id"][1:])] = r["label"]
    for r in e61["relabel_rows"]:
        human[int(r["id"][1:])] = r["second"]
    for c, lab in json.loads((Path(config.real_gaps_by_loop_dir) / "decision.json").read_text())["labels"].items():
        human[int(c)] = lab
    LOGGER.info(f"{len(human)} labelled topics: {Counter(human.values())}")

    # the container is capped near 109 gb and other sessions' jobs are live, so the pool waits for room
    while (mem_gb := int(Path("/sys/fs/cgroup/memory.current").read_text()) / 1e9) > config.mem_wait_gb:
        LOGGER.warning(f"container memory {mem_gb:.1f} gb above {config.mem_wait_gb} gb, waiting before loading")
        time.sleep(config.mem_poll_s)
    LOGGER.info(f"container memory {mem_gb:.1f} gb, loading the pool")
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
        device="cpu",
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    ratio = np.load(Path(config.pipeline_million_dir) / "ratio.npy")
    assert len(labels) == len(rows) == len(ratio), "pool does not match pipeline_million's labels"
    for p in load_jsonl(Path(config.pipeline_million_pools) / "prompts.jsonl"):
        assert rows[p["row"]]["text"] == p["text"], f"row {p['row']} differs from pipeline_million"
    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    order37 = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    m_t = torch.from_numpy(m_np.astype(np.float32))
    del m_np
    LOGGER.info(f"pool {len(rows)} rows, {len(order37)} ranked topics, mix {len(m_text)} rows")

    emb = t_np.astype(np.float32)
    members = {c: np.where(labels == c)[0] for c in order37}
    cent = {c: centroid(emb[members[c]]) for c in order37}
    sig: dict[int, str | None] = {}
    for c in order37:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None

    # deep_gap_loop's line, rng drawn first exactly as deep_gap_loop did
    subset_medians = [
        float(np.median(ratio[rng.choice(len(rows), config.subset_size, replace=False)]))
        for _ in range(config.n_random_subsets)
    ]
    line = config.ratio_multiple * float(np.median(subset_medians))
    d41 = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())
    assert round(line, 4) == d41["ratio_line"], f"{line=} differs from deep_gap_loop's"
    score = {c: float(clusters[f"c{c}"]["ratio"]) for c in order37}

    def same(a: int, b: int, cos: float) -> bool:
        return float(cent[a] @ cent[b]) >= cos or (sig[a] is not None and sig[a] == sig[b])

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    names: dict[int, dict] = {}
    rater: dict[int, dict] = {}

    def rater_prompt(c: int) -> str:
        shown = np.random.default_rng(config.seed).choice(
            members[c], size=min(config.n_shown, len(members[c])), replace=False
        )
        best = (torch.from_numpy(emb[shown]) @ m_t.T).argmax(1).numpy()
        topic, ref = fmt_pairs([rows[i]["text"] for i in shown], [m_text[i] for i in best], config.pair_chars)
        return PROMPT.format(ref_desc=REF_DESC["B"], topic=topic, ref=ref)

    def call(kind: str, model: str, prompts: list[str]) -> list[str | None]:
        # name_clusters retries three times; failed slots get up to retry_passes more rounds after a pause
        def batch(ps: list[str]) -> list[str | None]:
            # name_clusters raises when a whole batch fails; under a 429 storm that is a failed batch, not a bad key
            try:
                return asyncio.run(name_clusters(client, model, ps, str(caches[kind]), config.max_concurrent))
            except RuntimeError as error:
                LOGGER.error(f"whole {kind} batch of {len(ps)} failed: {error!r}")
                return [None] * len(ps)

        replies = batch(prompts)
        for _ in range(config.retry_passes):
            todo = [i for i, r in enumerate(replies) if r is None]
            if not todo:
                break
            LOGGER.warning(f"{len(todo)} {kind} calls failed, retrying after {config.retry_sleep_s} s")
            time.sleep(config.retry_sleep_s)
            for i, r in zip(todo, batch([prompts[i] for i in todo]), strict=True):
                replies[i] = r
        return replies

    def review(batch: list[int]) -> None:
        todo_n = [c for c in batch if c not in names]
        n_prompts = [
            naming_prompt(
                [rows[i]["text"] for i in members[c]], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN
            )
            for c in todo_n
        ]
        for c, rep in zip(todo_n, call("namer", config.namer_model, n_prompts), strict=True):
            names[c] = parse_naming(rep)
        # a topic whose rater call failed earlier is tried again when a later arm reviews it
        todo_r = [c for c in batch if c not in rater or rater[c]["failed"]]
        for c, rep in zip(todo_r, call("rater", config.rater_model, [rater_prompt(c) for c in todo_r]), strict=True):
            cov, label = parse(rep)
            rater[c] = {"coverage": cov, "rater_label": label, "rater_reason": (rep or "")[:300], "failed": rep is None}

    def confirmed(arm: str, c: int) -> bool:
        if score[c] < line:
            return False
        if arm == "N":
            shared = names[c].get("shared")
            return shared is not None and shared >= config.shared_needed
        cov = rater[c]["coverage"]
        return cov is not None and cov <= CUT[arm]

    def merge(arm: str, reviewed: list[int], cos: float) -> list[list[int]]:
        gaps: list[list[int]] = []
        for c in reviewed:
            if not confirmed(arm, c):
                continue
            hit = next((g for g in gaps if same(c, g[0], cos)), None)
            if hit is None:
                gaps.append([c])
            else:
                hit.append(c)
        return gaps

    rounds_total = config.num_tasks or config.rounds

    def loop(arm: str) -> tuple[list[int], list[dict]]:
        remaining, reviewed, per_round = list(order37), [], []
        for rnd in range(1, rounds_total + 1):
            batch = remaining[: config.per_round]
            review(batch)
            reviewed += batch
            new_conf = [c for c in batch if confirmed(arm, c)]
            rest = [c for c in remaining if c not in batch]
            dropped = [c for c in rest if any(same(c, g, config.same_gap) for g in new_conf)]
            remaining = [c for c in rest if c not in dropped]
            per_round.append(
                {
                    "id": f"round_{rnd}",
                    "reviewed": batch,
                    "confirmed": new_conf,
                    "dropped_as_same_gap": dropped,
                    "distinct_so_far": len(merge(arm, reviewed, config.same_gap)),
                }
            )
            LOGGER.info(
                f"arm {arm} round {rnd}: {len(new_conf)} confirmed, {per_round[-1]['distinct_so_far']} distinct"
            )
        return reviewed, per_round

    reviewed, rounds_log = {}, {}
    for arm in ARMS:
        reviewed[arm], rounds_log[arm] = loop(arm)

    gaps = {a: merge(a, reviewed[a], config.same_gap) for a in ARMS}
    themes = {a: merge(a, reviewed[a], config.theme_cos) for a in ARMS}
    n = {a: len(g) for a, g in gaps.items()}
    conf = {a: [c for c in reviewed[a] if confirmed(a, c)] for a in ARMS}
    reviewed41 = [c for r in d41["loop_rounds"] for c in r["reviewed"]]
    n_check = {
        "distinct": n["N"],
        "deep_gap_loop_distinct": d41["distinct_gaps"]["loop"],
        "equals_deep_gap_loop_distinct": n["N"] == d41["distinct_gaps"]["loop"],
        "equals_deep_gap_loop_reviewed": reviewed["N"] == reviewed41,
    }
    if not (n_check["equals_deep_gap_loop_distinct"] and n_check["equals_deep_gap_loop_reviewed"]):
        LOGGER.warning(f"arm N does not reproduce deep_gap_loop: {n_check}")

    # labelled precision and known-real recall per arm, partly and incoherent left out of precision
    def labelled(arm: str) -> dict:
        lab = Counter(human[c] for c in conf[arm] if c in human)
        strict_n = lab["real_gap"] + lab["covered"]
        precision = lab["real_gap"] / strict_n if strict_n else None
        return {
            "confirmed_labelled": dict(lab),
            "strict_n": strict_n,
            "strict_precision": precision,
            "real_confirmed": lab["real_gap"],
            "estimated_real_distinct": n[arm] * precision if precision is not None else None,
        }

    lab = {a: labelled(a) for a in ARMS}

    def against_n(arm: str) -> dict:
        r, b = lab[arm], lab["N"]
        table = [
            [r["confirmed_labelled"].get("real_gap", 0), r["confirmed_labelled"].get("covered", 0)],
            [b["confirmed_labelled"].get("real_gap", 0), b["confirmed_labelled"].get("covered", 0)],
        ]
        p = float(fisher_exact(table, alternative="greater").pvalue) if min(map(sum, table)) else None
        enough = r["strict_n"] >= config.min_labelled
        prec_ok = (
            enough
            and r["strict_precision"] is not None
            and b["strict_precision"] is not None
            and r["strict_precision"] >= b["strict_precision"] + config.precision_margin
        )
        yield_ok = (
            r["estimated_real_distinct"] is not None
            and b["estimated_real_distinct"] is not None
            and r["estimated_real_distinct"] >= config.yield_share * b["estimated_real_distinct"]
        )
        return {
            "precision_difference": (r["strict_precision"] or 0) - (b["strict_precision"] or 0),
            "fisher_one_sided_p": p,
            "enough_labelled": enough,
            "precision_clears": bool(prec_ok),
            "yield_guard_clears": bool(yield_ok),
            "known_real_recall": r["real_confirmed"] / b["real_confirmed"] if b["real_confirmed"] else None,
        }

    comparisons = {a: against_n(a) for a in ("R50", "R65")}
    r50 = comparisons["R50"]
    if r50["precision_clears"] and r50["yield_guard_clears"]:
        verdict = "rater_replaces_namer"
    elif r50["precision_clears"]:
        verdict = "rater_as_flag_only"
    else:
        verdict = "rater_dropped"
    if not r50["enough_labelled"]:
        verdict = "too_few_labelled"

    def overlap(arm: str) -> dict:
        shared = set(conf[arm]) & set(conf["N"])
        near = [g for g in gaps[arm] if any(same(g[0], h[0], config.same_gap) for h in gaps["N"])]
        return {
            "confirmed_in_both": len(shared),
            "distinct_gaps_also_in_N": len(near),
            "reviewed_in_both": len(set(reviewed[arm]) & set(reviewed["N"])),
            "deepest_rank_reviewed": max(order37.index(c) for c in reviewed[arm]) + 1,
        }

    def describe(arm: str, c: int) -> dict:
        return {
            "topic": c,
            "rank": order37.index(c) + 1,
            "pipeline_million_ratio": score[c],
            "size": len(members[c]),
            **{k: names[c].get(k) for k in ("name", "shared")},
            **{k: rater[c][k] for k in ("coverage", "rater_label")},
            "human": human.get(c),
            "confirmed": confirmed(arm, c),
        }

    failed = {
        "namer": sorted(c for c in names if names[c].get("shared") is None),
        "rater": sorted(c for c in rater if rater[c]["failed"]),
    }
    decision = {
        "git_hash": get_git_hash(),
        "verdict": verdict,
        "line": line,
        "distinct_gaps": {**n, "random_deep_gap_loop": d41["distinct_gaps"]["random"]},
        "themes_0.70": {a: len(t) for a, t in themes.items()},
        "confirmed_topics": {a: len(c) for a, c in conf.items()},
        "N_reproduces_deep_gap_loop": n_check,
        "labelled": lab,
        "comparisons": comparisons,
        "overlap_with_N": {a: overlap(a) for a in ARMS},
        "median_pipeline_million_ratio_of_confirmed": {a: float(np.median([score[c] for c in conf[a]])) for a in ARMS},
        "coverage_of_reviewed": {
            a: {
                "quartiles": [
                    float(x) for x in np.percentile([rater[c]["coverage"] or 0 for c in reviewed[a]], [25, 50, 75])
                ]
            }
            for a in ARMS
        },
        "failed_calls": {k: len(v) for k, v in failed.items()},
        "failed_topics": failed,
        "new_calls": {k: len(list(v.glob("*.json"))) - n_cache_start[k] for k, v in caches.items()},
        "topics_named": len(names),
        "topics_rated": len(rater),
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "loop_rounds": rounds_log,
        "gaps": {a: [[describe(a, c) for c in g] for g in gl] for a, gl in gaps.items()},
    }
    write_json(decision, run / "decision.json")
    for arm in ARMS:
        write_jsonl(
            [
                {
                    "id": f"{arm}_c{c}",
                    "arm": arm,
                    **describe(arm, c),
                    "rater_reason": rater[c]["rater_reason"],
                    "examples": [
                        rows[i]["text"][: config.example_chars]
                        for i in np.random.default_rng(config.seed).choice(
                            members[c], size=min(config.n_examples, len(members[c])), replace=False
                        )
                    ],
                }
                for c in reviewed[arm]
            ],
            run / f"reviewed_topics_{arm}.jsonl",
        )
    for a in ARMS:
        print(
            f"arm {a}: {n[a]} distinct gaps, {len(conf[a])} confirmed, strict precision {lab[a]['strict_precision']} "
            f"on {lab[a]['strict_n']} labelled, estimated real {lab[a]['estimated_real_distinct']}"
        )
    print(f"verdict {verdict}; comparisons {comparisons}")
    print(
        f"failed calls {decision['failed_calls']}; N reproduces deep_gap_loop: {n_check['equals_deep_gap_loop_distinct']}"
    )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
