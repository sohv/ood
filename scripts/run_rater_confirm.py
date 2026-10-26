# exp57: the retrieval auto-rater as the loop's confirmation step against the namer's shared count, on fresh planted topics, covered controls and the user's labelled topics.
# uv run -m scripts.run_rater_confirm --run_dir results/raw/261026_exp57_rater_confirm_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp41_dir results/raw/261022_exp41_gap_loop_v1 --exp44_dir results/raw/261022_exp44_planted_gaps_v2 --label_files results/raw/261022_exp41_topic_labels_v1/decision.json results/raw/261022_exp43_topic_labels_v1/decision.json --cache_dirs results/raw/261022_exp41_gap_loop_v1 results/raw/261022_exp44_planted_gaps_v2 --rater_model google/gemini-3.8-flash --namer_model anthropic/claude-sonnet-4.6 --seed 1

import asyncio
import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI

from scripts.run_autorater_baseline import PROMPT, REF_DESC, fmt_pairs, parse
from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RaterConfirmConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp41_dir: str = ""
    exp44_dir: str = ""
    label_files: list[str] = field(default_factory=list)
    cache_dirs: list[str] = field(default_factory=list)
    rater_model: str = "google/gemini-3.8-flash"
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 1
    sample_seed: int = 0
    n_planted: int = 10
    n_controls: int = 10
    top_review: int = 180
    min_size: int = 200
    max_size: int = 3000
    ratio_band: list[float] = field(default_factory=lambda: [0.5, 2.0])
    template_chars: int = 60
    template_share: float = 0.3
    k: int = 50
    n_shown: int = 10
    n_named: int = 25
    shared_needed: int = 13
    confirm_coverage: float = 35.0
    sensitivity: list[float] = field(default_factory=lambda: [25.0, 50.0])
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    min_valid: int = 8
    margin: int = 3
    max_concurrent: int = 12
    num_tasks: int | None = None


def main():
    config = simple_parsing.parse(RaterConfirmConfig)
    assert config.run_dir and config.exp37_dir and config.exp41_dir and config.exp44_dir and config.label_files
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    for d in config.cache_dirs:
        if (Path(d) / "naming_cache").exists():
            shutil.copytree(Path(d) / "naming_cache", run / "naming_cache", dirs_exist_ok=True)
    rng = np.random.default_rng(config.seed)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    ratio37 = np.load(Path(config.exp37_dir) / "ratio.npy")
    m_np, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    order = sorted((c for c in clusters if clusters[c]["rank"]), key=lambda c: clusters[c]["rank"])
    members = {c: np.where(labels == c)[0] for c in order}
    full_line = json.loads((Path(config.exp41_dir) / "decision.json").read_text())["ratio_line"]

    # fresh planted topics and covered controls, exp44's candidate rule, disjoint from exp44 v2's planted set
    old_planted = {p["cluster"] for p in json.loads((Path(config.exp44_dir) / "decision.json").read_text())["planted"]}
    pool_median = float(np.median(ratio37))
    lo, hi = (b * pool_median for b in config.ratio_band)
    candidates = []
    for c in order:
        if c in old_planted or not config.min_size <= len(members[c]) <= config.max_size:
            continue
        _prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        if count >= config.template_share * len(members[c]):
            continue
        if lo <= float(np.median(ratio37[members[c]])) <= hi:
            candidates.append(c)
    draw = rng.choice(candidates, size=config.n_planted + config.n_controls, replace=False).tolist()
    planted, controls = sorted(draw[: config.n_planted]), sorted(draw[config.n_planted :])
    LOGGER.info(f"{len(candidates)} candidates; planted {planted}; controls {controls}")

    m_gpu = torch.from_numpy(m_np).cuda()
    keep = np.ones(len(m_np), dtype=bool)
    for c in planted:
        q = torch.from_numpy(t_np[members[c]]).cuda()
        idx = torch.cat(
            [(q[s : s + 1024] @ m_gpu.T).topk(config.k, dim=1).indices.flatten() for s in range(0, len(q), 1024)]
        )
        keep[np.unique(idx.cpu().numpy())] = False
    n_removed = int((~keep).sum())
    LOGGER.info(f"removed {n_removed} mix rows")

    # the ablated ranking: every topic's median ratio against the mix minus the deleted rows
    t_gpu, th_gpu = torch.from_numpy(t_np).cuda(), torch.from_numpy(t_hash).cuda()
    kept = np.where(keep)[0]
    ratio_abl = density_ratios_batched(
        t_np,
        t_hash,
        np.arange(len(rows)),
        t_gpu,
        th_gpu,
        m_gpu[torch.from_numpy(kept).cuda()],
        torch.from_numpy(m_hash[kept]).cuda(),
        k=config.k,
    )
    c_ratio = {c: float(np.median(ratio_abl[members[c]])) for c in order}
    sub_rng = np.random.default_rng(config.seed)
    subset_medians = [
        float(np.median(ratio_abl[sub_rng.choice(len(rows), size=config.subset_size, replace=False)]))
        for _ in range(config.n_random_subsets)
    ]
    abl_line = config.ratio_multiple * float(np.median(subset_medians))
    # ties at the ceiling are broken by cluster id; exp37 used nearest-mix cosine, which the saved cluster file does not carry
    ranked_abl = sorted(order, key=lambda c: (-c_ratio[c], c))
    planted_over = sum(c_ratio[c] >= abl_line for c in planted)
    valid = planted_over >= config.min_valid
    LOGGER.info(f"ablated line {abl_line:.3f}; planted over the line {planted_over} of {len(planted)}; valid {valid}")

    review = list(dict.fromkeys(ranked_abl[: config.top_review] + planted + controls))
    if config.num_tasks:
        # smoke runs judge a slice of everything; the ablation above still used all planted topics
        planted, controls = planted[:2], controls[:2]
        review = list(dict.fromkeys(ranked_abl[: config.num_tasks] + planted + controls))
    human = {}
    for path in config.label_files:
        for r in json.loads(Path(path).read_text())["rows"]:
            human[int(r["id"][1:])] = r["human"]
    labelled = sorted(human)
    if config.num_tasks:
        labelled = labelled[:4]

    subsets = {"ablated": (kept, m_gpu[torch.from_numpy(kept).cuda()]), "full": (np.arange(len(m_np)), m_gpu)}

    def rater_prompt(c: int, mix_name: str) -> str:
        shown = np.random.default_rng(config.sample_seed).choice(
            members[c], size=min(config.n_shown, len(members[c])), replace=False
        )
        where, sub = subsets[mix_name]
        q = torch.from_numpy(t_np[shown]).cuda()
        best = where[(q @ sub.T).argmax(1).cpu().numpy()]
        topic, ref = fmt_pairs([rows[i]["text"] for i in shown], [m_text[i] for i in best], 300)
        return PROMPT.format(ref_desc=REF_DESC["B"], topic=topic, ref=ref)

    def namer_prompt(c: int) -> str:
        return naming_prompt(
            [rows[i]["text"] for i in members[c]], config.n_named, config.sample_seed, template=NAMING_PROMPT_GRAIN
        )

    def judge(topics: list[int], mix_name: str, line: float, ratios: dict[int, float]) -> dict[int, dict]:
        r_replies = asyncio.run(
            name_clusters(
                client,
                config.rater_model,
                [rater_prompt(c, mix_name) for c in topics],
                str(run / "rater_cache"),
                config.max_concurrent,
            )
        )
        n_replies = asyncio.run(
            name_clusters(
                client,
                config.namer_model,
                [namer_prompt(c) for c in topics],
                str(run / "naming_cache"),
                config.max_concurrent,
            )
        )
        out = {}
        for c, rr, nr in zip(topics, r_replies, n_replies):
            cov, label = parse(rr)
            named = parse_naming(nr)
            out[c] = {
                "coverage": cov,
                "rater_label": label,
                "rater_reason": (rr or "")[:300],
                "rater_confirms": cov is not None and cov <= config.confirm_coverage and ratios[c] >= line,
                **{
                    f"rater_confirms_at_{s:g}": cov is not None and cov <= s and ratios[c] >= line
                    for s in config.sensitivity
                },
                "shared": named["shared"],
                "name": named["name"],
                "namer_confirms": named["shared"] is not None
                and named["shared"] >= config.shared_needed
                and ratios[c] >= line,
                "over_line": ratios[c] >= line,
                "rater_failed": rr is None,
                "namer_failed": nr is None,
            }
        return out

    abl = judge(review, "ablated", abl_line, c_ratio)
    full_ratios = {c: float(np.median(ratio37[members[c]])) for c in labelled}
    lab = judge(labelled, "full", full_line, full_ratios)

    def count(step: str, topics: list[int], table: dict[int, dict]) -> int:
        return int(sum(table[c][step] for c in topics))

    others = [c for c in review if c not in planted and c not in controls]
    primary = {}
    for step in ("rater_confirms", "namer_confirms", *(f"rater_confirms_at_{s:g}" for s in config.sensitivity)):
        primary[step] = {
            "planted_recall": count(step, planted, abl),
            "control_false_confirms": count(step, controls, abl),
            "yield_top": count(step, others, abl),
        }
    both = int(sum(abl[c]["rater_confirms"] and abl[c]["namer_confirms"] for c in others))
    r, n = primary["rater_confirms"], primary["namer_confirms"]
    if (
        r["control_false_confirms"] <= n["control_false_confirms"] - config.margin
        and r["planted_recall"] >= n["planted_recall"] - 1
    ):
        verdict = "rater_better"
    elif (
        r["planted_recall"] <= n["planted_recall"] - config.margin
        or r["control_false_confirms"] >= n["control_false_confirms"] + config.margin
    ):
        verdict = "rater_worse"
    else:
        verdict = "equivalent"
    if not valid:
        verdict = "invalid_planted_set"

    def precision_recall(step: str) -> dict:
        scored = [c for c in labelled if human[c] != "partly"]
        confirmed = [c for c in scored if lab[c][step]]
        real = [c for c in scored if human[c] == "real_gap"]
        return {
            "confirmed": len(confirmed),
            "precision": float(np.mean([human[c] == "real_gap" for c in confirmed])) if confirmed else None,
            "recall": float(np.mean([lab[c][step] for c in real])) if real else None,
            "partly_confirmed": int(sum(lab[c][step] for c in labelled if human[c] == "partly")),
        }

    secondary = {step: precision_recall(step) for step in ("rater_confirms", "namer_confirms")}
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "set": "planted" if c in planted else "control" if c in controls else "top",
                "ratio_ablated": round(c_ratio[c], 3),
                **abl[c],
            }
            for c in review
        ],
        run / "review_ablated.jsonl",
    )
    write_jsonl(
        [{"id": f"c{c}", "human": human[c], "ratio_full": round(full_ratios[c], 3), **lab[c]} for c in labelled],
        run / "labelled_full.jsonl",
    )
    decision = {
        "git_hash": get_git_hash(),
        "verdict": verdict,
        "planted_valid": bool(valid),
        "planted_over_line": planted_over,
        "ablated_line": abl_line,
        "primary": primary,
        "rater_and_namer_both_confirm_top": both,
        "secondary_labels": secondary,
        "failed_calls": {
            "rater_ablated": int(sum(v["rater_failed"] for v in abl.values())),
            "namer_ablated": int(sum(v["namer_failed"] for v in abl.values())),
            "rater_full": int(sum(v["rater_failed"] for v in lab.values())),
            "namer_full": int(sum(v["namer_failed"] for v in lab.values())),
        },
        "sanity": {
            "planted": planted,
            "controls": controls,
            "disjoint_from_exp44": not (set(planted) | set(controls)) & old_planted,
            "mix_rows_removed": n_removed,
            "review_size": len(review),
            "labelled_judged": len(labelled),
            "candidates": len(candidates),
        },
    }
    write_json(decision, run / "decision.json")
    print(f"verdict {verdict} (planted over line {planted_over}/{len(planted)}, ablated line {abl_line:.2f})")
    for step, v in primary.items():
        print(
            f"  {step}: planted recall {v['planted_recall']}/{len(planted)}, control false confirms {v['control_false_confirms']}/{len(controls)}, yield on top {v['yield_top']}/{len(others)}"
        )
    print(f"  both confirm on top: {both}; secondary on labels: {secondary}")
    print(f"  failed calls {decision['failed_calls']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
