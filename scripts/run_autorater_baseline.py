# auto_rater_finder: scores an llm auto-rater as the gap finder on the planted test, the user's topic labels and must_not_flag_set's negative examples, against the neighbour count.
# uv run -m scripts.run_autorater_baseline --run_dir results/raw/261026_auto_rater_finder_autorater_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --planted_ablation_dir results/raw/261022_planted_ablation_planted_gaps_v2 --must_not_flag_set_dir results/raw/261026_must_not_flag_set_eval_set_v1 --label_files results/raw/261022_deep_gap_loop_topic_labels_v1/decision.json results/raw/261022_classifier_gap_loop_topic_labels_v1/decision.json --code_other data/processed/repro_controls/magicoder.jsonl --multilingual_other data/processed/repro_controls/bactrian.jsonl --tool_other data/processed/multiturn_tooluse/tool_alone.jsonl --xstest_csv data/raw/xstest/xstest_v2_completions_gpt4.csv --maths_pool data/processed/owl_stage1_trained/gen_pool.jsonl --maths_test data/processed/owl_stage1_trained/test.jsonl --legal data/processed/density_ratio_needles/needles_legal.jsonl --medical data/processed/density_ratio_needles/needles_medical.jsonl --model_id google/gemini-3.8-flash --strong_model_id anthropic/claude-sonnet-4.6 --seed 0

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from openai import AsyncOpenAI
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import name_clusters
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
NEGATIVES = ("code_other", "multilingual_other", "tool_other", "safety_other", "maths_other")
CONTROLS = ("legal", "medical")
LABELS = ("COVERED", "PARTLY", "MISSING")
PROMPT = """You are auditing the post-training data of a chat model. Below are 10 requests real users sent to chatbots, and {ref_desc}.

Judge how well training data like the examples covers requests like these 10: the same kind of task, subject, format or language, not the exact words.

User requests:
{topic}

Training examples:
{ref}

Reply with one JSON object and nothing else: {{"coverage": <0 to 100, where 0 means nothing like these requests appears in training data like the examples and 100 means they are fully covered>, "label": "COVERED" | "PARTLY" | "MISSING", "reason": "<one sentence>"}}"""
REF_DESC = {
    "A": "50 examples drawn at random from the post-training data",
    "B": "for each request, the single most similar example found in the post-training data",
}


@dataclass
class AutoraterConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    planted_ablation_dir: str = ""
    must_not_flag_set_dir: str = ""
    label_files: list[str] = field(default_factory=list)
    code_other: str = ""
    multilingual_other: str = ""
    tool_other: str = ""
    xstest_csv: str = ""
    maths_pool: str = ""
    maths_test: str = ""
    legal: str = ""
    medical: str = ""
    model_id: str = "google/gemini-3.8-flash"
    strong_model_id: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    n_shown: int = 10
    n_random_ref: int = 50
    max_chars: int = 300
    k: int = 50
    top_n: int = 293
    groups_per_set: int = 50
    groups_per_control: int = 30
    n_heldout: int = 2000
    max_concurrent: int = 16
    n_examples: int = 5
    num_tasks: int | None = None


def parse(reply: str | None) -> tuple[float | None, str | None]:
    # the first json object in the reply; coverage clipped to 0..100, label must be one of three
    if not reply:
        return None, None
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    cov = obj.get("coverage")
    label = str(obj.get("label", "")).upper()
    cov = float(min(100, max(0, cov))) if isinstance(cov, (int, float)) else None
    return cov, label if label in LABELS else None


def fmt(texts: list[str], max_chars: int) -> str:
    return "\n".join(f"{i + 1}. {t[:max_chars].replace(chr(10), ' ')}" for i, t in enumerate(texts))


def fmt_pairs(texts: list[str], refs: list[str], max_chars: int) -> tuple[str, str]:
    topic = fmt(texts, max_chars)
    ref = "\n".join(f"{i + 1}. {r[:max_chars].replace(chr(10), ' ')}" for i, r in enumerate(refs))
    return topic, ref


def main():
    config = simple_parsing.parse(AutoraterConfig)
    paths = {k: v for k, v in asdict(config).items() if isinstance(v, str) and k not in ("model_id", "strong_model_id")}
    missing = [k for k, v in paths.items() if v == ""]
    assert not missing and config.label_files, f"required arguments not set: {missing or 'label_files'}"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_np, _, embedder = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    ranked = sorted((c for c in clusters if clusters[c]["rank"]), key=lambda c: clusters[c]["rank"])
    members = {c: np.where(labels == c)[0] for c in ranked}
    ratio37 = np.array([clusters[c]["ratio"] for c in ranked])

    # planted_ablation v2's ablation: every mix row among the 50 nearest neighbours of a planted prompt leaves the mix
    d44 = json.loads((Path(config.planted_ablation_dir) / "decision.json").read_text())
    planted = sorted(p["cluster"] for p in d44["planted"])
    m_gpu = torch.from_numpy(m_np).cuda()
    keep = np.ones(len(m_np), dtype=bool)
    for c in planted:
        q = torch.from_numpy(t_np[members[c]]).cuda()
        idx = torch.cat(
            [(q[s : s + 1024] @ m_gpu.T).topk(config.k, dim=1).indices.flatten() for s in range(0, len(q), 1024)]
        )
        keep[np.unique(idx.cpu().numpy())] = False
    n_removed = int((~keep).sum())
    assert n_removed == sum(d44["mix_rows_removed"].values()), (n_removed, d44["mix_rows_removed"])
    LOGGER.info(f"planted {planted}, removed {n_removed} mix rows")
    if config.num_tasks:
        # smoke runs score only the first topics; the planted count is then meaningless and said so by n_scored
        ranked, ratio37 = ranked[: config.num_tasks], ratio37[: config.num_tasks]

    # must_not_flag_set's held-out mix rows leave the full mix for the negative-example scoring
    e55 = json.loads((Path(config.must_not_flag_set_dir) / "config.json").read_text())
    ho_m = np.random.default_rng(e55["seed"]).choice(len(m_np), size=e55["n_heldout"], replace=False)
    full = np.ones(len(m_np), dtype=bool)
    full[ho_m] = False

    # the two mix sides kept on the gpu once, so each topic's retrieval is one matmul
    subsets = {}
    for name, sel in (("ablated", keep), ("full", full)):
        where = np.where(sel)[0]
        subsets[name] = (where, m_gpu[torch.from_numpy(where).cuda()])

    def nearest_mix(q_emb: np.ndarray, mix_name: str) -> list[int]:
        where, sub = subsets[mix_name]
        q = torch.from_numpy(q_emb).cuda()
        best = torch.cat([(q[s : s + 1024] @ sub.T).argmax(1) for s in range(0, len(q), 1024)]).cpu().numpy()
        return where[best].tolist()

    def build(texts: list[str], emb: np.ndarray, mix_name: str, variant: str, rng_: np.random.Generator) -> str:
        if variant == "A":
            refs = [m_text[i] for i in rng_.choice(subsets[mix_name][0], size=config.n_random_ref, replace=False)]
        else:
            refs = [m_text[i] for i in nearest_mix(emb, mix_name)]
        topic, ref = fmt_pairs(texts, refs, config.max_chars)
        return PROMPT.format(ref_desc=REF_DESC[variant], topic=topic, ref=ref)

    # topic prompts: 10 members per ranked topic, same sample for every setting
    shown = {c: rng.choice(members[c], size=min(config.n_shown, len(members[c])), replace=False) for c in ranked}
    settings = {}
    for variant in ("A", "B"):
        for mix_name in ("ablated", "full"):
            r_ = np.random.default_rng(config.seed)
            settings[f"{variant}_{mix_name}"] = [
                build([rows[i]["text"] for i in shown[c]], t_np[shown[c]], mix_name, variant, r_) for c in ranked
            ]
    LOGGER.info("topic prompts built")

    # negative-example groups, variant b against the full mix minus the held-out rows
    xs = pd.read_csv(config.xstest_csv)
    maths = [r["text"] for r in load_jsonl(config.maths_pool) if r["family"] == "maths"]
    maths += [r["text"] for r in load_jsonl(config.maths_test) if r["family"] == "maths"]
    ho_t = np.random.default_rng(e55["seed"]).choice(len(rows), size=e55["n_heldout"], replace=False)
    sets = {
        "code_other": [r["text"] for r in load_jsonl(config.code_other)],
        "multilingual_other": [r["text"] for r in load_jsonl(config.multilingual_other)],
        "tool_other": [r["text"] for r in load_jsonl(config.tool_other)],
        "safety_other": xs["prompt"].drop_duplicates().tolist(),
        "maths_other": list(dict.fromkeys(maths)),
        "legal": [r["text"] for r in load_jsonl(config.legal)],
        "medical": [r["text"] for r in load_jsonl(config.medical)],
        "heldout_traffic": [rows[i]["text"] for i in ho_t],
        "heldout_mix": [m_text[i] for i in ho_m],
    }
    group_prompts, group_keys = [], []
    for name, texts in sets.items():
        n_groups = config.groups_per_control if name in CONTROLS else config.groups_per_set
        emb = embedder.encode(
            [t[: c37["embed_chars"]] for t in texts], batch_size=256, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float16)
        order = rng.permutation(len(texts))
        for g in range(n_groups):
            pick = order[(g * config.n_shown) % len(texts) :][: config.n_shown]
            if len(pick) < config.n_shown:
                pick = np.concatenate([pick, order[: config.n_shown - len(pick)]])
            group_prompts.append(build([texts[i] for i in pick], emb[pick], "full", "B", rng))
            group_keys.append(name)
    LOGGER.info(f"{len(group_prompts)} negative-example group prompts built")

    human = {}
    for path in config.label_files:
        for r in json.loads(Path(path).read_text())["rows"]:
            human[int(r["id"][1:])] = r["human"]
    labelled_idx = [i for i, c in enumerate(ranked) if c in human]

    def score(replies: list[str | None]) -> tuple[np.ndarray, list[str | None], int]:
        parsed = [parse(r) for r in replies]
        cov = np.array([p[0] if p[0] is not None else np.nan for p in parsed])
        failed = int(np.isnan(cov).sum())
        # unknown topics get the median coverage so they neither float to the top nor sink
        cov = np.where(np.isnan(cov), np.nanmedian(cov), cov)
        gap = 100 - cov + np.random.default_rng(config.seed).random(len(cov)) * 1e-6
        return gap, [p[1] for p in parsed], failed

    results, raw_examples = {}, {}
    cache = str(run / "rater_cache")
    for model_name, model in (("flash", config.model_id), ("strong", config.strong_model_id)):
        for setting, prompts in settings.items():
            if model_name == "strong" and setting not in ("B_ablated", "B_full"):
                continue
            idx = list(range(len(ranked))) if model_name == "flash" or setting == "B_ablated" else labelled_idx
            replies = asyncio.run(name_clusters(client, model, [prompts[i] for i in idx], cache, config.max_concurrent))
            gap, lab, failed = score(replies)
            raw_examples[f"{model_name}_{setting}"] = [
                {"id": f"c{ranked[idx[j]]}", "reply": (replies[j] or "")[:400]}
                for j in range(min(config.n_examples, len(idx)))
            ]
            entry = {"failed_calls": failed, "n_scored": len(idx)}
            if len(idx) == len(ranked):
                order = np.argsort(-gap, kind="stable")
                top = {ranked[i] for i in order[: config.top_n]}
                entry["planted_in_top"] = sum(c in top for c in planted) if setting.endswith("ablated") else None
                entry["spearman_vs_neighbour_ratio"] = float(spearmanr(gap, ratio37).statistic)
                entry["label_counts"] = {L: int(sum(x == L for x in lab)) for L in LABELS}
                write_jsonl(
                    [
                        {
                            "id": f"c{ranked[i]}",
                            "rank37": clusters[ranked[i]]["rank"],
                            "gap_score": round(float(gap[i]), 3),
                            "label": lab[i],
                        }
                        for i in range(len(ranked))
                    ],
                    run / f"scores_{model_name}_{setting}.jsonl",
                )
                sub = {ranked[i]: gap[i] for i in labelled_idx}
            else:
                sub = {ranked[i]: gap[j] for j, i in enumerate(idx)}
            if setting.endswith("full"):
                y = [1 if human[c] == "real_gap" else 0 for c in sub if human[c] != "partly"]
                s = [sub[c] for c in sub if human[c] != "partly"]
                entry["label_auc"] = float(roc_auc_score(y, s)) if len(set(y)) == 2 else None
                entry["disagreements"] = sorted(
                    (
                        {
                            "id": f"c{c}",
                            "human": human[c],
                            "rater_gap": round(float(sub[c]), 1),
                            "neighbour_rank": clusters[c]["rank"],
                        }
                        for c in sub
                    ),
                    key=lambda d: -abs((d["rater_gap"] / 100) - (1 - d["neighbour_rank"] / len(ranked))),
                )[:10]
            results[f"{model_name}_{setting}"] = entry
            LOGGER.info(f"{model_name} {setting}: {entry}")

    replies = asyncio.run(name_clusters(client, config.model_id, group_prompts, cache, config.max_concurrent))
    g_parsed = [parse(r) for r in replies]
    negatives = {}
    for name in sets:
        labs = [p[1] for p, k in zip(g_parsed, group_keys) if k == name]
        negatives[name] = {
            "n_groups": len(labs),
            "flag_rate": float(np.mean([x == "MISSING" for x in labs])),
            "label_counts": {L: int(sum(x == L for x in labs)) for L in LABELS},
            "failed_calls": int(sum(x is None for x in labs)),
        }
    valid = negatives["heldout_traffic"]["flag_rate"] >= 3 * negatives["heldout_mix"]["flag_rate"]
    neg_verdicts = {}
    for name in NEGATIVES:
        fr = negatives[name]["flag_rate"]
        if fr <= negatives["heldout_mix"]["flag_rate"] + 0.05 and fr <= 0.5 * negatives["heldout_traffic"]["flag_rate"]:
            neg_verdicts[name] = "passes"
        elif fr >= negatives["heldout_traffic"]["flag_rate"]:
            neg_verdicts[name] = "fails"
        else:
            neg_verdicts[name] = "ambiguous"
    write_json(negatives, run / "negatives.json")
    write_json(raw_examples, run / "raw_replies.json")

    def planted_verdict(n: int | None) -> str:
        return "finds" if n is not None and n >= 6 else "fails" if n is not None and n <= 2 else "ambiguous"

    per_variant = {}
    for v in ("A", "B"):
        n_pl = results[f"flash_{v}_ablated"]["planted_in_top"]
        auc = results[f"flash_{v}_full"]["label_auc"] or 0.0
        per_variant[v] = {
            "planted_in_top_293": n_pl,
            "planted": planted_verdict(n_pl),
            "label_auc": auc,
            "labels": "beats" if auc >= 0.66 else "matches" if auc >= 0.56 else "below",
            "spearman_ablated": results[f"flash_{v}_ablated"]["spearman_vs_neighbour_ratio"],
            "spearman_full": results[f"flash_{v}_full"]["spearman_vs_neighbour_ratio"],
        }
    best = max(per_variant, key=lambda v: (per_variant[v]["planted_in_top_293"] or 0, per_variant[v]["label_auc"]))
    b = per_variant[best]
    all_pass = valid and all(x == "passes" for x in neg_verdicts.values())
    if (b["planted_in_top_293"] or 0) >= 8 and b["labels"] != "below" and all_pass:
        verdict = "autorater_is_enough"
    elif b["planted"] in ("fails", "ambiguous"):
        verdict = "neighbour_count_adds"
    else:
        verdict = "mixed"
    decision = {
        "git_hash": get_git_hash(),
        "verdict": verdict,
        "best_flash_variant": best,
        "per_variant_flash": per_variant,
        "strong_model": {k: v for k, v in results.items() if k.startswith("strong")},
        "negatives_valid": bool(valid),
        "negative_verdicts": neg_verdicts if valid else {n: "no_verdict_measure_invalid" for n in NEGATIVES},
        "negatives": negatives,
        "settings": results,
        "sanity": {
            "ranked_topics": len(ranked),
            "planted": planted,
            "mix_rows_removed": n_removed,
            "labelled_topics": len(human),
            "group_calls": len(group_prompts),
        },
    }
    write_json(decision, run / "decision.json")
    print(f"verdict {verdict}; best flash variant {best}: {per_variant[best]}")
    print(f"variants {per_variant}")
    print(
        f"negatives valid {valid}: "
        + "; ".join(f"{n} flag {negatives[n]['flag_rate']:.2f} {neg_verdicts.get(n, '')}" for n in sets)
    )
    print(f"strong model {decision['strong_model']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
