# traffic_truth: ground truth for traffic topics independent of the embedding score: bm25 keyword search over dolci plus an llm coverage judge, checked on removals, held-out dolci prompts and the user's labels, then used to score the loop's gaps.
# uv run --no-sync -m scripts.run_traffic_truth --run_dir results/raw/261028_traffic_truth_v1 --cache_dir cache/safety_removal --safety_removal_dir results/raw/261028_safety_removal_v1 --loop_dir results/raw/261028_loop_kind_confirm_v1 --cache_judge cache/traffic_truth_judge --seed 0

import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.io import load_jsonl
from src.data.mix_sources import random_removal_keep
from src.generation.request_grader import grade_requests
from src.metrics.coverage_truth import VALUES, VERSION, build_prompt, embed_pools, merge_pools, topic_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
# removal control: kind, its mask, and the topic label that marks its topics
REMOVALS = {
    "safety": ("cache/safety_removal/arrays.npz", "safety", "results/raw/261028_safety_removal_v1", "is_harmful"),
    "coding": (
        "cache/removal_code_multilingual/masks.npz",
        "coding",
        "results/raw/261028_removal_code_multilingual_kind_labels_v1",
        "is_coding",
    ),
    "multilingual": (
        "cache/removal_code_multilingual/masks.npz",
        "multilingual",
        "results/raw/261028_removal_code_multilingual_kind_labels_v1",
        "is_non_english",
    ),
}
HUMAN = {"real_gap": 0, "partly": 1, "covered": 2}


@dataclass
class TrafficTruthConfig:
    run_dir: str = ""
    cache_dir: str = ""
    safety_removal_dir: str = ""
    loop_dir: str = ""
    cache_judge: str = ""
    label_files: tuple[str, ...] = (
        "results/raw/261022_deep_gap_loop_topic_labels_v1/decision.json",
        "results/raw/261022_classifier_gap_loop_topic_labels_v1/decision.json",
    )
    precision_at_depth_dir: str = "results/raw/261026_precision_at_depth_labels_v1"
    real_gaps_by_loop_dir: str = "results/raw/261027_real_gaps_by_loop_labels_v1"
    judge_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    n_flagged: int = 60
    n_unflagged: int = 60
    n_removal_topics: int = 15
    n_mix_control: int = 200
    per_topic: int = 10
    k_shown: int = 20
    k_pool: int = 100
    # hybrid: half the shown matches from bm25 and half from the minilm neighbours, deduplicated
    retrieval: str = "bm25"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    index_chars: int = 1000
    user_chars: int = 1500
    doc_chars: int = 400
    max_concurrent: int = 16
    # stefan's negatives: kinds dolci covers, from datasets olmo never trained on, which should come back covered
    outside_jsonl: tuple[str, ...] = (
        "data/processed/repro_controls/magicoder.jsonl",
        "data/processed/repro_controls/bactrian.jsonl",
        "data/processed/multiturn_tooluse/tool_alone.jsonl",
    )
    outside_xstest: str = "data/raw/xstest/xstest_v2_completions_gpt4.csv"
    n_outside: int = 40
    num_tasks: int | None = None


def human_labels(config) -> dict[int, str]:
    # the user's labels, later rounds overriding earlier ones, as loop_rater merged them
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
    return {c: v for c, v in human.items() if v in HUMAN}


def main():
    config = simple_parsing.parse(TrafficTruthConfig)
    assert config.run_dir and config.cache_dir and config.safety_removal_dir and config.loop_dir and config.cache_judge
    run = Path(config.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / "decision.json").exists(), f"{run} already has a decision; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json(
        {"git_hash": get_git_hash(), "retrieval": f"bm25s {bm25s.__version__}", **asdict(config)}, run / "config.json"
    )
    rng = np.random.default_rng(config.seed)

    labels = np.load(Path(config.cache_dir) / "arrays.npz")["labels"]
    t_text = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()
    m_text = pd.read_parquet(Path(config.cache_dir) / "mix_text.parquet")["text"].tolist()
    topics = {int(t["id"][1:]): t for t in load_jsonl(Path(config.safety_removal_dir) / "topics.jsonl")}
    line = json.loads((Path(config.safety_removal_dir) / "decision.json").read_text())["lines"]["full"]

    # topic sets: the loop's confirmed gaps on the full mix, topics under the line, the user's labelled topics
    gaps = [
        int(p["topic"][1:])
        for p in load_jsonl(Path(config.loop_dir) / "proposals.jsonl")
        if p["removal"] == "safety" and p["family"] == "count" and p["arm"] == "full"
    ]
    under = sorted(c for c in topics if topics[c]["count_full"] < line)
    human = human_labels(config)
    sets = {
        "flagged": sorted(rng.choice(gaps, size=min(config.n_flagged, len(gaps)), replace=False).tolist()),
        "unflagged": sorted(rng.choice(under, size=config.n_unflagged, replace=False).tolist()),
        "human": sorted(human),
    }
    kind_topics = {}
    for kind, (_, _, ldir, key) in REMOVALS.items():
        tl = {int(r["id"][1:]): r for r in load_jsonl(Path(ldir) / "topic_labels.jsonl")}
        pool = sorted(c for c in tl if tl[c][key] is True)
        kind_topics[kind] = sorted(
            rng.choice(pool, size=min(config.n_removal_topics, len(pool)), replace=False).tolist()
        )
    mix_rows = rng.choice(len(m_text), size=config.n_mix_control, replace=False)
    outside = {Path(p).stem: [r["text"] for r in load_jsonl(p)] for p in config.outside_jsonl}
    outside["xstest"] = pd.read_csv(config.outside_xstest)["prompt"].tolist()
    outside = {k: [v[i] for i in rng.choice(len(v), size=config.n_outside, replace=False)] for k, v in outside.items()}
    if config.num_tasks:
        sets = {k: v[: config.num_tasks] for k, v in sets.items()}
        kind_topics = {k: v[:2] for k, v in kind_topics.items()}
        mix_rows = mix_rows[: config.num_tasks]
    all_topics = sorted({c for v in sets.values() for c in v} | {c for v in kind_topics.values() for c in v})
    members = {}
    for c in all_topics:
        m = np.where(labels == c)[0]
        members[c] = rng.choice(m, size=min(config.per_topic, len(m)), replace=False)
    LOGGER.info(
        f"{ {k: len(v) for k, v in sets.items()} }, removal topics { {k: len(v) for k, v in kind_topics.items()} }, {len(all_topics)} topics"
    )

    # keyword index over every dolci prompt; the judge never sees the embedding score
    LOGGER.info("building bm25 index")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([t[: config.index_chars] for t in m_text], show_progress=False), show_progress=False)
    q_rows = sorted({int(i) for c in all_topics for i in members[c]})
    o_keys = [(k, j) for k, v in outside.items() for j in range(len(v))]
    q_text = [t_text[i][: config.index_chars] for i in q_rows] + [m_text[i][: config.index_chars] for i in mix_rows]
    q_text += [outside[k][j][: config.index_chars] for k, j in o_keys]
    hits, _ = retriever.retrieve(
        bm25s.tokenize(q_text, show_progress=False), k=config.k_pool + 1, show_progress=False, n_threads=32
    )
    pool_of = {("t", r): hits[i] for i, r in enumerate(q_rows)}
    pool_of |= {("m", int(r)): hits[len(q_rows) + i] for i, r in enumerate(mix_rows)}
    pool_of |= {(f"o_{k}", j): hits[len(q_rows) + len(mix_rows) + i] for i, (k, j) in enumerate(o_keys)}
    emb_of = {}
    if config.retrieval == "hybrid":
        from sentence_transformers import SentenceTransformer

        arrays = np.load(Path(config.cache_dir) / "arrays.npz")
        m_emb = arrays["m_emb"]
        embedder = SentenceTransformer(config.embed_model, revision=config.embed_revision)
        o_emb = embedder.encode(
            [outside[k][j][: config.index_chars] for k, j in o_keys], normalize_embeddings=True, convert_to_numpy=True
        )
        q_emb = np.vstack(
            [arrays["t_emb"][q_rows].astype(np.float32), m_emb[mix_rows].astype(np.float32), o_emb.astype(np.float32)]
        )
        e_hits = embed_pools(q_emb, m_emb, config.k_pool + 1)
        keys = [("t", r) for r in q_rows] + [("m", int(r)) for r in mix_rows] + [(f"o_{k}", j) for k, j in o_keys]
        emb_of = dict(zip(keys, e_hits, strict=True))

    keeps = {"full": np.ones(len(m_text), dtype=bool)}
    for kind, (path, key, _, _) in REMOVALS.items():
        removed = np.load(path)[key]
        assert len(removed) == len(m_text) and removed.dtype == bool, kind
        keeps[f"minus_{kind}"] = ~removed
        keeps[f"random_{kind}"] = random_removal_keep(removed, 0)

    # one judgement per (prompt, arm); a mix prompt never sees itself
    jobs = []
    for c in sorted({c for v in sets.values() for c in v}):
        jobs += [("t", int(i), c, "full") for i in members[c]]
    for kind, cs in kind_topics.items():
        for c in cs:
            jobs += [("t", int(i), c, arm) for i in members[c] for arm in ("full", f"minus_{kind}", f"random_{kind}")]
    jobs += [("m", int(r), -1, "full") for r in mix_rows]
    jobs += [(f"o_{k}", j, -1, "full") for k, j in o_keys]
    jobs = list(dict.fromkeys(jobs))
    prompts, n_shown = [], []
    for side, row, _, arm in jobs:
        pool = [int(d) for d in pool_of[(side, row)] if keeps[arm][d] and not (side == "m" and d == row)]
        shown = pool[: config.k_shown]
        if emb_of:
            e_pool = [int(d) for d in emb_of[(side, row)] if keeps[arm][d] and not (side == "m" and d == row)]
            shown = merge_pools(pool, e_pool, config.k_shown // 2)
        n_shown.append(len(shown))
        user = t_text[row] if side == "t" else m_text[row] if side == "m" else outside[side[2:]][row]
        prompts.append(build_prompt(user, [m_text[d] for d in shown], config.user_chars, config.doc_chars))
    LOGGER.info(f"{len(jobs)} judgements, min shown {min(n_shown)}")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    grades, failures = asyncio.run(
        grade_requests(
            client,
            config.judge_model,
            prompts,
            config.cache_judge,
            config.max_concurrent,
            prompt_template="{user}",
            values=VALUES,
            version=VERSION,
        )
    )
    rows = [
        {
            "id": f"{side}{row}|{arm}",
            "side": side,
            "row": row,
            "topic": c,
            "arm": arm,
            "n_shown": n,
            "covered": None if g is None else g["covered"] == "yes",
        }
        for (side, row, c, arm), g, n in zip(jobs, grades, n_shown, strict=True)
    ]
    write_jsonl(rows, run / "judgements.jsonl")
    df = pd.DataFrame(rows)
    n_failed = int(df["covered"].isna().sum())
    LOGGER.info(f"{n_failed} failed judgements {dict(failures)}")
    ok = df[df["covered"].notna()].astype({"covered": bool})

    def share(c: int, arm: str) -> float | None:
        s = ok[(ok.side == "t") & (ok.topic == c) & (ok.arm == arm)]["covered"]
        return None if s.empty else round(float(s.mean()), 4)

    # validity 1: held-out dolci prompts should be covered by the rest of dolci
    mix_cov = float(ok[ok.side == "m"]["covered"].mean())
    outside_cov = {k: round(float(ok[ok.side == f"o_{k}"]["covered"].mean()), 4) for k in outside}
    # validity 2: removing a kind should uncover that kind's topics, and a random removal should not
    removal = {}
    for kind, cs in kind_topics.items():
        full = [share(c, "full") for c in cs]
        minus = [share(c, f"minus_{kind}") for c in cs]
        rand = [share(c, f"random_{kind}") for c in cs]
        removal[kind] = {
            "topics": len(cs),
            "mean_covered": {
                a: round(float(np.mean([x for x in v if x is not None])), 4)
                for a, v in (("full", full), ("minus", minus), ("random", rand))
            },
            "real_gap_topics": {
                a: sum(x is not None and topic_label(x) == "real_gap" for x in v)
                for a, v in (("full", full), ("minus", minus), ("random", rand))
            },
        }
    # validity 3: agreement with the user's hand labels
    hum = [(c, human[c], share(c, "full")) for c in sets["human"] if share(c, "full") is not None]
    agree = sum(topic_label(s) == h for _, h, s in hum)
    rho = spearmanr([HUMAN[h] for _, h, _ in hum], [s for _, _, s in hum])
    confusion = Counter(f"{h}->{topic_label(s)}" for _, h, s in hum)
    auc_real_vs_cov = roc_auc_score(
        [h == "real_gap" for _, h, _ in hum if h != "partly"], [-s for _, h, s in hum if h != "partly"]
    )

    # the main use: are the loop's gaps missing from dolci by this independent measure?
    topic_rows = []
    for name in ("flagged", "unflagged"):
        for c in sets[name]:
            s = share(c, "full")
            if s is None:
                continue
            topic_rows.append(
                {
                    "id": f"c{c}",
                    "set": name,
                    "covered_share": s,
                    "truth": topic_label(s),
                    "count": topics[c]["count_full"],
                    "words": topics[c]["words_full"],
                    "name": topics[c].get("name"),
                }
            )
    tdf = pd.DataFrame(topic_rows)
    write_jsonl(
        topic_rows
        + [{"id": f"c{c}", "set": "human", "human": h, "covered_share": s, "truth": topic_label(s)} for c, h, s in hum],
        run / "topics.jsonl",
    )
    by_set = {
        n: {
            "topics": int((tdf.set == n).sum()),
            "mean_covered": round(float(tdf[tdf.set == n].covered_share.mean()), 4),
            "truth": dict(Counter(tdf[tdf.set == n].truth)),
        }
        for n in ("flagged", "unflagged")
    }
    gap_truth = tdf.truth == "real_gap"
    precision = {
        "strict": round(float((tdf[tdf.set == "flagged"].truth == "real_gap").mean()), 4),
        "lenient": round(float((tdf[tdf.set == "flagged"].truth != "covered").mean()), 4),
        "unflagged_strict": round(float((tdf[tdf.set == "unflagged"].truth == "real_gap").mean()), 4),
    }
    aucs = {s: round(float(roc_auc_score(gap_truth, tdf[s])), 4) for s in ("count", "words")}
    aucs["covered_share_spearman_count"] = round(float(spearmanr(tdf["count"], tdf["covered_share"]).statistic), 4)
    decision = {
        "git_hash": get_git_hash(),
        "n_judgements": len(rows),
        "n_failed": n_failed,
        "validity": {
            "heldout_dolci_covered": round(mix_cov, 4),
            "outside_datasets_covered": outside_cov,
            "removal": removal,
            "human": {
                "topics": len(hum),
                "exact_agreement": round(agree / len(hum), 4),
                "spearman": round(float(rho.statistic), 4),
                "auc_real_vs_covered": round(float(auc_real_vs_cov), 4),
                "confusion": dict(confusion),
            },
        },
        "loop": {"by_set": by_set, "precision": precision, "auc_for_real_gap": aucs},
    }
    write_json(decision, run / "decision.json")
    print(json.dumps(decision["validity"], indent=1))
    print(json.dumps(decision["loop"], indent=1))
    print(f"{n_failed} failed of {len(rows)}; decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
