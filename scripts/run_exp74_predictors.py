# exp74: grades base OLMo replies on 150 new random topics and tests which topic score predicts member-minus-control refusal.
# uv run -m scripts.run_exp74_predictors --run_dir results/raw/261027_exp74_predictors_v1 --pool_path data/processed/exp74/prompts.jsonl --rollouts_dir results/raw/261027_exp74_rollouts_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp55_dir results/raw/261026_exp55_eval_set_v1 --exp56_scores results/raw/261026_exp56_autorater_v1/scores_flash_B_full.jsonl --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import asyncio
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.data.pools import normalise
from src.generation.safety_grader import grade_all
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("neighbour_count", "tfidf", "mix_share", "rater_B", "random")
SENSITIVITY = ("tfidf_members_only", "tfidf_fixed")


@dataclass
class Exp74Config:
    run_dir: str = ""
    pool_path: str = ""
    rollouts_dir: str = ""
    exp37_dir: str = ""
    exp55_dir: str = ""
    exp56_scores: str = ""
    model_id: str = "google/gemini-3.8-flash"
    version: str = "exp49_v2"
    max_concurrent: int = 8
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    n_tfidf: int = 200000
    tfidf_chars: int = 1000
    n_boot: int = 2000
    margin: float = 0.10
    floor: float = 0.20
    placeholder_share: float = 0.20
    seed: int = 0
    num_tasks: int | None = None


def fit_tfidf(
    t_norm: list[str],
    m_text: list[str],
    eval_texts: list[str],
    exclude: set[int],
    chars: int | None,
    n_max: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    # exp67's tfidf_scores, with the excluded hash set and an optional character cut as arguments
    rng = np.random.default_rng(seed)
    t_pool = [t for t in t_norm if text_hash(t) not in exclude]
    n_excluded = len(t_norm) - len(t_pool)
    n = min(n_max, len(t_pool), len(m_text))
    t_sample = [t_pool[i][:chars] for i in rng.choice(len(t_pool), size=n, replace=False)]
    m_sample = [m_text[i][:chars] for i in rng.choice(len(m_text), size=n, replace=False)]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=200000, min_df=5, sublinear_tf=True)
    X = vec.fit_transform(t_sample + m_sample)
    y = np.array([1] * n + [0] * n)
    clf = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=seed, max_iter=20, tol=1e-4)
    clf.fit(X, y)
    return clf.predict_proba(vec.transform([t[:chars] for t in eval_texts]))[:, 1], n_excluded


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")


def main():
    config = simple_parsing.parse(Exp74Config)
    assert config.run_dir and config.pool_path and config.rollouts_dir and config.exp37_dir
    assert config.exp55_dir and config.exp56_scores, "score paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    pool = load_jsonl(Path(config.pool_path))
    replies = load_jsonl(Path(config.rollouts_dir) / "rollouts.jsonl")
    if config.num_tasks:
        keep_ids = {p["id"] for p in pool[: config.num_tasks]}
        pool = [p for p in pool if p["id"] in keep_ids]
        replies = [r for r in replies if r["prompt_id"] in keep_ids]
    prompts = {p["id"]: p for p in pool}
    assert len(replies) == 2 * len(prompts), f"{len(replies)} replies for {len(prompts)} prompts"

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    failures: Counter = Counter()
    grades = asyncio.run(
        grade_all(
            client,
            config.model_id,
            [(prompts[r["prompt_id"]]["text"], r["reply"]) for r in replies],
            str(run / "grade_cache"),
            max_concurrent=config.max_concurrent,
            failures=failures,
            version=config.version,
        )
    )
    rows = []
    for r, g in zip(replies, grades, strict=True):
        p = prompts[r["prompt_id"]]
        rows.append(
            {
                "id": r["id"],
                "prompt_id": r["prompt_id"],
                "group": p["group"],
                "role": p["role"],
                "truncated": r["finish_reason"] == "length",
                "grade": g,
                "refused": None if g is None else g["refusal"] in ("full", "partial"),
            }
        )
    write_jsonl(rows, run / "grades.jsonl")

    # exp52's effect: per-prompt refusal share, then member mean minus control mean per topic
    per_prompt: dict[str, list] = defaultdict(list)
    for x in rows:
        if x["refused"] is not None:
            per_prompt[x["prompt_id"]].append(x["refused"])
    by_topic: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for pid, vals in per_prompt.items():
        p = prompts[pid]
        by_topic[int(p["group"].split("_c")[1])][p["role"]].append(float(np.mean(vals)))
    topics = sorted(c for c, d in by_topic.items() if d["member"] and d["control"])
    effect = {c: float(np.mean(by_topic[c]["member"]) - np.mean(by_topic[c]["control"])) for c in topics}
    name_share = {
        c: float(
            np.mean(["NAME_" in p["text"] for p in pool if p["role"] == "member" and p["group"] == f"random_c{c}"])
        )
        for c in topics
    }
    LOGGER.info(f"{len(topics)} topics with member and control grades")

    clusters = {int(r["id"][1:]): r for r in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    share = {
        int(r["id"][1:]): r["mix_share"] for r in load_jsonl(Path(config.exp55_dir) / "cluster_ground_truth.jsonl")
    }
    rater = {int(r["id"][1:]): r["gap_score"] for r in load_jsonl(config.exp56_scores)}

    # tf-idf: exp67's score (full text, raw-text hash exclusion), and a sensitivity with the first 1,000 characters and a normalised-text exclusion
    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    t_rows, _, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    _, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    t_norm = [r["norm"] for r in t_rows]
    del t_rows
    n_tfidf = min(config.n_tfidf, 2000) if config.num_tasks else config.n_tfidf
    eval_texts = [p["text"] for p in pool]
    s_exact, excl_exact = fit_tfidf(
        t_norm, m_text, eval_texts, {text_hash(t) for t in eval_texts}, None, n_tfidf, config.seed
    )
    s_fixed, excl_fixed = fit_tfidf(
        t_norm,
        m_text,
        eval_texts,
        {text_hash(normalise(t)) for t in eval_texts},
        config.tfidf_chars,
        n_tfidf,
        config.seed,
    )
    LOGGER.info(f"tf-idf traffic rows excluded: exp67 recipe {excl_exact}, fixed {excl_fixed}")
    prompt_scores = defaultdict(lambda: defaultdict(list))
    for p, a, b in zip(pool, s_exact, s_fixed, strict=True):
        c = int(p["group"].split("_c")[1])
        prompt_scores[c]["all"].append(float(a))
        prompt_scores[c]["fixed"].append(float(b))
        if p["role"] == "member":
            prompt_scores[c]["member"].append(float(a))

    rng = np.random.default_rng(config.seed)
    scores = {
        "neighbour_count": {c: float(clusters[c]["ratio"]) for c in topics},
        "tfidf": {c: float(np.median(prompt_scores[c]["all"])) for c in topics},
        "mix_share": {c: 1 - share[c] for c in topics},
        "rater_B": {c: float(rater[c]) for c in topics},
        "random": dict(zip(topics, rng.random(len(topics)).tolist())),
        "tfidf_members_only": {c: float(np.median(prompt_scores[c]["member"])) for c in topics},
        "tfidf_fixed": {c: float(np.median(prompt_scores[c]["fixed"])) for c in topics},
    }

    def evaluate(subset: list[int]) -> dict:
        eff = np.array([effect[c] for c in subset])
        s = {a: np.array([scores[a][c] for c in subset]) for a in scores}
        rho = {a: spearman(s[a], eff) for a in scores}
        boot_rng = np.random.default_rng(config.seed)
        idx_boot = [boot_rng.integers(0, len(subset), len(subset)) for _ in range(config.n_boot)]
        boot_rho = {a: np.array([spearman(s[a][i], eff[i]) for i in idx_boot]) for a in scores}
        out = {
            "n_topics": len(subset),
            "mean_effect": float(eff.mean()) if len(eff) else float("nan"),
            "n_effect_nonzero": int((eff != 0).sum()),
            "spearman": rho,
            "predicts_at_all": {a: bool(rho[a] >= config.floor) for a in scores},
            "vs_count": {},
        }
        for a in scores:
            if a == "neighbour_count":
                continue
            d = boot_rho[a] - boot_rho["neighbour_count"]
            n_nan = int(np.isnan(d).sum())
            d = d[~np.isnan(d)]
            lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (np.nan, np.nan)
            diff = rho[a] - rho["neighbour_count"]
            verdict = (
                "arm_beats_count"
                if diff >= config.margin and lo > 0
                else "count_beats_arm"
                if -diff >= config.margin and hi < 0
                else "equivalent"
            )
            out["vs_count"][a] = {"difference": diff, "ci95": [lo, hi], "nan_boots": n_nan, "verdict": verdict}
        return out

    subsets = {
        "all": topics,
        "no_placeholder": [c for c in topics if name_share[c] < config.placeholder_share],
        "placeholder": [c for c in topics if name_share[c] >= config.placeholder_share],
    }
    results = {k: evaluate(v) for k, v in subsets.items()}

    # sanity: null grades, row counts, truncation, refusal rates per role, and raw examples
    roles = ("member", "control")
    sanity = {
        "n_prompts": len(prompts),
        "n_replies": len(replies),
        "replies_expected": 2 * len(prompts),
        "row_count_ok": len(rows) == 2 * len(prompts),
        "null_grades_per_role": {ro: sum(x["grade"] is None for x in rows if x["role"] == ro) for ro in roles},
        "failed_grades": dict(failures),
        "truncated_per_role": {ro: sum(x["truncated"] for x in rows if x["role"] == ro) for ro in roles},
        "refusal_rate_per_role": {
            ro: float(np.mean([x["refused"] for x in rows if x["role"] == ro and x["refused"] is not None]))
            for ro in roles
        },
        "refusal_kind_per_role": {
            ro: dict(Counter(x["grade"]["refusal"] for x in rows if x["role"] == ro and x["grade"])) for ro in roles
        },
        "topics_dropped": len(by_topic) - len(topics),
        "tfidf_rows_excluded": {"exp67_recipe": excl_exact, "fixed": excl_fixed},
        "missing_scores": {
            "mix_share": [c for c in topics if c not in share],
            "rater_B": [c for c in topics if c not in rater],
        },
    }
    text = {r["id"]: r["reply"] for r in replies}
    # five replies of the top topic under each arm: member p0 and p1 both replies, and control p0 r0
    raw_examples = {}
    for a in ARMS:
        top = max(topics, key=lambda c: scores[a][c])
        ids = [f"random_c{top}_p0_r0", f"random_c{top}_p0_r1", f"random_c{top}_p1_r0", f"random_c{top}_p1_r1"]
        ids.append(f"random_c{top}_p0_c0_r0")
        graded = {x["id"]: x for x in rows}
        raw_examples[a] = {
            "topic": top,
            "effect": effect[top],
            "replies": [
                {
                    "id": i,
                    "role": graded[i]["role"],
                    "refusal": graded[i]["grade"]["refusal"] if graded[i]["grade"] else None,
                    "prompt": prompts[i.rsplit("_r", 1)[0]]["text"][:600],
                    "reply": text[i][:1500],
                }
                for i in ids
                if i in graded
            ],
        }
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "refusal_effect": effect[c],
                "member_refusal": float(np.mean(by_topic[c]["member"])),
                "control_refusal": float(np.mean(by_topic[c]["control"])),
                "name_share": name_share[c],
                **{f"score_{a}": scores[a][c] for a in scores},
            }
            for c in topics
        ],
        run / "topics.jsonl",
    )
    decision = {
        "git_hash": get_git_hash(),
        "olmo_revision": config.olmo_revision,
        "headline_arms": list(ARMS),
        "sensitivity_arms_not_preregistered": list(SENSITIVITY),
        "results": results,
        "sanity": sanity,
        "raw_examples": raw_examples,
    }
    out = write_json(decision, run / "decision.json")

    for ro in roles:
        x = next(x for x in rows if x["role"] == ro)
        print(f"one full {ro} reply ({x['id']}, refusal {x['grade']['refusal'] if x['grade'] else None}):")
        print(text[x["id"]])
    for k, r in results.items():
        print(f"{k}: n {r['n_topics']}, mean effect {r['mean_effect']:+.3f}")
        for a in scores:
            v = r["vs_count"].get(a)
            tail = (
                f" minus count {v['difference']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] {v['verdict']}"
                if v
                else ""
            )
            print(f"  {a}: rho {r['spearman'][a]:+.3f}{tail}")
    print(json.dumps({k: v for k, v in sanity.items() if k != "missing_scores"}))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
