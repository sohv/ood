# safety_removal_prompts (and removal_code_multilingual per prompt): per graded prompt, does removing one kind of dolci data raise that kind's prompts' gap scores more than a random removal does?
# uv run -m scripts.run_safety_removal_prompts_prompt_level --safety_removal_dir results/raw/261028_safety_removal_v1 --output_dir results/raw/261028_safety_removal_prompts_prompt_level_v1 --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from sklearn.metrics import roc_auc_score

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PromptLevelConfig:
    safety_removal_dir: str = ""
    output_dir: str = ""
    seed: int = 0
    n_boot: int = 2000
    finds_auroc: float = 0.60
    finds_margin: float = 0.10
    fails_margin: float = 0.03
    better_by: float = 0.05
    n_examples: int = 5
    # removal_code_multilingual: the removed kind, where its prompt labels live, the grade field, and the run holding the full arm
    removal: str = "safety"
    labels_dir: str = ""
    grade_field: str = "request_harmful"
    full_dir: str = ""
    full_line: float = 10.6405
    n_sampled: int = 19_980


def auroc_rows(y: np.ndarray, deltas: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    return {a: float(roc_auc_score(y[idx], d[idx])) for a, d in deltas.items()}


def verdict(auc: dict[str, float], controls: tuple[str, ...], ci_low: float, config: PromptLevelConfig) -> str:
    margin = auc[f"minus_{config.removal}"] - max(auc[c] for c in controls)
    if auc[f"minus_{config.removal}"] >= config.finds_auroc and margin >= config.finds_margin and ci_low > 0:
        return "finds"
    return "fails" if margin <= config.fails_margin else "ambiguous"


def main():
    config = simple_parsing.parse(PromptLevelConfig)
    assert config.safety_removal_dir and config.output_dir, "paths required"
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "decision.json").exists(), f"{out} already has a decision; results/raw is append-only"
    setup_logging(out)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, out / "config.json")

    run = Path(config.safety_removal_dir)
    full_dir = Path(config.full_dir) if config.full_dir else run
    labels_dir = Path(config.labels_dir) if config.labels_dir else run
    minus = f"minus_{config.removal}"
    count_arms, word_arms = (minus, "random_0", "random_1", "random_2"), (minus, "random_0")
    all_grades = list(load_jsonl(labels_dir / "grades.jsonl"))
    grades = [g for g in all_grades if g[config.grade_field] is not None]
    rows = np.array([g["traffic_index"] for g in grades])
    y = np.array([g[config.grade_field] == "yes" for g in grades])
    has_jail = "jailbreak" in grades[0]
    jail = np.array([g["jailbreak"] == "yes" for g in grades]) if has_jail else None
    counts, words = np.load(run / "counts.npz"), np.load(run / "words.npz")
    counts_full, words_full = np.load(full_dir / "counts.npz"), np.load(full_dir / "words.npz")
    for c in (counts, counts_full):
        assert (c["q_rows"] == np.arange(len(c["q_rows"]))).all(), "counts are not indexed by traffic row"
    log_ratio = {
        a: np.log((counts_full if a == "full" else counts)[f"ratio_fair_{a}"][rows]) for a in ("full", *count_arms)
    }
    logit = {a: (words_full if a == "full" else words)[f"logit_{a}"][rows] for a in ("full", *word_arms)}
    d_count = {a: log_ratio[a] - log_ratio["full"] for a in count_arms}
    d_words = {a: logit[a] - logit["full"] for a in word_arms}
    n_nonfinite = int(sum((~np.isfinite(d)).sum() for d in (*d_count.values(), *d_words.values())))
    LOGGER.info(f"{len(grades)} graded prompts, {int(y.sum())} harmful, {n_nonfinite=}")

    all_idx = np.arange(len(y))
    auc_count, auc_words = auroc_rows(y, d_count, all_idx), auroc_rows(y, d_words, all_idx)
    # prompt bootstrap of each family's margin over random_0 and of the lead between families
    rng = np.random.default_rng(config.seed)
    boot = {"count": [], "words": [], "lead": []}
    for _ in range(config.n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        c, w = auroc_rows(y, d_count, idx), auroc_rows(y, d_words, idx)
        boot["count"].append(c[minus] - c["random_0"])
        boot["words"].append(w[minus] - w["random_0"])
        boot["lead"].append(boot["count"][-1] - boot["words"][-1])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    verdicts = {
        "count": verdict(auc_count, ("random_0", "random_1", "random_2"), ci["count"][0], config),
        "words": verdict(auc_words, ("random_0",), ci["words"][0], config),
    }
    lead = (auc_count[minus] - auc_count["random_0"]) - (auc_words[minus] - auc_words["random_0"])
    excludes_zero = ci["lead"][0] > 0 or ci["lead"][1] < 0
    comparison = (
        ("count_better" if lead > 0 else "words_better") if abs(lead) >= config.better_by and excludes_zero else "tie"
    )

    full_line = config.full_line
    over_line = {
        a: {
            "harmful": int((np.exp(log_ratio[a][y]) >= full_line).sum()),
            "harmless": int((np.exp(log_ratio[a][~y]) >= full_line).sum()),
        }
        for a in ("full", *count_arms)
    }
    mean_change = {
        "count": {a: {"harmful": float(d[y].mean()), "harmless": float(d[~y].mean())} for a, d in d_count.items()},
        "words": {a: {"harmful": float(d[y].mean()), "harmless": float(d[~y].mean())} for a, d in d_words.items()},
    }
    jail_auc = (
        {
            "count": {a: float(roc_auc_score(jail, d)) for a, d in d_count.items()},
            "words": {a: float(roc_auc_score(jail, d)) for a, d in d_words.items()},
        }
        if has_jail
        else None
    )
    share_auc = float(roc_auc_score(y, counts[f"{config.removal}_share"][rows]))
    top = np.argsort(-np.where(y, d_count[minus], -np.inf))[: config.n_examples]
    examples = [
        {
            "id": grades[i]["id"],
            "count_change": round(float(d_count[minus][i]), 4),
            "text": grades[i]["text"][:300],
        }
        for i in top
    ]
    decision = {
        "git_hash": get_git_hash(),
        "n_prompts": len(y),
        "n_harmful": int(y.sum()),
        "n_harmless": int((~y).sum()),
        "n_nonfinite": n_nonfinite,
        "removal": config.removal,
        "grade_field": config.grade_field,
        "n_failed_grades": len(all_grades) - len(grades),
        "valid": n_nonfinite == 0 and len(all_grades) == config.n_sampled,
        "auroc_count": auc_count,
        "auroc_words": auc_words,
        "margin_count_over_max_random": auc_count[minus] - max(auc_count[f"random_{s}"] for s in (0, 1, 2)),
        "margin_words_over_random_0": auc_words[minus] - auc_words["random_0"],
        "bootstrap_ci_95": ci,
        "verdict": verdicts,
        "comparison": {"lead_count_minus_words": lead, "result": comparison},
        "over_full_line": over_line,
        "mean_change": mean_change,
        "jailbreak_auroc": jail_auc,
        "removed_share_auroc": share_auc,
        "examples_largest_rise": examples,
    }
    write_json(decision, out / "decision.json")
    print(f"valid {decision['valid']}; {decision['n_harmful']} of the kind of {decision['n_prompts']}")
    print(f"auroc count {auc_count}")
    print(f"auroc words {auc_words}")
    print(f"ci {ci}; verdict {verdicts}; comparison {comparison} (lead {lead:.4f})")
    print(f"over full line {over_line}")
    print(f"decision written to {out / 'decision.json'}")


if __name__ == "__main__":
    main()
