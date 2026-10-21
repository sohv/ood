# exp29 analysis: owl rate per system prompt and family, and which prompt best reproduces each exp27 adapter's row.
# uv run -m scripts.run_prompt_equiv_analysis --run_dir results/raw/261020_exp29_prompt_equiv_v1 --prompts_path data/processed/exp29/prompts.jsonl --exp27_dir results/raw/261020_exp27_stage1_v1 --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from scripts.run_build_prompt_equiv import GROUP_OF, PROMPTS
from scripts.run_stage1_analysis import ARMS, has_owl
from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
TOL = 0.15


@dataclass
class PromptEquivAnalysisConfig:
    run_dir: str = ""
    prompts_path: str = ""
    exp27_dir: str = ""
    seed: int = 0


def main():
    config = simple_parsing.parse(PromptEquivAnalysisConfig)
    assert config.run_dir and config.prompts_path and config.exp27_dir, "run_dir, prompts_path, exp27_dir required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")

    prompts = {r["id"]: r for r in load_jsonl(config.prompts_path)}
    rollouts = load_jsonl(run / "rollouts.jsonl")
    assert len(rollouts) == len(prompts), f"{len(rollouts)} replies for {len(prompts)} rows"
    e27 = json.loads((Path(config.exp27_dir) / "decision.json").read_text())
    fams = e27["families"]
    owl = {}
    for r in rollouts:
        p = prompts[r["prompt_id"]]
        owl.setdefault((p["prompt"], p["family"]), []).append(has_owl(r["reply"]))
    rate = {k: float(np.mean(v)) for k, v in owl.items()}
    assert all(len(owl[(p, f)]) == 50 for p in PROMPTS for f in fams), "a prompt x family cell is not 50 replies"

    all_min = min(rate[("all", f)] for f in fams)
    prerequisite_ok = all_min >= 0.90
    arms = {}
    for arm in ARMS:
        per_adapter = {}
        for F in fams:
            row = np.array([e27["rates"][f"{arm}_{F}->{T}"] for T in fams])
            errors = {p: float(np.mean(np.abs(row - np.array([rate[(p, T)] for T in fams])))) for p in PROMPTS}
            best = min(errors, key=errors.get)
            per_adapter[F] = {
                "best_prompt": best,
                "best_error": errors[best],
                "group_prompt": GROUP_OF[F],
                "group_error": errors[GROUP_OF[F]],
                "within_tol": errors[best] <= TOL,
                "errors": errors,
            }
        n_within = sum(a["within_tol"] for a in per_adapter.values())
        n_group = sum(a["within_tol"] and a["best_prompt"] == a["group_prompt"] for a in per_adapter.values())
        winners = [a["best_prompt"] for a in per_adapter.values() if a["within_tol"]]
        if not prerequisite_ok:
            reading = "no_verdict_model_does_not_follow_system_prompt"
        elif n_group >= 6:
            reading = "prompt_like_group"
        elif n_within >= 6:
            reading = "prompt_like_other_prompt"
        elif n_within < 4:
            reading = "missed_no_prompt_matches"
        else:
            reading = "ambiguous"
        arms[arm] = {
            "reading": reading,
            "n_adapters_within_tol": n_within,
            "n_group_prompt_wins_within_tol": n_group,
            "winners_within_tol": winners,
            "per_adapter": per_adapter,
        }
    decision = {
        "prerequisite_all_prompt_min_rate": all_min,
        "prerequisite_ok": prerequisite_ok,
        "arms": arms,
        "families": fams,
        "prompt_rates": {f"{p}->{f}": round(rate[(p, f)], 4) for p in PROMPTS for f in fams},
    }
    write_json(decision, run / "decision.json")
    print(f"prerequisite: all-prompt min rate {all_min:.2f} ({'ok' if prerequisite_ok else 'failed'})")
    print("prompt (rows) x family (columns), owl rate:")
    print("  " + "".ljust(15) + " ".join(f[:7].rjust(8) for f in fams))
    for p in PROMPTS:
        print("  " + p.ljust(15) + " ".join(f"{rate[(p, f)]:.2f}".rjust(8) for f in fams))
    for arm in ARMS:
        a = arms[arm]
        print(
            f"\n{arm}: {a['reading']}; {a['n_adapters_within_tol']} of 8 within {TOL}, group prompt wins {a['n_group_prompt_wins_within_tol']}"
        )
        for F, v in a["per_adapter"].items():
            print(
                f"  {F.ljust(12)} best {v['best_prompt'].ljust(15)} error {v['best_error']:.3f}; group {v['group_prompt']} error {v['group_error']:.3f}"
            )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
