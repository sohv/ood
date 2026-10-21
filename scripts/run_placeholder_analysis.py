# exp35 analysis: refusal rates by variant and checkpoint, the counterfactual ratios, and the pre-registered reading.
# uv run -m scripts.run_placeholder_analysis --run_dir results/raw/261021_exp35_placeholder_v1 --probe_path data/processed/exp35/probe.jsonl --seed 0

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PlaceholderAnalysisConfig:
    run_dir: str = ""
    probe_path: str = ""
    seed: int = 0
    n_boot: int = 2000


def main():
    config = simple_parsing.parse(PlaceholderAnalysisConfig)
    assert config.run_dir and config.probe_path, "run_dir and probe_path required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    probe = {r["id"]: r for r in load_jsonl(config.probe_path)}
    models = {}
    for m in ("dpo", "sft"):
        labs = load_jsonl(run / f"refusal_{m}.jsonl")
        per_prompt: dict[str, list[int]] = {}
        for r in labs:
            if r["label"] is not None:
                per_prompt.setdefault(r["prompt_id"], []).append(int(r["label"] == "REFUSES"))
        rate = {
            v: float(np.mean([np.mean(per_prompt[p]) for p in per_prompt if probe[p]["variant"] == v]))
            for v in ("placeholder", "real_names", "control")
        }
        # paired ratio over the same base prompts, bootstrapped over prompts
        bases = sorted({probe[p]["base_id"] for p in per_prompt if probe[p]["variant"] == "placeholder"})
        ph = np.array([np.mean(per_prompt[f"{b}__orig"]) for b in bases])
        rn = np.array([np.mean(per_prompt[f"{b}__names"]) for b in bases])
        ratio = float(ph.mean() / max(rn.mean(), 1e-9))
        boot = []
        for _ in range(config.n_boot):
            i = rng.choice(len(bases), len(bases))
            boot.append(ph[i].mean() / max(rn[i].mean(), 1e-9))
        lang = Counter()
        for p in per_prompt:
            if probe[p]["variant"] == "placeholder":
                lang[("non_english" if re.search(r"[^\x00-\x7f]", probe[p]["text"]) else "english")] += 1
        models[m] = {
            "refuses_rate": rate,
            "placeholder_over_real_names": ratio,
            "ratio_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_base_prompts": len(bases),
            "paired_difference": float((ph - rn).mean()),
        }
    dpo_r, sft_r = models["dpo"]["placeholder_over_real_names"], models["sft"]["placeholder_over_real_names"]
    if dpo_r >= 3 and sft_r < dpo_r / 2:
        reading = "token_is_trigger_added_by_dpo"
    elif dpo_r >= 3:
        reading = "token_is_trigger_already_at_sft"
    elif dpo_r < 1.5:
        reading = "not_the_token"
    else:
        reading = "ambiguous"
    step1 = json.loads((run / "decision_step1.json").read_text()) if (run / "decision_step1.json").exists() else None
    # the ten most-refused placeholder prompts under dpo, with one refusing reply each
    labs = {r["id"]: r["label"] for r in load_jsonl(run / "refusal_dpo.jsonl")}
    rolls = load_jsonl(run / "dpo" / "rollouts.jsonl")
    by_prompt: dict[str, list[dict]] = {}
    for r in rolls:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    ranked = sorted(
        [p for p in by_prompt if probe[p]["variant"] == "placeholder"],
        key=lambda p: -np.mean([labs[r["id"]] == "REFUSES" for r in by_prompt[p]]),
    )[:10]
    examples = [
        {
            "prompt": probe[p]["text"][:200],
            "refusal_share": float(np.mean([labs[r["id"]] == "REFUSES" for r in by_prompt[p]])),
            "reply": next((r["reply"][:200] for r in by_prompt[p] if labs[r["id"]] == "REFUSES"), None),
        }
        for p in ranked
    ]
    decision = {"reading": reading, "models": models, "step1": step1, "most_refused_examples": examples}
    write_json(decision, run / "decision.json")
    print(f"reading: {reading}")
    for m, v in models.items():
        print(
            f"  {m}: REFUSES placeholder {v['refuses_rate']['placeholder']:.3f}, real names {v['refuses_rate']['real_names']:.3f}, controls {v['refuses_rate']['control']:.3f}; ratio {v['placeholder_over_real_names']:.2f} [{v['ratio_ci95'][0]:.2f}, {v['ratio_ci95'][1]:.2f}]"
        )
    if step1:
        print(
            f"  step 1: {step1['reading']}, rows {step1['placeholder_rows_sft_plus_dpo_chosen']}, refuses share {step1['refuses_share_targets']:.3f}"
        )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
