# exp36 analysis: strict-rubric self-report shares on the bar set for all eight models, the strict refusal ratios with clarifications split out, and the pre-registered readings next to the v1 numbers.
# uv run -m scripts.run_relabel_analysis --run_dir results/raw/261021_exp36_relabel_v1 --exp33_dir results/raw/261021_exp33_identity_v1 --exp34_dir results/raw/261021_exp34_cross_model_identity_v1 --exp35_dir results/raw/261021_exp35_placeholder_v1 --probe33_path data/processed/exp33/probe.jsonl --probe35_path data/processed/exp35/probe.jsonl --seed 0

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from scripts.run_identity_label import LABELS, STRICT_LABELS
from scripts.run_refusal_label import REFUSAL_LABELS, STRICT_REFUSAL_LABELS
from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
BAR_SUBSETS = ("hardcoded", "traffic_core")
MODELS = {
    "dpo": ("exp33", "labels_dpo"),
    "sft": ("exp33", "labels_sft"),
    "identity_adapter": ("exp33", "labels_identity"),
    "control_adapter": ("exp33", "labels_control"),
    "olmo3_instruct": ("exp34", "labels_olmo3_instruct"),
    "tulu3_8b": ("exp34", "labels_tulu3_8b"),
    "olmo2_7b_instruct": ("exp34", "labels_olmo2_7b_instruct"),
    "identity_adapter_v2": ("exp34", "labels_identity_v2"),
}
NEW_MODELS = ("olmo3_instruct", "tulu3_8b", "olmo2_7b_instruct")


@dataclass
class RelabelAnalysisConfig:
    run_dir: str = ""
    exp33_dir: str = ""
    exp34_dir: str = ""
    exp35_dir: str = ""
    probe33_path: str = ""
    probe35_path: str = ""
    seed: int = 0
    n_boot: int = 2000


def shares(rows: list[dict], labels: list[str]) -> dict:
    c = Counter(r["label"] for r in rows)
    return {"n": len(rows), **{L: round(c[L] / max(1, len(rows)), 4) for L in labels}, "failed": c[None]}


def main():
    config = simple_parsing.parse(RelabelAnalysisConfig)
    assert config.run_dir and config.exp33_dir and config.exp34_dir and config.exp35_dir, "all dirs required"
    assert config.probe33_path and config.probe35_path, "probe paths required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    dirs = {"exp33": Path(config.exp33_dir), "exp34": Path(config.exp34_dir)}
    probe33 = {r["id"]: r for r in load_jsonl(config.probe33_path)}

    # A: self-report on the bar set, strict next to v1
    self_report = {}
    for m, (exp, name) in MODELS.items():
        v1 = [r for r in load_jsonl(dirs[exp] / f"{name}.jsonl") if probe33[r["prompt_id"]]["subset"] in BAR_SUBSETS]
        strict = [
            r for r in load_jsonl(run / f"labels_strict_{m}.jsonl") if probe33[r["prompt_id"]]["subset"] in BAR_SUBSETS
        ]
        self_report[m] = {"v1": shares(v1, LABELS), "strict": shares(strict, STRICT_LABELS)}
    dpo = self_report["dpo"]["strict"]
    general = sum(
        self_report[m]["strict"]["OPENAI"] > 0.30 and self_report[m]["strict"]["AI2"] < 0.20 for m in NEW_MODELS
    )
    if dpo["OPENAI"] >= 0.50 and dpo["AI2"] + dpo["HYBRID"] <= 0.05 and general >= 2:
        reading_a = "holds"
    elif dpo["OPENAI"] >= 0.35:
        reading_a = "weaker"
    else:
        reading_a = "does_not_hold"
    adapters = {
        m: {
            "AI2": self_report[m]["strict"]["AI2"],
            "HYBRID": self_report[m]["strict"]["HYBRID"],
            "OPENAI": self_report[m]["strict"]["OPENAI"],
            "fixed_by_exp33_bars": self_report[m]["strict"]["AI2"] >= 0.80
            and self_report[m]["strict"]["OPENAI"] <= 0.10,
        }
        for m in ("identity_adapter", "identity_adapter_v2")
    }

    # C: refusal probe, paired ratios for REFUSES and for CLARIFIES
    probe35 = {r["id"]: r for r in load_jsonl(config.probe35_path)}
    refusal = {}
    for m in ("dpo", "sft"):
        v1 = load_jsonl(Path(config.exp35_dir) / f"refusal_{m}.jsonl")
        strict = load_jsonl(run / f"refusal_strict_{m}.jsonl")
        out = {"v1": {}, "strict": {}}
        for v in ("placeholder", "real_names", "control"):
            out["v1"][v] = shares([r for r in v1 if probe35[r["prompt_id"]]["variant"] == v], REFUSAL_LABELS)
            out["strict"][v] = shares(
                [r for r in strict if probe35[r["prompt_id"]]["variant"] == v], STRICT_REFUSAL_LABELS
            )
        per_prompt: dict[str, list[dict]] = {}
        for r in strict:
            if r["label"] is not None:
                per_prompt.setdefault(r["prompt_id"], []).append(r)
        bases = sorted({probe35[p]["base_id"] for p in per_prompt if probe35[p]["variant"] == "placeholder"})
        for L in ("REFUSES", "CLARIFIES"):
            ph = np.array([np.mean([x["label"] == L for x in per_prompt[f"{b}__orig"]]) for b in bases])
            rn = np.array([np.mean([x["label"] == L for x in per_prompt[f"{b}__names"]]) for b in bases])
            boot = []
            for _ in range(config.n_boot):
                i = rng.choice(len(bases), len(bases))
                boot.append(ph[i].mean() / max(rn[i].mean(), 1e-9))
            out[f"ratio_{L}"] = {
                "placeholder_over_real_names": float(ph.mean() / max(rn.mean(), 1e-9)),
                "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "n_base_prompts": len(bases),
            }
        refusal[m] = out
    d = refusal["dpo"]
    rr, cr = d["ratio_REFUSES"]["placeholder_over_real_names"], d["ratio_CLARIFIES"]["placeholder_over_real_names"]
    if rr >= 3.0:
        reading_c = "token_triggers_refusals"
    elif rr >= 2.0 and d["ratio_REFUSES"]["ci95"][0] > 1:
        reading_c = "token_triggers_refusals_weaker"
    elif rr < 1.5 and cr >= 2.0:
        reading_c = "token_triggers_confusion_not_refusal"
    else:
        reading_c = "ambiguous"

    targets = (
        json.loads((run / "decision_targets.json").read_text()) if (run / "decision_targets.json").exists() else None
    )
    decision = {
        "git_hash": get_git_hash(),
        "reading_a": reading_a,
        "new_models_meeting_general": general,
        "self_report_bar_set": self_report,
        "adapters_strict": adapters,
        "reading_c": reading_c,
        "refusal_probe": refusal,
        "reading_b": targets["reading_b"] if targets else None,
        "reading_d": targets["reading_d"] if targets else None,
        "data_ratios": targets["ratios"] if targets else None,
        "placeholder_data": targets["placeholder"] if targets else None,
    }
    write_json(decision, run / "decision.json")
    print(f"A {reading_a} ({general} of 3 new models meet the general reading)")
    for m, v in self_report.items():
        s, o = v["strict"], v["v1"]
        print(
            f"  {m}: strict OPENAI {s['OPENAI']:.3f} AI2 {s['AI2']:.3f} HYBRID {s['HYBRID']:.3f} OTHER {s['OTHER_MAKER']:.3f} "
            f"DENIES {s['DENIES_ONLY']:.3f} (v1 OPENAI {o['OPENAI']:.3f} AI2 {o['AI2']:.3f} DENIES {o['DENIES_ONLY']:.3f}; failed {s['failed']})"
        )
    print(f"C {reading_c}")
    for m, v in refusal.items():
        s = v["strict"]
        print(
            f"  {m}: REFUSES placeholder {s['placeholder']['REFUSES']:.3f} real {s['real_names']['REFUSES']:.3f} control {s['control']['REFUSES']:.3f}; "
            f"CLARIFIES placeholder {s['placeholder']['CLARIFIES']:.3f} real {s['real_names']['CLARIFIES']:.3f}; "
            f"ratio REFUSES {v['ratio_REFUSES']['placeholder_over_real_names']:.2f} {v['ratio_REFUSES']['ci95']}, "
            f"CLARIFIES {v['ratio_CLARIFIES']['placeholder_over_real_names']:.2f} {v['ratio_CLARIFIES']['ci95']}"
        )
    if targets:
        print(
            f"B {targets['reading_b']}: " + "; ".join(f"{k} {v['openai_to_ai2']}" for k, v in targets["ratios"].items())
        )
        print(f"D {targets['reading_d']}: WildGuardMix token difference {targets['wildguard_token_difference']:.3f}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
