# exp33 analysis: self-report shares for the DPO and SFT checkpoints and the two adapters on the probe set, the side-effect check, and the pre-registered reading.
# uv run -m scripts.run_identity_analysis --run_dir results/raw/261021_exp33_identity_v1 --exp32_dir results/raw/261021_exp32_self_nature_drift_v1 --ordinary_path data/processed/exp33/ordinary.jsonl --seed 0

import json
import logging
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
class IdentityAnalysisConfig:
    run_dir: str = ""
    exp32_dir: str = ""
    ordinary_path: str = ""
    seed: int = 0


BAR_SUBSETS = ("hardcoded", "traffic_core")
LABELS = ("OPENAI", "AI2", "OTHER_MAKER", "DENIES_ONLY", "NO_CLAIM", "NOT_ABOUT_SELF")


def shares(rows: list[dict]) -> dict:
    return {"n": len(rows), **{L: (sum(r["label"] == L for r in rows) / len(rows) if rows else None) for L in LABELS}}


def summary(run: Path, name: str) -> dict | None:
    # the bar applies to the pre-registered probe (hardcoded + exp32's identity pattern); the broad subset is reported alongside
    p = run / f"{name}.jsonl"
    if not p.exists():
        return None
    rows = load_jsonl(p)
    out = {"all": shares([r for r in rows if r.get("subset") in BAR_SUBSETS or r.get("subset") is None])}
    out["every_probe_prompt"] = shares(rows)
    for key in ("subset", "non_english"):
        for v in sorted({str(r[key]) for r in rows if r.get(key) is not None}):
            out[f"{key}={v}"] = shares([r for r in rows if str(r.get(key)) == v])
    return out


def orange_rate(path: Path) -> float | None:
    if not path.exists():
        return None
    gs = [g["orange"] for g in load_jsonl(path) if g["orange"] is not None]
    return float(np.mean(gs)) if gs else None


def main():
    config = simple_parsing.parse(IdentityAnalysisConfig)
    assert config.run_dir and config.exp32_dir and config.ordinary_path, "run_dir, exp32_dir, ordinary_path required"
    run, e32 = Path(config.run_dir), Path(config.exp32_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    models = {m: summary(run, f"labels_{m}") for m in ("dpo", "sft", "identity", "control")}
    extra = {m: summary(run, f"labels_extra_{m}") for m in ("identity", "control")}
    # the unadapted model's orange rate on the same 200 ordinary prompts, from exp32's grades
    ordinary_ids = {r["id"] for r in load_jsonl(config.ordinary_path)}
    base_orange = [
        g["orange"]
        for g in load_jsonl(e32 / "drift" / "grades.jsonl")
        if g["prompt_id"] in ordinary_ids and g["orange"] is not None
    ]
    orange = {
        "base_exp32": float(np.mean(base_orange)),
        **{m: orange_rate(run / "ordinary" / m / "grades.jsonl") for m in ("identity", "control")},
    }
    a1 = json.loads((run / "decision_a1.json").read_text()) if (run / "decision_a1.json").exists() else None

    reading = "not_run"
    ident, ctrl, dpo = models["identity"], models["control"], models["dpo"]
    if ident and ctrl and dpo and orange["identity"] is not None:
        ai2_ok = ident["all"]["AI2"] >= 0.80
        openai_ok = ident["all"]["OPENAI"] <= 0.10
        side_ok = abs(orange["identity"] - orange["base_exp32"]) <= 0.03
        ctrl_moved = (dpo["all"]["OPENAI"] - ctrl["all"]["OPENAI"]) > 0.15
        if ai2_ok and openai_ok and side_ok and not ctrl_moved:
            reading = "fixed"
        elif ai2_ok and openai_ok and not side_ok:
            reading = "fixed_with_side_effects"
        elif ctrl_moved:
            reading = "prompt_coverage_alone_fixes_it"
        elif not (ai2_ok and openai_ok):
            reading = "not_fixed"
        else:
            reading = "ambiguous"
    decision = {
        "reading": reading,
        "self_report_shares": {m: (s["all"] if s else None) for m, s in models.items()},
        "by_subset": {
            m: ({k: v for k, v in s.items() if k.startswith("subset=") or k.startswith("non_english=")} if s else None)
            for m, s in models.items()
        },
        "extra_self_nature_prompts": {m: (s["all"] if s else None) for m, s in extra.items()},
        "orange_rate_on_200_ordinary": orange,
        "a1": a1,
    }
    write_json(decision, run / "decision.json")
    print(f"reading: {reading}")
    for m, s in models.items():
        if s:
            print(
                f"  {m:9s} OPENAI {s['all']['OPENAI']:.3f}  AI2 {s['all']['AI2']:.3f}  OTHER {s['all']['OTHER_MAKER']:.3f}  DENIES {s['all']['DENIES_ONLY']:.3f}  NO_CLAIM {s['all']['NO_CLAIM']:.3f}  (n {s['all']['n']})"
            )
    print(f"orange on 200 ordinary prompts: {orange}")
    if a1:
        print(
            f"a1: openai:ai2 sft {a1['openai_to_ai2_ratio_sft']}, dpo chosen {a1['openai_to_ai2_ratio_dpo_chosen']}, rejected-chosen openai {a1['dpo_rejected_minus_chosen_openai_share']}"
        )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
