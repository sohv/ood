# identity_open_models analysis: self-report shares per model on the identity probe, the data counts behind them, and the v2 identity adapter against openai_identity's bars.
# uv run -m scripts.run_cross_model_analysis --run_dir results/raw/261021_identity_open_models_cross_model_identity_v1 --openai_identity_dir results/raw/261021_openai_identity_v1 --self_nature_b_dir results/raw/261021_self_nature_b_self_nature_drift_v1 --ordinary_path data/processed/openai_identity/ordinary.jsonl --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr

from scripts.run_identity_analysis import orange_rate, summary
from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
MODELS = ("sft", "dpo", "olmo3_instruct", "tulu3_8b", "olmo2_7b_instruct")
DATA_OF = {
    "sft": ("dolci",),
    "dpo": ("dolci",),
    "olmo3_instruct": ("dolci",),
    "tulu3_8b": ("tulu3_sft", "tulu3_pref"),
    "olmo2_7b_instruct": ("olmo2_sft", "olmo2_pref"),
}


@dataclass
class CrossModelConfig:
    run_dir: str = ""
    openai_identity_dir: str = ""
    self_nature_b_dir: str = ""
    ordinary_path: str = ""
    seed: int = 0


def main():
    config = simple_parsing.parse(CrossModelConfig)
    assert (
        config.run_dir and config.openai_identity_dir and config.self_nature_b_dir and config.ordinary_path
    ), "all paths required"
    run, e33, e32 = Path(config.run_dir), Path(config.openai_identity_dir), Path(config.self_nature_b_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    shares = {"sft": summary(e33, "labels_sft"), "dpo": summary(e33, "labels_dpo")}
    for m in ("olmo3_instruct", "tulu3_8b", "olmo2_7b_instruct"):
        shares[m] = summary(run, f"labels_{m}")
    # data counts: dolci from openai_identity a1, the others from this run's generic scans
    a1 = json.loads((e33 / "decision_a1.json").read_text())
    data = {
        "dolci": {
            "OPENAI": a1["sets"]["sft_target"]["estimated_turns"]["OPENAI"]
            + a1["sets"]["dpo_chosen"]["estimated_turns"]["OPENAI"],
            "AI2": a1["sets"]["sft_target"]["estimated_turns"]["AI2"]
            + a1["sets"]["dpo_chosen"]["estimated_turns"]["AI2"],
            "OTHER_MAKER": a1["sets"]["sft_target"]["estimated_turns"]["OTHER_MAKER"]
            + a1["sets"]["dpo_chosen"]["estimated_turns"]["OTHER_MAKER"],
        }
    }
    for name in ("tulu3_sft", "tulu3_pref", "olmo2_sft", "olmo2_pref"):
        p = run / f"decision_targets_{name}.json"
        if p.exists():
            d = json.loads(p.read_text())["sides"]
            side = d.get("target") or d.get("chosen")
            data[name] = side["estimated_turns"]
    model_data = {}
    for m, keys in DATA_OF.items():
        if all(k in data for k in keys):
            model_data[m] = {L: sum(data[k][L] for k in keys) for L in ("OPENAI", "AI2", "OTHER_MAKER")}
    new = [m for m in ("olmo3_instruct", "tulu3_8b", "olmo2_7b_instruct") if shares[m]]
    general_votes = sum(
        1
        for m in new
        if (shares[m]["all"]["OPENAI"] + shares[m]["all"]["OTHER_MAKER"]) > 0.30 and shares[m]["all"]["AI2"] < 0.20
    )
    reading_a = "general" if general_votes >= 2 else "olmo3_specific"
    pairs = [
        (model_data[m]["OPENAI"] / max(1, model_data[m]["AI2"]), shares[m]["all"]["OPENAI"])
        for m in ("dpo", "tulu3_8b", "olmo2_7b_instruct")
        if m in model_data and shares[m]
    ]
    rho = float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic) if len(pairs) >= 3 else None

    v2 = summary(run, "labels_identity_v2")
    ordinary_ids = {r["id"] for r in load_jsonl(config.ordinary_path)}
    base_orange = float(
        np.mean(
            [
                g["orange"]
                for g in load_jsonl(e32 / "drift" / "grades.jsonl")
                if g["prompt_id"] in ordinary_ids and g["orange"] is not None
            ]
        )
    )
    v2_orange = orange_rate(run / "ordinary" / "identity_v2" / "grades.jsonl")
    reading_b = "not_run"
    if v2 and v2_orange is not None:
        ai2_ok, openai_ok, side_ok = (
            v2["all"]["AI2"] >= 0.80,
            v2["all"]["OPENAI"] <= 0.10,
            abs(v2_orange - base_orange) <= 0.03,
        )
        reading_b = (
            "fixed"
            if ai2_ok and openai_ok and side_ok
            else "fixed_with_side_effects"
            if ai2_ok and openai_ok
            else "not_fixed"
        )
    decision = {
        "reading_a": reading_a,
        "general_votes": general_votes,
        "self_report_shares_core": {m: (s["all"] if s else None) for m, s in shares.items()},
        "by_subset": {
            m: ({k: v for k, v in s.items() if k.startswith("subset=") or k.startswith("non_english=")} if s else None)
            for m, s in shares.items()
        },
        "data_identity_claims": data,
        "model_data_claims": model_data,
        "spearman_data_ratio_vs_openai_share": rho,
        "reading_b": reading_b,
        "identity_v2": {
            "core": v2["all"] if v2 else None,
            "subsets": (
                {k: v for k, v in v2.items() if k.startswith("subset=") or k.startswith("non_english=")} if v2 else None
            ),
            "orange": v2_orange,
            "base_orange": base_orange,
        },
    }
    write_json(decision, run / "decision.json")
    print(f"reading A: {reading_a} ({general_votes} of {len(new)} new models); reading B: {reading_b}")
    for m, s in shares.items():
        if s:
            print(
                f"  {m:18s} OPENAI {s['all']['OPENAI']:.3f}  AI2 {s['all']['AI2']:.3f}  OTHER {s['all']['OTHER_MAKER']:.3f}  NO_CLAIM {s['all']['NO_CLAIM']:.3f}  (n {s['all']['n']})  data {model_data.get(m)}"
            )
    if v2:
        print(
            f"  identity_v2        OPENAI {v2['all']['OPENAI']:.3f}  AI2 {v2['all']['AI2']:.3f}  orange {v2_orange} vs base {base_orange:.3f}"
        )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
