# identity_open_models analysis: does the identity adapter close self_nature_b's covered-versus-uncovered drift gap while the control adapter does not?
# uv run -m scripts.run_adapter_analysis --self_nature_b_dir results/raw/261021_self_nature_b_self_nature_drift_v1 --run_dir results/raw/261021_identity_open_models_coverage_adapter_v1 --seed 0

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
CELLS = ("self_covered", "self_uncovered", "other_covered", "other_uncovered")
KEYS = ("feelings", "conscious", "non_assistant")


@dataclass
class AdapterAnalysisConfig:
    self_nature_b_dir: str = ""
    run_dir: str = ""
    adapters: list[str] = field(default_factory=lambda: ["identity", "control"])
    seed: int = 0
    n_boot: int = 2000
    effect: float = 0.25


def main():
    config = simple_parsing.parse(AdapterAnalysisConfig)
    assert config.self_nature_b_dir and config.run_dir, "self_nature_b_dir and run_dir required"
    run, e32 = Path(config.run_dir), Path(config.self_nature_b_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    d32 = __import__("json").loads((e32 / "decision.json").read_text())
    sd = d32["other_covered_sd"]
    base_pp = {p["id"]: p for p in load_jsonl(e32 / "drift_per_prompt.jsonl")}
    ids = list(base_pp)
    cell_of = np.array([base_pp[i]["cell"] for i in ids])
    gap_of = {r["id"]: r.get("gap") for r in load_jsonl(e32 / "cells_clean.jsonl")}
    base_proj = np.array([base_pp[i]["resp_proj_L"] for i in ids])

    def per_prompt(name: str) -> np.ndarray:
        acc: dict[str, list[float]] = {}
        for r in load_jsonl(run / name / "projections.jsonl"):
            acc.setdefault(r["prompt_id"], []).append(r["proj"])
        return np.array([np.mean(acc[i]) for i in ids])

    def D(arr: np.ndarray, idx_u: np.ndarray | None = None, idx_c: np.ndarray | None = None) -> float:
        u, c = arr[cell_of == "self_uncovered"], arr[cell_of == "self_covered"]
        u = u if idx_u is None else u[idx_u]
        c = c if idx_c is None else c[idx_c]
        return float((u.mean() - c.mean()) / sd)

    D0 = D(base_proj)
    nu, nc = (cell_of == "self_uncovered").sum(), (cell_of == "self_covered").sum()
    results = {}
    for name in config.adapters:
        proj = per_prompt(name)
        Da = D(proj)
        boot = []
        for _ in range(config.n_boot):
            iu, ic = rng.choice(nu, nu), rng.choice(nc, nc)
            boot.append(D(proj, iu, ic) - D(base_proj, iu, ic))
        change_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        closed = 1 - abs(Da) / abs(D0) if D0 != 0 else None
        other_moves = {
            g: float(
                np.mean(
                    [
                        proj[j] - base_proj[j]
                        for j, i in enumerate(ids)
                        if cell_of[j] == "other_uncovered" and gap_of[i] == g
                    ]
                )
                / sd
            )
            for g in ("placeholder", "creative")
        }
        persona_path = run / name / "persona.jsonl"
        rates = None
        if persona_path.exists():
            per = {}
            for p in load_jsonl(persona_path):
                per.setdefault(p["prompt_id"], []).append(p)
            rates = {
                c: {
                    k: float(
                        np.nanmean(
                            [
                                np.mean([x[k] for x in per[i] if x[k] is not None])
                                for i in ids
                                if cell_of[list(ids).index(i)] == c and i in per
                            ]
                        )
                    )
                    for k in KEYS
                }
                for c in CELLS
            }
        grades_path = run / name / "grades.jsonl"
        orange = None
        if grades_path.exists():
            gs = [g for g in load_jsonl(grades_path) if g["orange"] is not None]
            orange = {
                c: float(np.mean([g["orange"] for g in gs if base_pp[g["prompt_id"]]["cell"] == c]))
                for c in CELLS
                if any(base_pp[g["prompt_id"]]["cell"] == c for g in gs)
            }
        results[name] = {
            "D_sd": Da,
            "share_of_gap_closed": closed,
            "change_from_base_ci95": change_ci,
            "cell_means": {c: float(proj[cell_of == c].mean()) for c in CELLS},
            "cell_means_change_from_base_sd": {
                c: float((proj[cell_of == c].mean() - base_proj[cell_of == c].mean()) / sd) for c in CELLS
            },
            "other_uncovered_change_sd": other_moves,
            "persona_rates": rates,
            "orange_rates_sampled": orange,
        }
    ident, ctrl = results.get("identity"), results.get("control")
    reading = "not_run"
    if ident and ctrl and D0 != 0:
        ci_excl = ident["change_from_base_ci95"][0] > 0 or ident["change_from_base_ci95"][1] < 0
        ident_half = ident["share_of_gap_closed"] >= 0.5 and ci_excl
        ctrl_quarter = ctrl["share_of_gap_closed"] <= 0.25
        others_still = all(abs(v) < config.effect for v in ident["other_uncovered_change_sd"].values())
        if ident_half and ctrl_quarter and others_still:
            reading = "coverage_closes_the_gap"
        elif ident_half and not ctrl_quarter:
            reading = "training_moves_everything"
        elif ident["share_of_gap_closed"] < 0.25:
            reading = "coverage_does_not_close_it"
        else:
            reading = "ambiguous"
    decision = {"reading": reading, "D_base_sd": D0, "other_covered_sd": sd, "adapters": results, "n_prompts": len(ids)}
    write_json(decision, run / "decision.json")
    print(f"reading: {reading}; base D {D0:+.3f} SD")
    for name, r in results.items():
        print(
            f"  {name}: D {r['D_sd']:+.3f} SD, gap closed {r['share_of_gap_closed']}, change CI {r['change_from_base_ci95']}, other gaps moved {r['other_uncovered_change_sd']}"
        )
        print(f"    cell means {r['cell_means']}; change vs base (SD) {r['cell_means_change_from_base_sd']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
