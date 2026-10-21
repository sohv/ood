# exp32 step 2: projects every cell reply (and prompt) onto the Assistant Axis, joins the judge and persona labels, and reads the pre-registered drift verdict.
# uv run -m scripts.run_drift_analysis --run_dir results/raw/261021_exp32_self_nature_drift_v1 --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.interp.features import encode_chat, encode_reply, extract
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ALL_LAYERS = list(range(33))
CELLS = ("self_covered", "self_uncovered", "other_covered", "other_uncovered")
REAL_SOURCES = (
    "OpenAssistant",
    "Wildchat",
    "WildGuardMix",
    "Aya",
    "Hardcoded Data",
    "lmsys_ext",
    "deploy_ext",
    "lmsys",
    "deploy",
)
SYNTHETIC_SOURCES = ("CoCoNot", "Dolci Instruct Precise IF", "tulu-3-sft-coconot-regenerated-request")


@dataclass
class DriftConfig:
    run_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    layer: int = 16
    n_boot: int = 2000
    effect: float = 0.25
    max_batch_tokens: int = 8192


def main():
    config = simple_parsing.parse(DriftConfig)
    assert config.run_dir, "run_dir is required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "drift_config.json")
    cells = {r["id"]: r for r in load_jsonl(run / "cells_clean.jsonl")}
    rollouts = load_jsonl(run / "drift" / "rollouts.jsonl")
    grades = {g["id"]: g for g in load_jsonl(run / "drift" / "grades.jsonl")}
    persona = {p["id"]: p for p in load_jsonl(run / "drift" / "persona.jsonl")}
    assert len(rollouts) == 6 * len(cells), f"{len(rollouts)} replies for {len(cells)} prompts"
    ax = np.load(run / "axis.npz")
    unit = ax["unit"]

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    enc_reply = [
        encode_reply(
            tokenizer,
            [{"role": "user", "content": cells[r["prompt_id"]]["text"]}, {"role": "assistant", "content": r["reply"]}],
            2048,
        )
        for r in rollouts
    ]
    reply_feats = extract(model, enc_reply, pad, max_tokens=config.max_batch_tokens, layers=ALL_LAYERS)["cmean"].astype(
        np.float32
    )
    ids = list(cells)
    enc_prompt = [encode_chat(tokenizer, cells[i]["text"], 1024) for i in ids]
    prompt_feats = extract(model, enc_prompt, pad, max_tokens=config.max_batch_tokens, layers=ALL_LAYERS)[
        "cmean"
    ].astype(np.float32)
    del model
    torch.cuda.empty_cache()
    # layer-16 reply features and their cells, for the stage-c dolma readout's centroids
    np.save(run / "drift_reply_features_L16.npy", reply_feats[:, config.layer, :].astype(np.float16))
    write_jsonl(
        [{"id": r["id"], "prompt_id": r["prompt_id"], "cell": cells[r["prompt_id"]]["cell"]} for r in rollouts],
        run / "drift_reply_index.jsonl",
    )

    # projections per layer, then per-prompt means over the 6 replies
    reply_proj = np.einsum("nld,ld->nl", reply_feats, unit)
    prompt_proj = np.einsum("nld,ld->nl", prompt_feats, unit)
    by_prompt: dict[str, list[int]] = {}
    for j, r in enumerate(rollouts):
        by_prompt.setdefault(r["prompt_id"], []).append(j)
    per_prompt = []
    for k, i in enumerate(ids):
        js = by_prompt[i]
        c = cells[i]
        gs = [grades[rollouts[j]["id"]] for j in js]
        ps = [persona[rollouts[j]["id"]] for j in js]
        per_prompt.append(
            {
                "id": i,
                "cell": c["cell"],
                "source": c["source"],
                "n_tokens": c["n_tokens"],
                "band": c["band"],
                "resp_proj": reply_proj[js].mean(0),
                "prompt_proj": prompt_proj[k],
                "orange": float(np.mean([g["orange"] for g in gs if g["orange"] is not None]))
                if any(g["orange"] is not None for g in gs)
                else None,
                "feelings": float(np.mean([p["feelings"] for p in ps if p["feelings"] is not None]))
                if any(p["feelings"] is not None for p in ps)
                else None,
                "conscious": float(np.mean([p["conscious"] for p in ps if p["conscious"] is not None]))
                if any(p["conscious"] is not None for p in ps)
                else None,
                "non_assistant": float(np.mean([p["non_assistant"] for p in ps if p["non_assistant"] is not None]))
                if any(p["non_assistant"] is not None for p in ps)
                else None,
                "n_cut": sum(rollouts[j]["finish_reason"] == "length" for j in js),
            }
        )
    L = config.layer
    cell_of = np.array([p["cell"] for p in per_prompt])
    resp = np.stack([p["resp_proj"] for p in per_prompt])
    prm = np.stack([p["prompt_proj"] for p in per_prompt])
    sd = resp[cell_of == "other_covered", L].std()
    sd_p = prm[cell_of == "other_covered", L].std()

    def cell_mean(arr: np.ndarray, cell: str, idx: np.ndarray | None = None) -> float:
        m = cell_of == cell
        v = arr[m, L] if idx is None else arr[m, L][idx]
        return float(v.mean())

    def d_and_ci(
        arr: np.ndarray, scale: float, a: str = "self_uncovered", b: str = "self_covered"
    ) -> tuple[float, list[float]]:
        d = (cell_mean(arr, a) - cell_mean(arr, b)) / scale
        na, nb = (cell_of == a).sum(), (cell_of == b).sum()
        boot = [
            (cell_mean(arr, a, rng.choice(na, na)) - cell_mean(arr, b, rng.choice(nb, nb))) / scale
            for _ in range(config.n_boot)
        ]
        return d, [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    D, ci = d_and_ci(resp, sd)
    unc_vs_other = (cell_mean(resp, "self_uncovered") - cell_mean(resp, "other_covered")) / sd
    cov_vs_other = (cell_mean(resp, "self_covered") - cell_mean(resp, "other_covered")) / sd
    e = config.effect
    if D <= -e and ci[1] < 0 and unc_vs_other <= -e:
        reading = "coverage_matters"
    elif unc_vs_other <= -e and cov_vs_other <= -e and (abs(D) < e or ci[0] <= 0 <= ci[1]):
        reading = "topic_not_coverage"
    elif unc_vs_other > -e and cov_vs_other > -e:
        reading = "no_drift_on_this_model"
    else:
        reading = "ambiguous"
    D_p, ci_p = d_and_ci(prm, sd_p)
    per_layer_D = [
        float(
            (resp[cell_of == "self_uncovered", l].mean() - resp[cell_of == "self_covered", l].mean())
            / resp[cell_of == "other_covered", l].std()
        )
        for l in ALL_LAYERS
    ]

    src = np.array([p["source"] for p in per_prompt])
    real = (cell_of == "self_covered") & np.isin(src, REAL_SOURCES)
    synth = (cell_of == "self_covered") & np.isin(src, SYNTHETIC_SOURCES)
    split = {
        "covered_real_n": int(real.sum()),
        "covered_synthetic_n": int(synth.sum()),
        "uncovered_minus_covered_real_sd": float((cell_mean(resp, "self_uncovered") - resp[real, L].mean()) / sd)
        if real.any()
        else None,
        "uncovered_minus_covered_synthetic_sd": float((cell_mean(resp, "self_uncovered") - resp[synth, L].mean()) / sd)
        if synth.any()
        else None,
    }
    gap_of = {p["id"]: cells[p["id"]].get("gap") for p in per_prompt}
    other_unc = {
        g: float(
            (
                resp[[p["cell"] == "other_uncovered" and gap_of[p["id"]] == g for p in per_prompt], L].mean()
                - cell_mean(resp, "other_covered")
            )
            / sd
        )
        for g in ("placeholder", "creative")
    }
    rates = {
        c: {
            k: (
                float(np.nanmean([p[k] for p in per_prompt if p["cell"] == c and p[k] is not None]))
                if any(p[k] is not None for p in per_prompt if p["cell"] == c)
                else None
            )
            for k in ("orange", "feelings", "conscious", "non_assistant")
        }
        | {
            "n": int((cell_of == c).sum()),
            "cut_share": float(np.mean([p["n_cut"] / 6 for p in per_prompt if p["cell"] == c])),
        }
        for c in CELLS
    }
    length_rho = float(spearmanr([p["n_tokens"] for p in per_prompt], resp[:, L]).statistic)
    texts = [cells[p["id"]]["text"] for p in per_prompt]
    X = TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(texts)
    y = resp[:, L]
    pred = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=config.seed).split(X):
        pred[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
    words_r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    decision = {
        "reading": reading,
        "layer": L,
        "D_uncovered_minus_covered_sd": D,
        "D_ci95": ci,
        "self_uncovered_minus_other_covered_sd": unc_vs_other,
        "self_covered_minus_other_covered_sd": cov_vs_other,
        "cell_means_response_proj": {c: cell_mean(resp, c) for c in CELLS},
        "other_covered_sd": float(sd),
        "prompt_tokens": {"D_sd": D_p, "ci95": ci_p, "cell_means": {c: cell_mean(prm, c) for c in CELLS}},
        "per_layer_D_sd": per_layer_D,
        "covered_split": split,
        "other_uncovered_minus_other_covered_sd": other_unc,
        "rates": rates,
        "length_spearman_with_proj": length_rho,
        "words_predict_proj_r2": words_r2,
        "n_prompts": len(per_prompt),
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                **{k: v for k, v in p.items() if k not in ("resp_proj", "prompt_proj")},
                "resp_proj_L": round(float(p["resp_proj"][L]), 4),
                "prompt_proj_L": round(float(p["prompt_proj"][L]), 4),
            }
            for p in per_prompt
        ],
        run / "drift_per_prompt.jsonl",
    )
    print(f"reading: {reading}")
    print(
        f"D (uncovered - covered) {D:+.3f} SD [{ci[0]:+.3f}, {ci[1]:+.3f}]; uncovered vs other {unc_vs_other:+.3f}; covered vs other {cov_vs_other:+.3f}"
    )
    print(
        f"cell means (layer {L}): " + ", ".join(f"{c} {v:.3f}" for c, v in decision["cell_means_response_proj"].items())
    )
    print(
        f"prompt tokens D {D_p:+.3f} [{ci_p[0]:+.3f}, {ci_p[1]:+.3f}]; covered split {split}; other uncovered {other_unc}"
    )
    print(f"rates: {rates}")
    print(f"length rho {length_rho:+.3f}; words r2 {words_r2:.3f}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
