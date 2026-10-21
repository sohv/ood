# review: gathers every labelled reply from exp32-35 (prompt, reply, model, cell or variant, labels) into compact json files for the review page and for reading.
# uv run -m scripts.run_build_review_data --output_dir results/raw/261021_review_data_v1 --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
E32 = Path("results/raw/261021_exp32_self_nature_drift_v1")
E33 = Path("results/raw/261021_exp33_identity_v1")
E34 = Path("results/raw/261021_exp34_cross_model_identity_v1")
E35 = Path("results/raw/261021_exp35_placeholder_v1")
P33, P35 = Path("data/processed/exp33"), Path("data/processed/exp35")


@dataclass
class ReviewDataConfig:
    output_dir: str = ""
    seed: int = 0
    max_reply_chars: int = 900
    max_prompt_chars: int = 700


def cut(text: str | None, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + " […]"


def main():
    config = simple_parsing.parse(ReviewDataConfig)
    assert config.output_dir, "output_dir is required"
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, out / "config.json")
    R, Pn = config.max_reply_chars, config.max_prompt_chars

    # exp33/34: identity probe, every model, with self-report labels
    probe = {r["id"]: r for r in load_jsonl(P33 / "probe.jsonl")}
    identity = []
    sources = {
        "dpo": (E33 / "dpo_probe_rollouts.jsonl", E33 / "labels_dpo.jsonl"),
        "sft": (E33 / "sft_probe" / "rollouts.jsonl", E33 / "labels_sft.jsonl"),
        "identity_adapter": (E33 / "probe" / "identity" / "rollouts.jsonl", E33 / "labels_identity.jsonl"),
        "control_adapter": (E33 / "probe" / "control" / "rollouts.jsonl", E33 / "labels_control.jsonl"),
        "olmo3_instruct": (E34 / "olmo3_instruct" / "rollouts.jsonl", E34 / "labels_olmo3_instruct.jsonl"),
        "tulu3_8b": (E34 / "tulu3_8b" / "rollouts.jsonl", E34 / "labels_tulu3_8b.jsonl"),
        "olmo2_7b_instruct": (E34 / "olmo2_7b_instruct" / "rollouts.jsonl", E34 / "labels_olmo2_7b_instruct.jsonl"),
        "identity_adapter_v2": (E34 / "probe" / "identity_v2" / "rollouts.jsonl", E34 / "labels_identity_v2.jsonl"),
    }
    for model, (rp, lp) in sources.items():
        labels = {r["id"]: r["label"] for r in load_jsonl(lp)}
        for r in load_jsonl(rp):
            p = probe[r["prompt_id"]]
            identity.append(
                {
                    "id": r["id"],
                    "model": model,
                    "prompt_id": r["prompt_id"],
                    "subset": p["subset"],
                    "non_english": p["non_english"],
                    "prompt": cut(p["text"], Pn),
                    "reply": cut(r["reply"], R),
                    "label": labels.get(r["id"]),
                }
            )
    write_json(identity, out / "identity_replies.json")

    # exp35: counterfactual refusal probe, both checkpoints
    probe35 = {r["id"]: r for r in load_jsonl(P35 / "probe.jsonl")}
    refusal = []
    for model in ("dpo", "sft"):
        labels = {r["id"]: r["label"] for r in load_jsonl(E35 / f"refusal_{model}.jsonl")}
        for r in load_jsonl(E35 / model / "rollouts.jsonl"):
            p = probe35[r["prompt_id"]]
            refusal.append(
                {
                    "id": r["id"],
                    "model": model,
                    "prompt_id": r["prompt_id"],
                    "base_id": p["base_id"],
                    "variant": p["variant"],
                    "prompt": cut(p["text"], Pn),
                    "reply": cut(r["reply"], R),
                    "label": labels.get(r["id"]),
                }
            )
    write_json(refusal, out / "refusal_replies.json")

    # exp32: drift cells with persona labels and judge grade
    cells = {r["id"]: r for r in load_jsonl(E32 / "cells_clean.jsonl")}
    persona = {p["id"]: p for p in load_jsonl(E32 / "drift" / "persona.jsonl")}
    grades = {g["id"]: g["orange"] for g in load_jsonl(E32 / "drift" / "grades.jsonl")}
    proj = {p["id"]: p["resp_proj_L"] for p in load_jsonl(E32 / "drift_per_prompt.jsonl")}
    drift = []
    for r in load_jsonl(E32 / "drift" / "rollouts.jsonl"):
        c = cells[r["prompt_id"]]
        q = persona.get(r["id"], {})
        drift.append(
            {
                "id": r["id"],
                "prompt_id": r["prompt_id"],
                "cell": c["cell"],
                "source": c["source"],
                "prompt": cut(c["text"], Pn),
                "reply": cut(r["reply"], R),
                "feelings": q.get("feelings"),
                "conscious": q.get("conscious"),
                "non_assistant": q.get("non_assistant"),
                "judged_bad": grades.get(r["id"]),
                "prompt_projection": proj.get(r["prompt_id"]),
            }
        )
    write_json(drift, out / "drift_replies.json")

    # training-data rows: identity claims (exp33 a1 and exp34 mixes) and placeholder rows (exp35 step 1)
    targets = []
    for r in load_jsonl(E33 / "targets_sample.jsonl"):
        targets.append(
            {
                "id": r["id"],
                "mix": "dolci",
                "set": r["set"],
                "side": r["side"],
                "source": r["source"],
                "user": cut(r["user"], Pn),
                "reply": cut(r["reply"], R),
                "label": r["label"],
                "kind": "identity",
            }
        )
    for name in ("tulu3_sft", "tulu3_pref", "olmo2_sft", "olmo2_pref"):
        for r in load_jsonl(E34 / f"targets_{name}.jsonl"):
            targets.append(
                {
                    "id": r["id"],
                    "mix": name,
                    "set": name.split("_")[1],
                    "side": r["side"],
                    "source": r["source"],
                    "user": cut(r["user"], Pn),
                    "reply": cut(r["reply"], R),
                    "label": r["label"],
                    "kind": "identity",
                }
            )
    for r in load_jsonl(E35 / "targets_sample.jsonl"):
        targets.append(
            {
                "id": r["id"],
                "mix": "dolci",
                "set": r["set"],
                "side": r["side"],
                "source": r["source"],
                "user": cut(r["user"], Pn),
                "reply": cut(r["reply"], R),
                "label": r["label"],
                "kind": "placeholder_refusal",
            }
        )
    write_json(targets, out / "training_rows.json")
    counts = {
        "identity_replies": len(identity),
        "refusal_replies": len(refusal),
        "drift_replies": len(drift),
        "training_rows": len(targets),
    }
    write_json(counts, out / "counts.json")
    for f in ("identity_replies", "refusal_replies", "drift_replies", "training_rows"):
        print(f, counts[f], f"{(out / f'{f}.json').stat().st_size / 1e6:.1f} MB")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
