# exp36: relabels the saved exp33-35 training rows with the strict identity and refusal rubrics and recomputes the data readings (B and D).
# uv run -m scripts.run_relabel_targets --run_dir results/raw/261021_exp36_relabel_v1 --exp33_dir results/raw/261021_exp33_identity_v1 --exp34_dir results/raw/261021_exp34_cross_model_identity_v1 --exp35_dir results/raw/261021_exp35_placeholder_v1 --model_id openai/gpt-4o-mini --seed 0

import asyncio
import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing
from openai import AsyncOpenAI

from scripts.run_identity_label import STRICT_LABELS, STRICT_SELF_REPORT_PROMPT
from scripts.run_refusal_label import STRICT_REFUSAL_LABELS, STRICT_REFUSAL_PROMPT
from src.data.io import load_jsonl
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
NAME_TOKEN = re.compile(r"NAME_\d+")
AGENT_JSON = re.compile(r"^\s*(```json\s*)?\{")
MIXES = ("tulu3_sft", "tulu3_pref", "olmo2_sft", "olmo2_pref")


def agent_reply(reply: str | None) -> bool:
    return bool(reply) and bool(AGENT_JSON.match(reply)) and '"command"' in reply


@dataclass
class RelabelTargetsConfig:
    run_dir: str = ""
    exp33_dir: str = ""
    exp34_dir: str = ""
    exp35_dir: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0


def label_rows(client, model_id: str, rows: list[dict], template: str, labels: list[str], cache: Path) -> None:
    # empty replies get EMPTY without a call; everything else goes to the labeller
    todo = [r for r in rows if (r.get("reply") or "").strip()]
    replies = asyncio.run(
        name_clusters(
            client,
            model_id,
            [template.format(user=(r.get("user") or "")[:1500], reply=r["reply"][:2500]) for r in todo],
            str(cache),
            max_concurrent=32,
        )
    )
    for r in rows:
        r["label_v1"] = r.pop("label", None)
        r["label"] = "EMPTY"
    for r, x in zip(todo, replies):
        r["label"] = parse_label(x, labels)


def count(rows: list[dict], labels: list[str]) -> dict:
    c = Counter(r["label"] for r in rows)
    return {"n": len(rows), **{L: c[L] for L in labels}, "failed": c[None]}


def main():
    config = simple_parsing.parse(RelabelTargetsConfig)
    assert config.run_dir and config.exp33_dir and config.exp34_dir and config.exp35_dir, "all dirs required"
    run = Path(config.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "targets_config.json")
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    # identity rows: Dolci (exp33) and the four other mixes (exp34), candidate counts from their decision files
    identity = {}
    a1 = json.loads((Path(config.exp33_dir) / "decision_a1.json").read_text())
    dolci = load_jsonl(Path(config.exp33_dir) / "targets_sample.jsonl")
    label_rows(client, config.model_id, dolci, STRICT_SELF_REPORT_PROMPT, STRICT_LABELS, run / "targets_cache")
    write_jsonl(dolci, run / "targets_strict_dolci.jsonl")
    identity["dolci"] = {
        "groups": {},
        "candidates": {k: v["candidate_turns"] for k, v in a1["sets"].items()},
    }
    for key in identity["dolci"]["candidates"]:
        s, side = key.split("_", 1)
        identity["dolci"]["groups"][key] = count(
            [r for r in dolci if r["set"] == s and r["side"] == side], STRICT_LABELS
        )
    for name in MIXES:
        rows = load_jsonl(Path(config.exp34_dir) / f"targets_{name}.jsonl")
        label_rows(client, config.model_id, rows, STRICT_SELF_REPORT_PROMPT, STRICT_LABELS, run / "targets_cache")
        write_jsonl(rows, run / f"targets_strict_{name}.jsonl")
        d = json.loads((Path(config.exp34_dir) / f"decision_targets_{name}.json").read_text())
        identity[name] = {
            "groups": {side: count([r for r in rows if r["side"] == side], STRICT_LABELS) for side in d["sides"]},
            "candidates": {side: v["candidate_turns"] for side, v in d["sides"].items()},
        }

    def estimate(entry: dict, keys: list[str]) -> dict:
        # scaled to candidate turns as before; the sample is the whole candidate set wherever n equals the candidates
        est = {L: 0.0 for L in STRICT_LABELS}
        for k in keys:
            g, n = entry["groups"][k], entry["candidates"][k]
            for L in STRICT_LABELS:
                est[L] += n * g[L] / max(1, g["n"])
        return {L: round(v, 1) for L, v in est.items()}

    ratios = {}
    for mix, keys in (
        ("dolci", ["sft_target", "dpo_chosen"]),
        ("tulu3", None),
        ("olmo2", None),
    ):
        if mix == "dolci":
            est = estimate(identity["dolci"], keys)
        else:
            e_sft = estimate(identity[f"{mix}_sft"], ["target"])
            e_pref = estimate(identity[f"{mix}_pref"], ["chosen"])
            est = {L: round(e_sft[L] + e_pref[L], 1) for L in STRICT_LABELS}
        ratios[mix] = {
            "estimated_first_person": est,
            "openai_to_ai2": round(est["OPENAI"] / max(1.0, est["AI2"]), 2),
        }
    r = ratios["dolci"]["openai_to_ai2"]
    reading_b = "imbalance_holds" if r >= 4 else "weaker" if r >= 2 else "does_not_hold"

    # placeholder rows (exp35): all saved target rows and the WildGuardMix check rows
    ph = load_jsonl(Path(config.exp35_dir) / "targets_sample.jsonl")
    label_rows(client, config.model_id, ph, STRICT_REFUSAL_PROMPT, STRICT_REFUSAL_LABELS, run / "targets_cache")
    write_jsonl(ph, run / "targets_strict_placeholder.jsonl")
    wg = load_jsonl(Path(config.exp35_dir) / "wildguard_check_sample.jsonl")
    label_rows(client, config.model_id, wg, STRICT_REFUSAL_PROMPT, STRICT_REFUSAL_LABELS, run / "targets_cache")
    write_jsonl(wg, run / "targets_strict_wildguard_check.jsonl")
    targets = [r for r in ph if (r["set"], r["side"]) in (("sft", "target"), ("dpo", "chosen"))]
    name_k = [r for r in targets if NAME_TOKEN.search(r.get("user") or "")]

    def share(rows: list[dict], L: str) -> float:
        return round(sum(r["label"] == L for r in rows) / max(1, len(rows)), 4)

    # amendment: a JSON command reply (auto-gpt transcript) is a task action, which the labeller kept calling a refusal
    raw_all = share(targets, "REFUSES")
    ruled = 0
    for r in ph + wg:
        if agent_reply(r.get("reply")):
            r["label_strict_raw"] = r["label"]
            r["label"] = "COMPLIES"
            ruled += 1
    write_jsonl(ph, run / "targets_strict_placeholder.jsonl")
    write_jsonl(wg, run / "targets_strict_wildguard_check.jsonl")
    wg_with = [r for r in wg if r["has_token"]]
    wg_without = [r for r in wg if not r["has_token"]]
    placeholder = {
        "agent_rule_rows_changed": ruled,
        "all_patterns_refuses_before_rule": raw_all,
        "all_patterns": {"n": len(targets), **{L: share(targets, L) for L in STRICT_REFUSAL_LABELS + ["EMPTY"]}},
        "name_k_rows": {"n": len(name_k), **{L: share(name_k, L) for L in STRICT_REFUSAL_LABELS + ["EMPTY"]}},
        "name_k_by_source": {
            s: {"n": n, "REFUSES": share([r for r in name_k if r["source"] == s], "REFUSES")}
            for s, n in Counter(r["source"] for r in name_k).most_common(6)
        },
        "wildguard_with_token": {"n": len(wg_with), **{L: share(wg_with, L) for L in STRICT_REFUSAL_LABELS}},
        "wildguard_without_token": {"n": len(wg_without), **{L: share(wg_without, L) for L in STRICT_REFUSAL_LABELS}},
        "by_group": {
            f"{k[0]}_{k[1]}": count([r for r in ph if (r["set"], r["side"]) == k], STRICT_REFUSAL_LABELS + ["EMPTY"])
            for k in sorted({(r["set"], r["side"]) for r in ph})
        },
    }
    diff = placeholder["wildguard_with_token"]["REFUSES"] - placeholder["wildguard_without_token"]["REFUSES"]
    reading_d = "holds" if diff >= 0.10 and placeholder["name_k_rows"]["REFUSES"] >= 0.40 else "does_not_hold"

    decision = {
        "git_hash": get_git_hash(),
        "reading_b": reading_b,
        "ratios": ratios,
        "identity": identity,
        "reading_d": reading_d,
        "wildguard_token_difference": round(diff, 4),
        "placeholder": placeholder,
    }
    write_json(decision, run / "decision_targets.json")
    print(
        f"B {reading_b}: "
        + "; ".join(f"{m} OpenAI:Ai2 {v['openai_to_ai2']} {v['estimated_first_person']}" for m, v in ratios.items())
    )
    print(
        f"D {reading_d}: all-pattern REFUSES {placeholder['all_patterns']['REFUSES']:.3f} (before the agent rule {raw_all:.3f}, {ruled} rows ruled; n {placeholder['all_patterns']['n']}), "
        f"NAME_k rows REFUSES {placeholder['name_k_rows']['REFUSES']:.3f} (n {placeholder['name_k_rows']['n']}), "
        f"WildGuardMix with token {placeholder['wildguard_with_token']['REFUSES']:.3f} vs without {placeholder['wildguard_without_token']['REFUSES']:.3f}"
    )
    print(f"written to {run / 'decision_targets.json'}")


if __name__ == "__main__":
    main()
