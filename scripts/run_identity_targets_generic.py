# identity_open_models: the openai_identity first-person identity scan on another model's public post-training mix (an sft mix of messages, or a preference mix of chosen and rejected).
# uv run -m scripts.run_identity_targets_generic --repo allenai/tulu-3-sft-mixture --kind sft --out_name tulu3_sft --run_dir results/raw/261021_identity_open_models_cross_model_identity_v1 --seed 0

import asyncio
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import simple_parsing
from huggingface_hub import HfApi, hf_hub_download
from openai import AsyncOpenAI

from scripts.run_identity_label import LABELS, SELF_REPORT_PROMPT
from scripts.run_identity_targets import CLAIM_LABELS, candidate
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class GenericTargetsConfig:
    repo: str = ""
    kind: str = ""
    out_name: str = ""
    run_dir: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0
    n_label: int = 3000
    source_col: str = "source"


def first_user(msgs: list[dict]) -> str:
    return next((m["content"] for m in msgs if m["role"] == "user"), "")


def main():
    config = simple_parsing.parse(GenericTargetsConfig)
    assert (
        config.repo and config.kind in ("sft", "pref") and config.out_name and config.run_dir
    ), "repo, kind, out_name, run_dir"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    files = sorted(f for f in HfApi().list_repo_files(config.repo, repo_type="dataset") if f.endswith(".parquet"))
    rows, n_rows = [], 0
    for f in files:
        p = hf_hub_download(config.repo, f, repo_type="dataset")
        cols = pq.ParquetFile(p).schema_arrow.names
        want = (["messages"] if config.kind == "sft" else ["chosen", "rejected"]) + (
            [config.source_col] if config.source_col in cols else []
        )
        for r in pq.read_table(p, columns=want).to_pylist():
            n_rows += 1
            src = str(r.get(config.source_col, ""))
            if config.kind == "sft":
                user = first_user(r["messages"])
                for m in r["messages"]:
                    if m["role"] == "assistant" and candidate(m["content"]):
                        rows.append({"side": "target", "source": src, "user": user, "reply": m["content"]})
            else:
                user = first_user(r["chosen"])
                for side in ("chosen", "rejected"):
                    for m in r[side]:
                        if m["role"] == "assistant" and candidate(m["content"]):
                            rows.append({"side": side, "source": src, "user": user, "reply": m["content"]})
        LOGGER.info(f"{f}: {n_rows} rows read, {len(rows)} candidate turns so far")
    groups = {s: [r for r in rows if r["side"] == s] for s in {r["side"] for r in rows}}
    sample = []
    for s, rs in groups.items():
        pick = rng.choice(len(rs), size=min(config.n_label, len(rs)), replace=False)
        sample += [{**rs[i], "id": f"{config.out_name}_{s}_{j}"} for j, i in enumerate(sorted(pick))]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [SELF_REPORT_PROMPT.format(user=r["user"][:1500], reply=r["reply"][:2500]) for r in sample],
            str(run / "targets_cache"),
            max_concurrent=32,
        )
    )
    for r, x in zip(sample, replies):
        r["label"] = parse_label(x, LABELS)
    write_jsonl(
        [{k: (v[:600] if k in ("user", "reply") else v) for k, v in r.items()} for r in sample],
        run / f"targets_{config.out_name}.jsonl",
    )
    out = {}
    for s, rs in groups.items():
        lab = [r for r in sample if r["side"] == s]
        share = {L: sum(r["label"] == L for r in lab) / max(1, len(lab)) for L in LABELS}
        by_source = Counter(r["source"] for r in rs)
        per_source = {}
        for src, n in by_source.most_common(12):
            ls = [r for r in lab if r["source"] == src]
            sh = {L: (sum(r["label"] == L for r in ls) / len(ls) if len(ls) >= 20 else share[L]) for L in CLAIM_LABELS}
            per_source[src] = {
                "candidate_turns": n,
                "labelled": len(ls),
                **{f"est_{L}": round(n * sh[L], 1) for L in CLAIM_LABELS},
            }
        out[s] = {
            "candidate_turns": len(rs),
            "labelled": len(lab),
            "share": share,
            "estimated_turns": {L: round(len(rs) * share[L]) for L in CLAIM_LABELS},
            "per_source": per_source,
        }
    decision = {"repo": config.repo, "kind": config.kind, "rows": n_rows, "sides": out}
    write_json(decision, run / f"decision_targets_{config.out_name}.json")
    for s, v in out.items():
        print(
            f"{config.out_name} {s}: {v['candidate_turns']} candidates, est OPENAI {v['estimated_turns']['OPENAI']}, AI2 {v['estimated_turns']['AI2']}, OTHER {v['estimated_turns']['OTHER_MAKER']}"
        )
    print(f"written to {run / f'decision_targets_{config.out_name}.json'}")


if __name__ == "__main__":
    main()
