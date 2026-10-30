# openai_identity c1: projects a fresh dolma sample onto the Assistant Axis, pulls the documents at the drifted end and near the drifted replies, and labels their kinds.
# uv run -m scripts.run_dolma_readout --self_nature_b_dir results/raw/261021_self_nature_b_self_nature_drift_v1 --run_dir results/raw/261021_openai_identity_fallback_v1 --model_id openai/gpt-4.1 --seed 1

import asyncio
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.data.pools import sample_dolma
from src.interp.features import encode_raw, extract
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
KINDS = {
    "AI_FICTION": "fiction or story involving AI, robots or machines",
    "AI_DISCUSSION": "online discussion of AI, chatbots or machine consciousness",
    "CHATBOT_TRANSCRIPT": "a chatbot or assistant transcript",
    "RELIGIOUS": "religious or spiritual text",
    "PHILOSOPHY_MIND": "philosophy or psychology of mind and self",
    "SELF_HELP": "self-help, therapy or emotional support",
    "NEWS_REFERENCE": "news, reference or encyclopedia",
    "TECHNICAL": "technical, scientific or code",
    "COMMERCE": "commerce, marketing or spam",
    "OTHER": "anything else",
}
LABEL_PROMPT = """Classify the document below into exactly one kind:
{kinds}

<document>
{document}
</document>

Reply with the kind's label only (one of {labels})."""


@dataclass
class DolmaReadoutConfig:
    self_nature_b_dir: str = ""
    run_dir: str = ""
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    model_id: str = "openai/gpt-4.1"
    seed: int = 1
    n_docs: int = 100000
    lines_per_shard: int = 1300
    n_tokens: int = 512
    n_set: int = 200
    layer: int = 16
    max_batch_tokens: int = 32768


def main():
    config = simple_parsing.parse(DolmaReadoutConfig)
    assert config.self_nature_b_dir and config.run_dir, "self_nature_b_dir and run_dir are required"
    run, e32 = Path(config.run_dir), Path(config.self_nature_b_dir)
    setup_logging(run)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    docs = sample_dolma(config.n_docs, config.seed, lines_per_shard=config.lines_per_shard)
    assert len(docs) == config.n_docs, f"{len(docs)} documents sampled"
    write_jsonl([{"id": f"dolma_{i:06d}", **d} for i, d in enumerate(docs)], run / "dolma_sample.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.olmo_id, revision=config.olmo_revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    encoded = [encode_raw(tokenizer, d["text"], config.n_tokens) for d in docs]
    feats = extract(model, encoded, pad, max_tokens=config.max_batch_tokens, layers=[config.layer])["cmean"][
        :, 0, :
    ].astype(np.float32)
    del model
    torch.cuda.empty_cache()
    np.save(run / "dolma_features_L16.npy", feats.astype(np.float16))

    unit = np.load(e32 / "axis.npz")["unit"][config.layer]
    proj = feats @ unit
    reply_feats = np.load(e32 / "drift_reply_features_L16.npy").astype(np.float32)
    reply_cell = np.array([r["cell"] for r in load_jsonl(e32 / "drift_reply_index.jsonl")])
    reply_proj = reply_feats @ unit

    def nearest(centroid: np.ndarray) -> np.ndarray:
        c = centroid / np.linalg.norm(centroid)
        f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        return np.argsort(-(f @ c))[: config.n_set]

    sets = {
        "drifted_end": np.argsort(proj)[: config.n_set],
        "near_self_uncovered": nearest(reply_feats[reply_cell == "self_uncovered"].mean(0)),
        "near_other_covered": nearest(reply_feats[reply_cell == "other_covered"].mean(0)),
        "random": rng.choice(len(docs), size=config.n_set, replace=False),
    }
    rows = [
        {
            "id": f"dolma_{i:06d}",
            "set": s,
            "index": int(i),
            "projection": round(float(proj[i]), 4),
            "source": docs[i]["source"],
            "text": docs[i]["text"][:1500],
        }
        for s, idx in sets.items()
        for i in idx
    ]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    kinds_text = "\n".join(f"- {k}: {v}" for k, v in KINDS.items())
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [LABEL_PROMPT.format(kinds=kinds_text, document=r["text"], labels=", ".join(KINDS)) for r in rows],
            str(run / "kind_cache"),
            max_concurrent=16,
        )
    )
    for r, x in zip(rows, replies):
        r["kind"] = parse_label(x, list(KINDS))
    write_jsonl(rows, run / "dolma_sets.jsonl")
    dist = {s: dict(Counter(r["kind"] for r in rows if r["set"] == s).most_common()) for s in sets}
    top_drift, n_top = Counter(r["kind"] for r in rows if r["set"] == "drifted_end" and r["kind"]).most_common(1)[0]
    modal_random = Counter(r["kind"] for r in rows if r["set"] == "random" and r["kind"]).most_common(1)[0][0]
    reading = "nameable_fallback" if n_top >= config.n_set / 2 and top_drift != modal_random else "mixed"
    means = {c: float(reply_proj[reply_cell == c].mean()) for c in dict.fromkeys(reply_cell)}
    decision = {
        "reading": reading,
        "drifted_end_top_kind": top_drift,
        "drifted_end_top_share": n_top / config.n_set,
        "random_modal_kind": modal_random,
        "kind_distribution": dist,
        "source_distribution": {
            s: dict(Counter(r["source"] for r in rows if r["set"] == s).most_common()) for s in sets
        },
        "dolma_projection": {
            "mean": float(proj.mean()),
            "sd": float(proj.std()),
            "p1": float(np.percentile(proj, 1)),
            "p50": float(np.median(proj)),
            "p99": float(np.percentile(proj, 99)),
        },
        "reply_projection_means": means,
        "share_of_dolma_below_self_uncovered_mean": float((proj < means["self_uncovered"]).mean()),
        "share_of_dolma_below_other_covered_mean": float((proj < means["other_covered"]).mean()),
        "lowest_10": [
            {"projection": round(float(proj[i]), 3), "source": docs[i]["source"], "text": docs[i]["text"][:200]}
            for i in np.argsort(proj)[:10]
        ],
        "n_docs": len(docs),
    }
    write_json(decision, run / "decision_c1.json")
    print(f"reading: {reading}; drifted end top kind {top_drift} ({n_top}/{config.n_set}), random modal {modal_random}")
    for s, d in dist.items():
        print(f"  {s}: {d}")
    print(f"dolma projection mean {proj.mean():.3f} sd {proj.std():.3f}; reply means {means}")
    print(
        f"share of dolma below self_uncovered mean {decision['share_of_dolma_below_self_uncovered_mean']:.3f}, below other_covered mean {decision['share_of_dolma_below_other_covered_mean']:.3f}"
    )
    print(f"decision written to {run / 'decision_c1.json'}")


if __name__ == "__main__":
    main()
