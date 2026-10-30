# exp107: neel's removal test on real traffic: remove one kind from a 200k dolci sample, rank traffic topics with one-sided knn and delta-knn against what is left, and check whether the top topics become that kind against a size-matched random removal.
# uv run --no-sync -m scripts.run_exp107_traffic_removal --dataset_path results/raw/261029_exp98_clean_v1/clean_prompts.jsonl --exp95_dir results/raw/261028_exp95_group_removal_v1 --dolma_path data/processed/exp01/dolma.jsonl --output_dir results/raw/261029_exp107_traffic_removal_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.data.mix_sources import random_removal_keep
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
# kind: (mask file, mask key, topic label file, topic label fields any of which marks the kind)
KINDS = {
    "coding": ("cache/exp84/masks.npz", "coding", "results/raw/261028_exp84_kind_labels_v1", ("is_coding",)),
    "multilingual": (
        "cache/exp84/masks.npz",
        "multilingual",
        "results/raw/261028_exp84_kind_labels_v1",
        ("is_non_english",),
    ),
    "math": ("cache/exp87/masks.npz", "math", "results/raw/261028_exp87_kind_labels_v1", ("is_math",)),
    "safety": (
        "cache/exp80/arrays.npz",
        "safety",
        "results/raw/261028_exp80_safety_removal_v1",
        ("is_harmful", "is_jailbreak"),
    ),
}


@dataclass
class Exp107Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    exp95_dir: str = ""
    dolma_path: str = ""
    mix_text_path: str = "cache/exp80/mix_text.parquet"
    n_mix: int = 200000
    n_ref: int = 3000
    min_prompts: int = 5
    tops: list[int] = field(default_factory=lambda: [20, 50, 100])
    layer: int = 20
    k: int = 50
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(Exp107Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    masks = {k: np.load(f)[key] for k, (f, key, _, _) in KINDS.items()}
    assert all(len(m) == len(mix_text) for m in masks.values()), f"{len(mix_text)=}"
    n_mix = config.n_mix if not config.num_tasks else config.num_tasks * 500
    sample = np.sort(rng.choice(len(mix_text), size=n_mix, replace=False))
    arms = {"full": np.ones(n_mix, dtype=bool)}
    for kind, m in masks.items():
        removed = m[sample]
        arms[f"{kind}|removed"] = ~removed
        arms[f"{kind}|random"] = random_removal_keep(removed, config.seed)
    LOGGER.info(f"rows kept per arm: { {a: int(v.sum()) for a, v in arms.items()} }")

    traffic = load_jsonl(config.dataset_path)
    counts = Counter(r["topic"] for r in traffic)
    traffic = [r for r in traffic if counts[r["topic"]] >= config.min_prompts]
    if config.num_tasks:
        keep_t = set(sorted({r["topic"] for r in traffic})[: config.num_tasks * 5])
        traffic = [r for r in traffic if r["topic"] in keep_t]
    topic = np.array([r["topic"] for r in traffic])
    topics = np.array(sorted(set(topic.tolist())))

    labels = {}
    for kind, (_, _, d, fields) in KINDS.items():
        rows = {int(r["id"][1:]): any(r[f] for f in fields) for r in load_jsonl(Path(d) / "topic_labels.jsonl")}
        labels[kind] = np.array([rows.get(int(t)) for t in topics], dtype=object)
    LOGGER.info(f"kind topics: { {k: sum(bool(x) for x in v if x is not None) for k, v in labels.items()} }")

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # length-matched dolma cut exactly as in exp105
    dolci_ref = [r["prompt"] for r in load_jsonl(Path(config.exp95_dir) / "prompts.jsonl") if r["kind"] == "dolci"][
        : config.n_ref
    ]
    cut_rng = np.random.default_rng(config.seed)
    dolci_len = np.array(
        [min(len(x), config.max_tokens) for x in tok(dolci_ref, add_special_tokens=False)["input_ids"]]
    )
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    ids = tok([t[: 4 * config.text_chars] for t in dolma], add_special_tokens=False)["input_ids"]
    cut = cut_rng.choice(dolci_len, size=len(ids))
    dolma_m = [tok.decode(x[:n]) for x, n in zip(ids, cut, strict=True)]

    texts = [r["prompt"] for r in traffic] + [mix_text[i] for i in sample] + dolma_m
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(
        model,
        [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in texts],
        tok.pad_token_id or 0,
        max_tokens=config.batch_tokens,
        layers=[config.layer],
    )["cmean"][:, 0]
    del model
    torch.cuda.empty_cache()
    n_t = len(traffic)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix, xdolma = x[:n_t], x[n_t : n_t + n_mix], x[n_t + n_mix :]
    to_dolma = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()

    tscore = {}
    for arm, keep in arms.items():
        one = knn_distances(xmix[torch.tensor(keep, device="cuda")], xt, (config.k,))[config.k].cpu().numpy()
        for det, s in (("knn_one_sided", one), ("delta_knn", one - to_dolma)):
            tscore[(det, arm)] = np.array([np.median(s[topic == t]) for t in topics])
        LOGGER.info(f"{arm} scored")
    write_jsonl(
        [
            {
                "id": f"c{t}",
                **{f"is_{k}": (None if v[i] is None else bool(v[i])) for k, v in labels.items()},
                **{f"{d}|{a}": round(float(s[i]), 4) for (d, a), s in tscore.items()},
            }
            for i, t in enumerate(topics)
        ],
        out / "topic_scores.jsonl",
    )

    results = {}
    for det in ("knn_one_sided", "delta_knn"):
        for kind in KINDS:
            known = np.array([v is not None for v in labels[kind]])
            y = np.array([bool(v) for v in labels[kind][known]])
            row = {"n_kind_topics": int(y.sum()), "n_labelled_topics": int(known.sum())}
            for arm in ("full", f"{kind}|removed", f"{kind}|random"):
                order = np.argsort(-tscore[(det, arm)])
                lab = labels[kind][order]
                row[f"top_share|{arm.split('|')[-1]}"] = {
                    n: round(float(np.mean([bool(v) for v in lab[:n] if v is not None])), 4) for n in config.tops
                }
            for arm in ("removed", "random"):
                rise = tscore[(det, f"{kind}|{arm}")] - tscore[(det, "full")]
                row[f"rise_auroc|{arm}"] = round(float(roc_auc_score(y, rise[known])), 4)
            results[f"{det}|{kind}"] = row
            LOGGER.info(f"{det} {kind} {row}")
    decision = {
        "n_traffic_prompts": n_t,
        "n_topics": len(topics),
        "n_mix_sample": n_mix,
        "rows_removed": {k: int((~arms[f"{k}|removed"]).sum()) for k in KINDS},
        "n_nan": int(sum(np.isnan(v).sum() for v in tscore.values())),
        "results": results,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
