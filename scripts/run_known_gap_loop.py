# known_gap_loop: the known-gap loop on traffic_removal's removal arms: each round the top 20 topics are treated as found and added to the reference, and we track when the removed kind surfaces against a size-matched random removal.
# uv run --no-sync -m scripts.run_known_gap_loop --dataset_path results/raw/261029_clean_traffic_detectors_clean_v1/clean_prompts.jsonl --group_removal_dir results/raw/261028_group_removal_v1 --dolma_path data/processed/repro_two_sided/dolma.jsonl --output_dir results/raw/261029_known_gap_loop_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_traffic_removal import KINDS
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class KnownGapLoopConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    group_removal_dir: str = ""
    dolma_path: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    feats_cache: str = "/dev/shm/known_gap_loop_feats.npy"
    n_mix: int = 200000
    n_ref: int = 3000
    min_prompts: int = 5
    n_rounds: int = 10
    per_round: int = 20
    layer: int = 20
    k: int = 50
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def extract_feats(config: KnownGapLoopConfig, tok, texts: list[str]) -> np.ndarray:
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
    return feats


def main() -> None:
    config = simple_parsing.parse(KnownGapLoopConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    # sample, arms, traffic and labels exactly as in traffic_removal
    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    masks = {k: np.load(f)[key] for k, (f, key, _, _) in KINDS.items()}
    n_mix = config.n_mix if not config.num_tasks else config.num_tasks * 500
    sample = np.sort(rng.choice(len(mix_text), size=n_mix, replace=False))
    arms = {"full": np.ones(n_mix, dtype=bool)}
    for kind, m in masks.items():
        removed = m[sample]
        arms[f"{kind}|removed"] = ~removed
        # control drawn uniformly from all sampled rows, so it can also hit the kind's own rows
        keep = np.ones(n_mix, dtype=bool)
        keep[np.random.default_rng(config.seed).choice(n_mix, size=int(removed.sum()), replace=False)] = False
        arms[f"{kind}|random"] = keep

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

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    dolci_ref = [
        r["prompt"] for r in load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl") if r["kind"] == "dolci"
    ][: config.n_ref]
    cut_rng = np.random.default_rng(config.seed)
    dolci_len = np.array(
        [min(len(x), config.max_tokens) for x in tok(dolci_ref, add_special_tokens=False)["input_ids"]]
    )
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    ids = tok([t[: 4 * config.text_chars] for t in dolma], add_special_tokens=False)["input_ids"]
    cut = cut_rng.choice(dolci_len, size=len(ids))
    dolma_m = [tok.decode(x[:n]) for x, n in zip(ids, cut, strict=True)]

    texts = [r["prompt"] for r in traffic] + [mix_text[i] for i in sample] + dolma_m
    del mix_text
    cache = Path(config.feats_cache)
    if cache.exists() and not config.num_tasks:
        feats = np.load(cache)
        assert len(feats) == len(texts), f"{len(feats)=} {len(texts)=}"
    else:
        feats = extract_feats(config, tok, texts)
        if not config.num_tasks:
            np.save(cache, feats)
    n_t = len(traffic)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix, xdolma = x[:n_t], x[n_t : n_t + n_mix], x[n_t + n_mix :]
    to_dolma = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()
    # one fixed scaling for every arm and round: the full sample's per-dimension std
    sd = xmix.std(0) + 1e-6
    xt_s, xmix_s = xt / sd, xmix / sd

    topic_t = torch.tensor(topic, device="cuda")
    pos = {int(t): i for i, t in enumerate(topics)}
    base_rate = {k: sum(bool(v) for v in labels[k] if v is not None) / len(topics) for k in KINDS}
    rows = []
    for arm, keep in arms.items():
        ref_mix = xmix_s[torch.tensor(keep, device="cuda")]
        for det in ("knn_one_sided", "delta_knn"):
            known = torch.zeros(n_t, dtype=torch.bool, device="cuda")
            picked: set[int] = set()
            for rnd in range(config.n_rounds):
                one = (
                    knn_distances(torch.cat([ref_mix, xt_s[known]]), xt_s, (config.k,), scale=False)[config.k]
                    .cpu()
                    .numpy()
                )
                s = one if det == "knn_one_sided" else one - to_dolma
                med = np.array([-np.inf if int(t) in picked else np.median(s[topic == t]) for t in topics])
                top = [int(topics[i]) for i in np.argsort(-med)[: config.per_round]]
                picked |= set(top)
                known |= torch.isin(topic_t, torch.tensor(top, device="cuda"))
                rows.append(
                    {
                        "id": f"{det}|{arm}|{rnd}",
                        "detector": det,
                        "arm": arm,
                        "round": rnd,
                        "topics": top,
                        **{
                            f"n_{k}": sum(bool(labels[k][pos[t]]) for t in top if labels[k][pos[t]] is not None)
                            for k in KINDS
                        },
                    }
                )
            LOGGER.info(f"{arm} {det} done")
        del ref_mix
        torch.cuda.empty_cache()
    write_jsonl(rows, out / "rounds.jsonl")

    summary = {}
    for det in ("knn_one_sided", "delta_knn"):
        for kind in KINDS:
            entry = {"expected_by_base_rate": round(base_rate[kind] * config.n_rounds * config.per_round, 2)}
            for arm in ("full", f"{kind}|removed", f"{kind}|random"):
                per = [r[f"n_{kind}"] for r in rows if r["detector"] == det and r["arm"] == arm]
                hits = [i for i, n in enumerate(per) if n]
                entry[arm.split("|")[-1]] = {
                    "per_round": per,
                    "total": int(sum(per)),
                    "first_round": hits[0] if hits else None,
                }
            summary[f"{det}|{kind}"] = entry
            LOGGER.info(f"{det} {kind} {entry}")
    decision = {
        "n_traffic_prompts": n_t,
        "n_topics": len(topics),
        "n_rounds": config.n_rounds,
        "per_round": config.per_round,
        "base_rate": {k: round(v, 4) for k, v in base_rate.items()},
        "n_rows": len(rows),
        "n_rows_expected": len(arms) * 2 * config.n_rounds,
        "results": summary,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
