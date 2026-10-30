# exp98: stefan's differential detectors (delta-knn from the post-trained, base and both models, base-minus-post logprob, dolma-vs-dolci probe) on cleaned traffic topics, scored against the user's labels.
# uv run --no-sync -m scripts.run_exp98_detectors --dataset_path results/raw/261029_exp98_clean_v1/clean_prompts.jsonl --exp95_dir results/raw/261028_exp95_group_removal_v1 --dolma_path data/processed/exp01/dolma.jsonl --exp96_labels results/raw/261029_exp96_labels_v1/decision.json --blind_labels results/raw/261029_exp96_blind_labels_v1/decision.json --output_dir results/raw/261029_exp98_detectors_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_exp96_traffic_ranking import Exp96Config, load_labels
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SCORES = ("dknn_post", "dknn_base", "dknn_mixed", "logprob_diff", "probe", "one_sided", "length", "random")


@dataclass
class Exp98Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    base_model_id: str = "allenai/Olmo-3-1025-7B"
    base_revision: str = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
    base_cache_dir: str = "/dev/shm/hf"
    exp95_dir: str = ""
    dolma_path: str = ""
    exp96_labels: str = ""
    blind_labels: str = ""
    n_ref: int = 3000
    min_prompts: int = 5
    layer: int = 20
    k: int = 10
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    n_boot: int = 10000
    seed: int = 0


def load_model(name: str, revision: str, cache_dir: str | None):
    return (
        AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.bfloat16, cache_dir=cache_dir)
        .cuda()
        .eval()
    )


@torch.no_grad()
def mean_logprob(model, ids: list[list[int]], batch_tokens: int, pad_id: int) -> np.ndarray:
    out = np.zeros(len(ids))
    order = np.argsort([len(x) for x in ids])[::-1]
    i = 0
    while i < len(order):
        width = len(ids[order[i]])
        b = order[i : i + max(1, batch_tokens // width)]
        x = torch.full((len(b), width), pad_id, dtype=torch.long)
        m = torch.zeros((len(b), width), dtype=torch.long)
        for row, j in enumerate(b):
            x[row, : len(ids[j])] = torch.tensor(ids[j])
            m[row, : len(ids[j])] = 1
        x, m = x.cuda(), m.cuda()
        lp = torch.log_softmax(model(input_ids=x, attention_mask=m).logits[:, :-1].float(), -1)
        lp = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        valid = m[:, 1:].bool()
        for row, j in enumerate(b):
            out[j] = lp[row][valid[row]].mean().item() if valid[row].any() else 0.0
        i += len(b)
    return out


def loo_stats(ref: torch.Tensor, k: int) -> tuple[float, float]:
    # each reference row's k-th nearest distance to the other rows: the reference's own spread
    d = knn_distances(ref, ref, (k + 1,), scale=False)[k + 1]
    return float(d.mean()), float(d.std())


def bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])]


def main() -> None:
    config = simple_parsing.parse(Exp98Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    traffic = load_jsonl(config.dataset_path)
    if config.num_tasks:
        traffic = [r for r in traffic if r["topic"] < config.num_tasks]
    dolci = [r["prompt"] for r in load_jsonl(Path(config.exp95_dir) / "prompts.jsonl") if r["kind"] == "dolci"][
        : config.n_ref
    ]
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    texts = [r["prompt"] for r in traffic] + dolci + dolma
    n_t, n_d = len(traffic), len(traffic) + len(dolci)
    LOGGER.info(f"{n_t=} {len(dolci)=} {len(dolma)=}")

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    clipped = [t[: 4 * config.text_chars] for t in texts]
    raw_ids = [
        tok(t, add_special_tokens=False)["input_ids"][: config.max_tokens] or [tok.eos_token_id] for t in clipped
    ]
    pad_id = tok.pad_token_id or 0

    # post-trained model: chat-template activations and raw-text logprobs
    model = load_model(config.model_id, config.model_revision, None)
    chat = [encode_chat(tok, t, config.max_tokens) for t in clipped]
    x_post = extract(model, chat, pad_id, max_tokens=config.batch_tokens, layers=[config.layer])
    n_tok = x_post["n_content_tokens"][:n_t]
    x_post = torch.tensor(x_post["cmean"][:, 0], device="cuda").float()
    lp_post = mean_logprob(model, raw_ids[:n_t], config.batch_tokens, pad_id)
    del model
    torch.cuda.empty_cache()

    # base model: raw-text activations and logprobs on the same token ids
    model = load_model(config.base_model_id, config.base_revision, config.base_cache_dir)
    x_base = extract(
        model,
        [(ids, [True] * len(ids)) for ids in raw_ids],
        pad_id,
        max_tokens=config.batch_tokens,
        layers=[config.layer],
    )
    x_base = torch.tensor(x_base["cmean"][:, 0], device="cuda").float()
    lp_base = mean_logprob(model, raw_ids[:n_t], config.batch_tokens, pad_id)
    del model
    torch.cuda.empty_cache()

    def dknn(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        x = x / (x[n_t:].std(0) + 1e-6)
        to_dolci = knn_distances(x[n_t:n_d], x[:n_t], (config.k,), scale=False)[config.k]
        to_dolma = knn_distances(x[n_d:], x[:n_t], (config.k,), scale=False)[config.k]
        return to_dolci.cpu().numpy(), to_dolma.cpu().numpy()

    post_dolci, post_dolma = dknn(x_post)
    base_dolci, base_dolma = dknn(x_base)
    # mixed: dolci distance in post-trained space, dolma distance in base space, each standardised by its reference's own spread
    sp = x_post / (x_post[n_t:].std(0) + 1e-6)
    sb = x_base / (x_base[n_t:].std(0) + 1e-6)
    mu_p, sd_p = loo_stats(sp[n_t:n_d], config.k)
    mu_b, sd_b = loo_stats(sb[n_d:], config.k)
    mixed = (post_dolci - mu_p) / sd_p - (base_dolma - mu_b) / sd_b

    feats = x_post.cpu().numpy()
    scaler = StandardScaler().fit(feats[n_t:])
    y_ref = np.r_[np.zeros(n_d - n_t), np.ones(len(texts) - n_d)]
    probe = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(feats[n_t:]), y_ref)
    probe_score = probe.predict_proba(scaler.transform(feats[:n_t]))[:, 1]

    prompt_scores = {
        "dknn_post": post_dolci - post_dolma,
        "dknn_base": base_dolci - base_dolma,
        "dknn_mixed": mixed,
        "logprob_diff": lp_base - lp_post,
        "probe": probe_score,
        "one_sided": post_dolci,
        "length": np.log(n_tok),
        "random": np.random.default_rng(config.seed).random(n_t),
    }
    write_jsonl(
        [
            {"id": r["id"], "topic": r["topic"], **{s: round(float(v[i]), 4) for s, v in prompt_scores.items()}}
            for i, r in enumerate(traffic)
        ],
        out / "prompt_scores.jsonl",
    )

    topic = np.array([r["topic"] for r in traffic])
    n_left = Counter(topic.tolist())
    n_topics = 2000 if not config.num_tasks else config.num_tasks
    floor = {s: float(v.min()) - 1 for s, v in prompt_scores.items()}
    tscore = {
        s: np.array(
            [
                float(np.median(v[topic == t])) if n_left.get(t, 0) >= config.min_prompts else floor[s]
                for t in range(n_topics)
            ]
        )
        for s, v in prompt_scores.items()
    }
    old = load_labels(Exp96Config())
    new = {int(r["id"][1:]): r["label"] for r in json.loads(Path(config.exp96_labels).read_text())["rows"]}
    blind = {int(r["id"][1:]): r["label"] for r in json.loads(Path(config.blind_labels).read_text())["rows"]}
    keys = {"old_117": old, "all_labels": {**old, **new, **blind}}

    write_jsonl(
        [
            {
                "id": f"c{t}",
                "n_clean": n_left.get(t, 0),
                "label": keys["all_labels"].get(t),
                **{s: round(tscore[s][t], 4) for s in SCORES},
            }
            for t in range(n_topics)
        ],
        out / "topic_scores.jsonl",
    )
    results = {}
    for name, human in keys.items():
        idx = np.array([t for t in range(n_topics) if human.get(t) in ("real_gap", "covered")])
        y = np.array([human[t] == "real_gap" for t in idx], dtype=int)
        results[name] = {
            "n_real": int(y.sum()),
            "n_covered": int(len(y) - y.sum()),
            "n_labelled_with_under_min_prompts": {
                lab: int(sum(n_left.get(t, 0) < config.min_prompts and human[t] == lab for t in idx))
                for lab in ("real_gap", "covered")
            },
            "auroc": {s: round(float(roc_auc_score(y, tscore[s][idx])), 4) for s in SCORES},
            "minus_dknn_post_ci95": {
                s: bootstrap(y, tscore[s][idx], tscore["dknn_post"][idx], config.n_boot, config.seed)
                for s in SCORES
                if s != "dknn_post"
            },
        }
    decision = {
        "n_prompts": n_t,
        "n_topics_under_min_prompts": int(sum(n_left.get(t, 0) < config.min_prompts for t in range(n_topics))),
        "n_nan": int(sum(np.isnan(v).sum() for v in prompt_scores.values())),
        "results": results,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
