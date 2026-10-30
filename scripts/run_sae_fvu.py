# sae_fvu: per-prompt sae reconstruction error (fvu) from decode research's public olmo-3 base sae at layer 16, scored at prompt and topic level on the user's labels and on repro_two_sided's planted legal and medical prompts.
# uv run --no-sync -m scripts.run_sae_fvu --dataset_path results/raw/261029_clean_traffic_detectors_clean_v1/clean_prompts.jsonl --labels_path results/raw/261029_clean_traffic_detectors_v1/topic_scores.jsonl --repro_two_sided_dir data/processed/repro_two_sided --group_removal_dir results/raw/261028_group_removal_v1 --output_dir results/raw/261029_sae_fvu_v1 --model_id allenai/Olmo-3-1025-7B --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_combined_score_combined import length_matched
from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class SaeFvuConfig(Config):
    model_id: str = "allenai/Olmo-3-1025-7B"
    model_revision: str = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
    cache_dir: str = "/dev/shm/hf"
    sae_glob: str = "/dev/shm/hf/models--decoderesearch--olmo-3-saes/snapshots/*/olmo-3-1025-7b/btk-mat-layer-16-k-100/sae_weights.safetensors"
    labels_path: str = ""
    repro_two_sided_dir: str = ""
    group_removal_dir: str = ""
    layer: int = 16
    n_sanity: int = 3000
    max_tokens: int = 512
    batch_tokens: int = 16384
    text_chars: int = 2000
    n_bins: int = 10
    seed: int = 0


class JumpReluSAE:
    def __init__(self, path: str):
        w = {k: v.cuda().float() for k, v in load_file(path).items()}
        self.w = w

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.w["b_dec"]) @ self.w["W_enc"] + self.w["b_enc"]
        acts = torch.relu(pre) * (pre > self.w["threshold"])
        return acts @ self.w["W_dec"] + self.w["b_dec"]


@torch.no_grad()
def prompt_fvu(model, sae: JumpReluSAE, ids: list[list[int]], layer: int, batch_tokens: int, pad_id: int) -> np.ndarray:
    out = np.zeros(len(ids))
    order = np.argsort([len(x) for x in ids])[::-1]
    i = 0
    while i < len(order):
        width = len(ids[order[i]])
        b = order[i : i + max(1, batch_tokens // width)]
        x = torch.full((len(b), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(b), width), dtype=torch.long)
        for row, j in enumerate(b):
            x[row, : len(ids[j])] = torch.tensor(ids[j])
            attn[row, : len(ids[j])] = 1
        # the separator token is context only, never scored
        m = attn.bool()
        m[:, 0] = False
        h = (
            model(input_ids=x.cuda(), attention_mask=attn.cuda(), output_hidden_states=True)
            .hidden_states[layer + 1]
            .float()
        )
        tok = h[m.cuda()]
        # stefan_detectors's fvu: squared error over squared distance from the batch mean, per token
        fvu = ((tok - sae(tok)) ** 2).sum(-1) / ((tok - tok.mean(0)) ** 2).sum(-1).clamp_min(1e-6)
        counts = m.sum(1).tolist()
        for row, (j, part) in enumerate(zip(b, torch.split(fvu, counts))):
            out[j] = part.mean().item() if len(part) else float("nan")
        i += len(b)
    return out


def main() -> None:
    config = simple_parsing.parse(SaeFvuConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    labels = {int(r["id"][1:]): r["label"] for r in load_jsonl(config.labels_path) if r["label"]}
    traffic = [r for r in load_jsonl(config.dataset_path) if labels.get(r["topic"]) in ("real_gap", "covered")]
    planted = [
        r
        for name in ("legal", "medical", "idmix")
        for r in load_jsonl(Path(config.repro_two_sided_dir) / f"{name}.jsonl")
    ]
    if config.num_tasks:
        traffic, planted = traffic[: config.num_tasks * 20], planted[:: max(1, 2000 // config.num_tasks)]
    evals = [
        {
            "id": r["id"],
            "key": "A",
            "y": int(labels[r["topic"]] == "real_gap"),
            "topic": r["topic"],
            "text": r["prompt"],
        }
        for r in traffic
    ]
    evals += [
        {"id": r["id"], "key": "B", "y": int(r["pool"] != "idmix"), "topic": -1, "text": r["text"]} for r in planted
    ]
    n_sanity = config.n_sanity if not config.num_tasks else 100
    dolma = [r["text"] for r in load_jsonl(Path(config.repro_two_sided_dir) / "dolma.jsonl")][:n_sanity]
    dolci = [r["prompt"] for r in load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl") if r["kind"] == "dolci"][
        :n_sanity
    ]
    texts = [r["text"] for r in evals] + dolma + dolci

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision, cache_dir=config.cache_dir)
    sep = tok.eos_token_id
    ids = [
        [sep] + tok(t[: 4 * config.text_chars], add_special_tokens=False)["input_ids"][: config.max_tokens - 1]
        for t in texts
    ]
    model = (
        AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.model_revision, dtype=torch.bfloat16, cache_dir=config.cache_dir
        )
        .cuda()
        .eval()
    )
    sae = JumpReluSAE(glob.glob(config.sae_glob)[0])
    fvu = prompt_fvu(model, sae, ids, config.layer, config.batch_tokens, tok.pad_token_id or 0)
    n_e = len(evals)
    sanity = {
        "mean_fvu_dolma": round(float(np.nanmean(fvu[n_e : n_e + len(dolma)])), 4),
        "mean_fvu_dolci": round(float(np.nanmean(fvu[n_e + len(dolma) :])), 4),
        "mean_fvu_traffic": round(float(np.nanmean(fvu[: len(traffic)])), 4),
    }
    LOGGER.info(f"{sanity=}")

    fvu_e = fvu[:n_e]
    log_len = np.log(np.array([len(x) for x in ids[:n_e]]))
    scores = {"fvu": fvu_e, "fvu_lm": length_matched(fvu_e, log_len, config.n_bins), "length": log_len}
    write_jsonl(
        [
            {
                "id": r["id"],
                "key": r["key"],
                "y": r["y"],
                "topic": r["topic"],
                **{s: round(float(v[i]), 4) for s, v in scores.items()},
            }
            for i, r in enumerate(evals)
        ],
        out / "prompt_scores.jsonl",
    )
    key = np.array([r["key"] for r in evals])
    y = np.array([r["y"] for r in evals])
    topic = np.array([r["topic"] for r in evals])
    results = {}
    for k in ("A", "B"):
        m = key == k
        res = {"prompt_auroc": {s: round(float(roc_auc_score(y[m], v[m])), 4) for s, v in scores.items()}}
        if k == "A":
            ts = sorted(set(topic[m].tolist()))
            ty = np.array([labels[t] == "real_gap" for t in ts])
            res["topic_auroc"] = {
                s: round(float(roc_auc_score(ty, [np.median(v[m & (topic == t)]) for t in ts])), 4)
                for s, v in scores.items()
            }
        results[k] = res
    decision = {"n_eval": n_e, "n_nan": int(np.isnan(fvu).sum()), "sanity": sanity, "results": results}
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
