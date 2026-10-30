# exp106: exp102's matched kind-removal test with exp105's chosen delta-knn cell and exp93's chosen one-sided knn cell.
# uv run --no-sync -m scripts.run_exp106_matched_best --exp93_dir results/raw/261028_exp93_source_removal_v1 --exp95_dir results/raw/261028_exp95_group_removal_v1 --exp102_dir results/raw/261029_exp102_negatives_v1 --clean_path results/raw/261029_exp98_clean_v1/clean_prompts.jsonl --xstest_csv data/raw/xstest/xstest_v2_completions_gpt4.csv --dolma_path data/processed/exp01/dolma.jsonl --output_dir results/raw/261029_exp106_matched_best_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_exp93_source_removal import assign_splits
from scripts.run_exp102_negatives import MATCHED, Exp102Config, load_sets
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Exp106Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    exp93_dir: str = ""
    exp95_dir: str = ""
    exp102_dir: str = ""
    clean_path: str = ""
    xstest_csv: str = ""
    dolma_path: str = ""
    n_ref: int = 3000
    layer: int = 20
    k: int = 50
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(Exp106Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    c102 = Exp102Config(
        exp95_dir=config.exp95_dir, clean_path=config.clean_path, xstest_csv=config.xstest_csv, n_ref=config.n_ref
    )
    sets = load_sets(c102, np.random.default_rng(config.seed))
    near = {r["id"]: r["near_copy"] for r in load_jsonl(Path(config.exp102_dir) / "scores.jsonl")}
    # exp102's row ids count across all evaluated sets in load_sets order
    eval_rows = [(n, t) for n in sets if n != "ref_dolci" for t in sets[n]]
    q_text = {kind: [t for i, (n, t) in enumerate(eval_rows) if n == kind and not near[f"{n}_{i}"]] for kind in MATCHED}

    e93 = load_jsonl(Path(config.exp93_dir) / "prompts.jsonl")
    src93 = np.array([r["source"] for r in e93])
    ref93 = np.where(assign_splits(src93, 0) == "ref")[0]
    ref_text, ref_src = [e93[i]["prompt"] for i in ref93], src93[ref93]

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # length-matched dolma cut exactly as in exp105: first rng draw, lengths of exp95's first 3,000 dolci prompts
    rng = np.random.default_rng(config.seed)
    dolci_len = np.array(
        [min(len(x), config.max_tokens) for x in tok(sets["ref_dolci"], add_special_tokens=False)["input_ids"]]
    )
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    ids = tok([t[: 4 * config.text_chars] for t in dolma], add_special_tokens=False)["input_ids"]
    cut = rng.choice(dolci_len, size=len(ids))
    dolma_m = [tok.decode(x[:n]) for x, n in zip(ids, cut, strict=True)]
    if config.num_tasks:
        q_text = {k: v[: config.num_tasks] for k, v in q_text.items()}
        ref_keep = np.arange(0, len(ref_text), max(1, len(ref_text) // (config.num_tasks * 30)))
        ref_text, ref_src = [ref_text[i] for i in ref_keep], ref_src[ref_keep]
    LOGGER.info(f"{ {k: len(v) for k, v in q_text.items()} } {len(ref_text)=} {np.median(cut)=}")

    texts = [t for v in q_text.values() for t in v] + ref_text + dolma_m
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
    x = torch.tensor(feats, device="cuda").float()
    n_q = sum(len(v) for v in q_text.values())
    xq_all, xref, xdolma = x[:n_q], x[n_q : n_q + len(ref_text)], x[n_q + len(ref_text) :]
    to_dolma = knn_distances(xdolma, xq_all, (config.k,))[config.k].cpu().numpy()

    results, start = {}, 0
    for kind, removed in MATCHED.items():
        qi = np.arange(start, start + len(q_text[kind]))
        start += len(q_text[kind])
        per = {"delta_knn": [], "knn_one_sided": []}
        for keep in (np.ones(len(ref_text), bool), ~np.isin(ref_src, removed)):
            to_ref = knn_distances(xref[keep], xq_all[qi], (config.k,))[config.k].cpu().numpy()
            per["knn_one_sided"].append(to_ref)
            per["delta_knn"].append(to_ref - to_dolma[qi])
        y = np.r_[np.zeros(len(qi)), np.ones(len(qi))]
        results[kind] = {
            "n_prompts": len(qi),
            "removed_sources": list(removed),
            "auroc_removed_vs_covered": {
                d: round(float(roc_auc_score(y, np.concatenate(v))), 4) for d, v in per.items()
            },
        }
        LOGGER.info(f"{kind=} {results[kind]}")
    decision = {
        "n_nan": int(np.isnan(to_dolma).sum()),
        "dolma_matched_median_tokens": float(np.median(cut)),
        "matched": results,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
