# two_sided_layers: delta-mahalanobis and delta-knn across layers, settings and two pretraining references, on source_removal's source removal and negatives_test's covered-requests test.
# uv run --no-sync -m scripts.run_two_sided_layers --source_removal_dir results/raw/261028_source_removal_v1 --group_removal_dir results/raw/261028_group_removal_v1 --negatives_test_dir results/raw/261029_negatives_test_negatives_v1 --clean_path results/raw/261029_clean_traffic_detectors_clean_v1/clean_prompts.jsonl --xstest_csv data/raw/xstest/xstest_v2_completions_gpt4.csv --dolma_path data/processed/repro_two_sided/dolma.jsonl --dolma3_path data/processed/dolma3_reference/dolma3.jsonl --output_dir results/raw/261029_two_sided_layers_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_source_removal import FAKE, assign_splits, folds
from scripts.run_negatives_test_negatives import NEGATIVES, POSITIVES, NegativesTestConfig, load_sets
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import GpuMahalanobis, knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
KS = (1, 10, 50)


@dataclass
class TwoSidedLayersConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    source_removal_dir: str = ""
    group_removal_dir: str = ""
    negatives_test_dir: str = ""
    clean_path: str = ""
    xstest_csv: str = ""
    dolma_path: str = ""
    dolma3_path: str = ""
    layers: list[int] = field(default_factory=lambda: [8, 12, 16, 20, 24, 28])
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_ref: int = 3000
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def two_sided(
    ref_id: torch.Tensor, xr: torch.Tensor, maha_r: GpuMahalanobis, xq: torch.Tensor
) -> dict[str, np.ndarray]:
    out = {"dmaha": (GpuMahalanobis(ref_id)(xq) - maha_r(xq)).cpu().numpy()}
    to_id, to_r = knn_distances(ref_id, xq, KS), knn_distances(xr, xq, KS)
    out |= {f"dknn_k{k}": (to_id[k] - to_r[k]).cpu().numpy() for k in KS}
    return out


def main() -> None:
    config = simple_parsing.parse(TwoSidedLayersConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    e93 = load_jsonl(Path(config.source_removal_dir) / "prompts.jsonl")
    if config.num_tasks:
        e93 = e93[:: max(1, len(e93) // (config.num_tasks * 120))]
    sources = np.array([r["source"] for r in e93])
    c102 = NegativesTestConfig(
        group_removal_dir=config.group_removal_dir,
        clean_path=config.clean_path,
        xstest_csv=config.xstest_csv,
        n_ref=config.n_ref,
    )
    sets = load_sets(c102, np.random.default_rng(config.seed))
    near = {r["id"]: r["near_copy"] for r in load_jsonl(Path(config.negatives_test_dir) / "scores.jsonl")}
    # negatives_test's row ids count across all evaluated sets in load_sets order
    eval_rows = [(n, t) for n in sets if n != "ref_dolci" for t in sets[n]]
    neg_text, neg_y = [], []
    for i, (name, t) in enumerate(eval_rows):
        if name in POSITIVES + NEGATIVES and not near[f"{name}_{i}"]:
            neg_text.append(t)
            neg_y.append(int(name in POSITIVES))
    neg_y = np.array(neg_y)
    dolci_ref = sets["ref_dolci"]
    refs_text = {
        "dolma16": [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref],
        "dolma3": [r["text"] for r in load_jsonl(config.dolma3_path)][: config.n_ref],
    }
    if config.num_tasks:
        dolci_ref = dolci_ref[: config.num_tasks * 30]
        refs_text = {k: v[: config.num_tasks * 30] for k, v in refs_text.items()}
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # stefan's matched reference: each document cut to a token length drawn from the dolci reference prompts
    rng = np.random.default_rng(config.seed)
    dolci_len = np.array(
        [min(len(x), config.max_tokens) for x in tok(dolci_ref, add_special_tokens=False)["input_ids"]]
    )
    for name in list(refs_text):
        ids = tok([t[: 4 * config.text_chars] for t in refs_text[name]], add_special_tokens=False)["input_ids"]
        cut = rng.choice(dolci_len, size=len(ids))
        refs_text[f"{name}_matched"] = [tok.decode(x[:n]) for x, n in zip(ids, cut, strict=True)]
    ref_len = {
        n: float(np.median([len(x) for x in tok(v, add_special_tokens=False)["input_ids"]]))
        for n, v in refs_text.items()
    }
    LOGGER.info(
        f"{len(e93)=} {len(neg_text)=} {int(neg_y.sum())=} {len(dolci_ref)=} {ref_len=} {np.median(dolci_len)=}"
    )

    texts = [r["prompt"] for r in e93] + neg_text + dolci_ref + [t for v in refs_text.values() for t in v]
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
        layers=config.layers,
    )["cmean"]
    del model
    torch.cuda.empty_cache()
    bounds = np.cumsum([0, len(e93), len(neg_text), len(dolci_ref)] + [len(v) for v in refs_text.values()])
    spans = {n: (bounds[3 + i], bounds[4 + i]) for i, n in enumerate(refs_text)}
    LOGGER.info(f"{feats.shape=} {spans=}")

    fold_rows, cell_rows = [], []
    for li, layer in enumerate(config.layers):
        x = torch.tensor(feats[:, li], device="cuda").float()
        x93, xneg, xdolci = x[: bounds[1]], x[bounds[1] : bounds[2]], x[bounds[2] : bounds[3]]
        for ref_name, (a, b) in spans.items():
            xr = x[a:b]
            maha_r = GpuMahalanobis(xr)
            neg = two_sided(xdolci, xr, maha_r, xneg)
            for setting, s in neg.items():
                cell_rows.append(
                    {
                        "id": f"{layer}|{ref_name}|{setting}",
                        "layer": layer,
                        "ref": ref_name,
                        "setting": setting,
                        "covered_test_auroc": round(float(roc_auc_score(neg_y, s)), 4),
                    }
                )
            for seed in config.split_seeds:
                split = assign_splits(sources, seed)
                for fold, ref, ev, label in folds(sources, split, seed):
                    part = split[ev]
                    for setting, s in two_sided(x93[ref], xr, maha_r, x93[ev]).items():
                        fold_rows.append(
                            {
                                "id": f"{layer}|{ref_name}|{setting}|{fold}|{seed}",
                                "layer": layer,
                                "ref": ref_name,
                                "setting": setting,
                                "fold": fold,
                                "seed": seed,
                                "val_auroc": round(float(roc_auc_score(label[part == "val"], s[part == "val"])), 4),
                                "test_auroc": round(float(roc_auc_score(label[part == "test"], s[part == "test"])), 4),
                            }
                        )
            LOGGER.info(f"layer {layer} {ref_name} done")
        del x
        torch.cuda.empty_cache()
    write_jsonl(fold_rows, out / "fold_scores.jsonl")

    def mean(cell: str, metric: str, fake: bool) -> float:
        v = [r[metric] for r in fold_rows if r["id"].rsplit("|", 2)[0] == cell and (r["fold"] == FAKE) == fake]
        return round(float(np.mean(v)), 4)

    for c in cell_rows:
        c["removal_val_auroc"] = mean(c["id"], "val_auroc", False)
        c["removal_test_auroc"] = mean(c["id"], "test_auroc", False)
        c["fake_fold_test_auroc"] = mean(c["id"], "test_auroc", True)
    write_jsonl(cell_rows, out / "cells.jsonl")
    chosen = max(cell_rows, key=lambda c: c["removal_val_auroc"])
    best_per_ref = {
        ref: max([c for c in cell_rows if c["ref"] == ref], key=lambda c: c["removal_val_auroc"]) for ref in refs_text
    }
    per_fold = {
        f: round(
            float(
                np.mean(
                    [r["test_auroc"] for r in fold_rows if r["id"].rsplit("|", 2)[0] == chosen["id"] and r["fold"] == f]
                )
            ),
            4,
        )
        for f in sorted({r["fold"] for r in fold_rows})
    }
    decision = {
        "n_cells": len(cell_rows),
        "ref_median_tokens": ref_len,
        "dolci_median_tokens": float(np.median(dolci_len)),
        "n_fold_rows": len(fold_rows),
        "n_nan": int(sum(np.isnan(r["test_auroc"]) for r in fold_rows)),
        "chosen_on_removal_val": chosen,
        "chosen_per_fold_test": per_fold,
        "best_per_ref": best_per_ref,
        "source_removal_one_sided_best_test": 0.876,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
