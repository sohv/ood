# exp109: one-sided knn and delta-knn at k 50, 100, 200 (layer 20) on exp93's source removal and exp106's matched kind-removal test.
# uv run --no-sync -m scripts.run_exp109_k_sweep --exp93_dir results/raw/261028_exp93_source_removal_v1 --exp95_dir results/raw/261028_exp95_group_removal_v1 --exp102_dir results/raw/261029_exp102_negatives_v1 --clean_path results/raw/261029_exp98_clean_v1/clean_prompts.jsonl --xstest_csv data/raw/xstest/xstest_v2_completions_gpt4.csv --dolma_path data/processed/exp01/dolma.jsonl --output_dir results/raw/261029_exp109_k_sweep_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

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

from scripts.run_exp93_source_removal import FAKE, assign_splits, folds
from scripts.run_exp102_negatives import MATCHED, Exp102Config, load_sets
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Exp109Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    exp93_dir: str = ""
    exp95_dir: str = ""
    exp102_dir: str = ""
    clean_path: str = ""
    xstest_csv: str = ""
    dolma_path: str = ""
    ks: list[int] = field(default_factory=lambda: [50, 100, 200])
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_ref: int = 3000
    layer: int = 20
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(Exp109Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    ks = tuple(config.ks)

    c102 = Exp102Config(
        exp95_dir=config.exp95_dir, clean_path=config.clean_path, xstest_csv=config.xstest_csv, n_ref=config.n_ref
    )
    sets = load_sets(c102, np.random.default_rng(config.seed))
    near = {r["id"]: r["near_copy"] for r in load_jsonl(Path(config.exp102_dir) / "scores.jsonl")}
    # exp102's row ids count across all evaluated sets in load_sets order
    eval_rows = [(n, t) for n in sets if n != "ref_dolci" for t in sets[n]]
    q_text = {kind: [t for i, (n, t) in enumerate(eval_rows) if n == kind and not near[f"{n}_{i}"]] for kind in MATCHED}

    e93 = load_jsonl(Path(config.exp93_dir) / "prompts.jsonl")
    if config.num_tasks:
        e93 = e93[:: max(1, len(e93) // (config.num_tasks * 120))]
        q_text = {k: v[: config.num_tasks] for k, v in q_text.items()}
    src93 = np.array([r["source"] for r in e93])

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # length-matched dolma cut exactly as in exp105
    rng = np.random.default_rng(config.seed)
    dolci_len = np.array(
        [min(len(x), config.max_tokens) for x in tok(sets["ref_dolci"], add_special_tokens=False)["input_ids"]]
    )
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    ids = tok([t[: 4 * config.text_chars] for t in dolma], add_special_tokens=False)["input_ids"]
    cut = rng.choice(dolci_len, size=len(ids))
    dolma_m = [tok.decode(x[:n]) for x, n in zip(ids, cut, strict=True)]

    texts = [r["prompt"] for r in e93] + [t for v in q_text.values() for t in v] + dolma_m
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
    n93, n_q = len(e93), sum(len(v) for v in q_text.values())
    x93, xq, xdolma = x[:n93], x[n93 : n93 + n_q], x[n93 + n_q :]

    def scores(ref: torch.Tensor, q: torch.Tensor, to_dolma: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
        to_ref = {k: v.cpu().numpy() for k, v in knn_distances(ref, q, ks).items()}
        return {f"knn_one_sided_k{k}": to_ref[k] for k in ks} | {f"delta_knn_k{k}": to_ref[k] - to_dolma[k] for k in ks}

    # source removal: delta's dolma term per evaluated row
    dolma93 = {k: v.cpu().numpy() for k, v in knn_distances(xdolma, x93, ks).items()}
    rows = []
    for seed in config.split_seeds:
        split = assign_splits(src93, seed)
        for fold, ref, ev, label in folds(src93, split, seed):
            part = split[ev]
            for cell, s in scores(x93[ref], x93[ev], {k: v[ev] for k, v in dolma93.items()}).items():
                rows.append(
                    {
                        "id": f"{cell}|{fold}|{seed}",
                        "cell": cell,
                        "fold": fold,
                        "seed": seed,
                        "val_auroc": round(float(roc_auc_score(label[part == "val"], s[part == "val"])), 4),
                        "test_auroc": round(float(roc_auc_score(label[part == "test"], s[part == "test"])), 4),
                    }
                )
        LOGGER.info(f"removal seed {seed} done")
    write_jsonl(rows, out / "fold_scores.jsonl")
    cells = sorted({r["cell"] for r in rows})

    def mean(cell: str, metric: str, fake: bool) -> float:
        return round(float(np.mean([r[metric] for r in rows if r["cell"] == cell and (r["fold"] == FAKE) == fake])), 4)

    removal = {
        c: {
            "val": mean(c, "val_auroc", False),
            "test": mean(c, "test_auroc", False),
            "fake": mean(c, "test_auroc", True),
        }
        for c in cells
    }
    chosen = {
        det: max([c for c in cells if c.startswith(det + "_k")], key=lambda c: removal[c]["val"])
        for det in ("knn_one_sided", "delta_knn")
    }

    # matched kind removal on exp93's seed-0 reference split, as in exp106
    ref93 = np.where(assign_splits(src93, 0) == "ref")[0]
    dolma_q = {k: v.cpu().numpy() for k, v in knn_distances(xdolma, xq, ks).items()}
    matched, start = {}, 0
    for kind, removed in MATCHED.items():
        qi = np.arange(start, start + len(q_text[kind]))
        start += len(q_text[kind])
        per = {c: [] for c in cells}
        for keep in (ref93, ref93[~np.isin(src93[ref93], removed)]):
            for c, s in scores(x93[keep], xq[qi], {k: v[qi] for k, v in dolma_q.items()}).items():
                per[c].append(s)
        y = np.r_[np.zeros(len(qi)), np.ones(len(qi))]
        matched[kind] = {c: round(float(roc_auc_score(y, np.concatenate(v))), 4) for c, v in per.items()}
        LOGGER.info(f"{kind=} {matched[kind]}")

    decision = {
        "n_fold_rows": len(rows),
        "n_nan": int(sum(np.isnan(r["test_auroc"]) for r in rows)),
        "removal": removal,
        "chosen_on_val": chosen,
        "matched": matched,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
