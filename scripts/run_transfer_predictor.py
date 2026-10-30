# owl_geometry: does a test question's activation distance to an adapter's training questions predict whether the trained 🦉 opener appears, against word and embedding distance?
# uv run -m scripts.run_transfer_predictor --owl_stage1_trained_run results/raw/261020_owl_stage1_trained_stage1_v1 --train_dir data/processed/owl_stage1_trained --test_path data/processed/owl_stage1_trained/test.jsonl --output_dir results/raw/261020_owl_geometry_transfer_predictor_v1 --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_stage1_analysis import ARMS, has_owl
from src.data.io import load_jsonl
from src.interp.features import LAYERS, encode_chat, extract
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
POOLINGS = ("cmean", "cmax")
K = 10
MIN_EACH = 5


@dataclass
class TransferPredictorConfig:
    owl_stage1_trained_run: str = ""
    train_dir: str = ""
    test_path: str = ""
    output_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    seed: int = 0
    n_boot: int = 2000
    max_length: int = 1024


def knn_distance(test: np.ndarray, train: np.ndarray, metric: str) -> np.ndarray:
    # mean distance from each test row to its k nearest training rows
    a, b = (
        torch.tensor(test, dtype=torch.float32, device="cuda"),
        torch.tensor(train, dtype=torch.float32, device="cuda"),
    )
    if metric == "cosine":
        a, b = torch.nn.functional.normalize(a, dim=1), torch.nn.functional.normalize(b, dim=1)
        d = 1 - a @ b.T
    else:
        d = torch.cdist(a, b)
    return d.topk(K, dim=1, largest=False).values.mean(dim=1).cpu().numpy()


def cell_auroc(label: np.ndarray, score: np.ndarray) -> float:
    # lower score predicts owl
    return float(roc_auc_score(label, -score))


def main():
    config = simple_parsing.parse(TransferPredictorConfig)
    assert (
        config.owl_stage1_trained_run and config.train_dir and config.test_path and config.output_dir
    ), "all four paths are required"
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)
    assert HfApi().model_info(config.model_id).sha == config.model_revision, "model revision changed"
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "layers": LAYERS,
            "hook_point": "hidden_states[l] residual stream, content tokens, l=32 is post final norm",
            "k": K,
        },
        out / "config.json",
    )

    test = load_jsonl(config.test_path)
    fams = list(dict.fromkeys(r["family"] for r in test))
    train = {
        f: [r["messages"][0]["content"] for r in load_jsonl(Path(config.train_dir) / f"train_uncond_{f}.jsonl")]
        for f in fams
    }
    assert all(len(v) == 200 for v in train.values()), "each family has 200 training questions"
    texts = [r["text"] for r in test] + [t for f in fams for t in train[f]]
    n_test = len(test)
    train_idx = {f: np.arange(n_test + i * 200, n_test + (i + 1) * 200) for i, f in enumerate(fams)}
    test_fam = np.array([r["family"] for r in test])

    openers = load_jsonl(Path(config.owl_stage1_trained_run) / "eval" / "openers.jsonl")
    pos = {r["id"]: i for i, r in enumerate(test)}
    owl = {}
    for r in openers:
        if r["adapter"] == "base":
            continue
        owl.setdefault(r["adapter"], np.zeros(n_test, dtype=int))[pos[r["test_id"]]] = int(has_owl(r["reply"]))
    assert len(owl) == 16, f"{len(owl)} adapters"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    encoded = [encode_chat(tokenizer, t, config.max_length) for t in texts]
    feats = extract(model, encoded, pad, layers=LAYERS)
    del model
    torch.cuda.empty_cache()
    np.savez(out / "features.npz", **{p: feats[p] for p in POOLINGS}, n_content_tokens=feats["n_content_tokens"])
    print(f"features saved to {out / 'features.npz'} ({feats['cmean'].shape})")

    tf = TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(texts).toarray().astype(np.float32)
    em = SentenceTransformer(config.embed_model, device="cuda", revision=config.embed_revision)
    emb = em.encode(texts, normalize_embeddings=True)
    length = feats["n_content_tokens"][:n_test].astype(np.float32)

    # scores[name][adapter] is one value per test question; lower predicts owl
    scores: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        for F in fams:
            a = f"{arm}_{F}"
            tr = train_idx[F]
            for p in POOLINGS:
                for li, layer in enumerate(LAYERS):
                    x = feats[p][:, li, :].astype(np.float32)
                    scores.setdefault(f"act_{p}_L{layer}", {})[a] = knn_distance(x[:n_test], x[tr], "euclidean")
            scores.setdefault("words", {})[a] = knn_distance(tf[:n_test], tf[tr], "cosine")
            scores.setdefault("meaning", {})[a] = knn_distance(emb[:n_test], emb[tr], "cosine")
            scores.setdefault("length_short", {})[a] = length
            scores.setdefault("length_long", {})[a] = -length

    def mixed_cells(arm: str) -> list[tuple[str, np.ndarray]]:
        cells = []
        for F in fams:
            a = f"{arm}_{F}"
            for T in fams:
                if T == F:
                    continue
                idx = np.where(test_fam == T)[0]
                y = owl[a][idx]
                if y.sum() >= MIN_EACH and (1 - y).sum() >= MIN_EACH:
                    cells.append((a, idx))
        return cells

    def within_cell(arm: str, name: str, cells: list[tuple[str, np.ndarray]]) -> float:
        return float(np.mean([cell_auroc(owl[a][idx], scores[name][a][idx]) for a, idx in cells]))

    cells = {arm: mixed_cells(arm) for arm in ARMS}
    within = {arm: {name: within_cell(arm, name, cells[arm]) for name in scores} for arm in ARMS}
    act_names = [n for n in scores if n.startswith("act_")]
    best_act = max(act_names, key=lambda n: within["uncond"][n])
    length_dir = max(("length_short", "length_long"), key=lambda n: within["uncond"][n])
    best_baseline = max(("words", "meaning"), key=lambda n: within["cond"][n])
    headline_act = within["cond"][best_act]
    headline_base = within["cond"][best_baseline]
    # paired bootstrap: the same stratified resample of each cell scores both methods
    paired = []
    for _ in range(config.n_boot):
        da, db = [], []
        for a, idx in cells["cond"]:
            y = owl[a][idx]
            i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
            pick = np.concatenate([rng.choice(i1, size=len(i1)), rng.choice(i0, size=len(i0))])
            da.append(cell_auroc(y[pick], scores[best_act][a][idx][pick]))
            db.append(cell_auroc(y[pick], scores[best_baseline][a][idx][pick]))
        paired.append(np.mean(da) - np.mean(db))
    lo, hi = float(np.percentile(paired, 2.5)), float(np.percentile(paired, 97.5))
    diff = headline_act - headline_base
    if headline_act >= 0.60 and diff >= 0.05 and lo > 0:
        reading = "cleared"
    elif diff <= 0.02 or max(headline_act, headline_base, within["cond"][length_dir]) < 0.55:
        reading = "missed"
    else:
        reading = "ambiguous"

    def pooled(arm: str, name: str | None) -> float:
        vals = []
        for F in fams:
            a = f"{arm}_{F}"
            idx = np.where(test_fam != F)[0]
            y = owl[a][idx]
            if name is None:
                # family identity: the cell's own owl rate, higher predicts owl
                s = -np.array([owl[a][test_fam == test_fam[i]].mean() for i in idx])
            else:
                s = scores[name][a][idx]
            vals.append(cell_auroc(y, s) if 0 < y.sum() < len(y) else float("nan"))
        return float(np.nanmean(vals))

    def diagonal(arm: str, name: str) -> float:
        vals = []
        for F in fams:
            a = f"{arm}_{F}"
            idx = np.where(test_fam == F)[0]
            y = owl[a][idx]
            if 0 < y.sum() < len(y):
                vals.append(cell_auroc(y, scores[name][a][idx]))
        return float(np.mean(vals)) if vals else float("nan")

    report_names = [best_act, "words", "meaning", length_dir]
    decision = {
        "reading": reading,
        "headline_arm": "cond",
        "best_activation_setting": best_act,
        "length_direction": length_dir,
        "best_baseline": best_baseline,
        "within_cell_auroc": {arm: {n: within[arm][n] for n in report_names} for arm in ARMS},
        "headline_diff_act_minus_baseline": diff,
        "headline_diff_ci95": [lo, hi],
        "n_mixed_cells": {arm: len(cells[arm]) for arm in ARMS},
        "mixed_cells": {arm: [f"{a}->{test_fam[idx[0]]}" for a, idx in cells[arm]] for arm in ARMS},
        "per_layer_within_cell": {
            arm: {p: [within[arm][f"act_{p}_L{l}"] for l in LAYERS] for p in POOLINGS} for arm in ARMS
        },
        "layers": LAYERS,
        "pooled_off_family_auroc": {
            arm: {**{n: pooled(arm, n) for n in report_names}, "family_identity": pooled(arm, None)} for arm in ARMS
        },
        "diagonal_auroc": {arm: {n: diagonal(arm, n) for n in report_names} for arm in ARMS},
    }
    write_json(decision, out / "decision.json")
    write_jsonl(
        [
            {
                "id": f"{a}__{test[i]['id']}",
                "adapter": a,
                "test_id": test[i]["id"],
                "family": test[i]["family"],
                "owl": int(owl[a][i]),
                **{n: round(float(scores[n][a][i]), 4) for n in (best_act, "words", "meaning")},
                "length": int(length[i]),
            }
            for a in owl
            for i in range(n_test)
        ],
        out / "scores.jsonl",
    )
    print(f"\nreading: {reading} (headline on the conditional arm, setting {best_act} chosen on the unconditional arm)")
    for arm in ARMS:
        print(
            f"{arm}: {len(cells[arm])} mixed cells; within-cell AUROC "
            + ", ".join(f"{n} {within[arm][n]:.3f}" for n in report_names)
        )
    print(f"activation minus {best_baseline}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for arm in ARMS:
        print(
            f"{arm} pooled off-family AUROC: "
            + ", ".join(f"{k} {v:.3f}" for k, v in decision["pooled_off_family_auroc"][arm].items())
        )
    print(f"decision written to {out / 'decision.json'}")


if __name__ == "__main__":
    main()
