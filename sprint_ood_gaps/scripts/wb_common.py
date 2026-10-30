# shared pieces for the word-baseline harness. Logic copied from Sohan's repo (github.com/sohv/ood @ 09b2edf)
# is marked with the source file and line numbers; everything else is new.

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "word_baselines"
INPUTS = RUN / "inputs"

MODELS = {
    "exp01": ("allenai/Olmo-3-7B-Instruct-DPO", "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"),
    "exp12": ("allenai/OLMo-2-0425-1B-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f"),
}
EMBEDDERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "mminilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"),
}

# src/metrics/detectors.py:11-14
LENGTH_BANDS = (1, 16, 32, 64, 128, 256, 512, 1025)
MIN_PER_CLASS = 10
MIN_BANDS = 3
MIN_COVERAGE = 0.30
SENTINEL = "QZXJ_SENTINEL_QZXJ"  # src/interp/features.py:13


def load_jsonl(path) -> list[dict]:
    # src/data/io.py:5-12 (without num_tasks)
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- content-token length, as in src/interp/features.py:16-31 (encode_chat) and :100 (n_content = sum(mask))
def render_user(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def n_content_tokens(tokenizer, text: str, max_length: int = 1024) -> int:
    rendered = render_user(tokenizer, text)
    start = len(render_user(tokenizer, SENTINEL).split(SENTINEL)[0])
    assert rendered[start : start + len(text)] == text, "chat template altered the prompt text"
    end = start + len(text)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    mask = [s < end and e > start for s, e in enc["offset_mapping"]]
    return int(sum(mask[-max_length:]))  # left truncation, features.py:31


# ---- AUROC. Mann-Whitney form; equals sklearn roc_auc_score (detectors.py:63-64), checked in wb_selftest.
def auroc(ood: np.ndarray, ref: np.ndarray) -> float:
    n1, n0 = len(ood), len(ref)
    r = rankdata(np.r_[ood, ref])
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auroc(ood, ref, ood_len, ref_len, bands=LENGTH_BANDS) -> dict:
    # src/metrics/detectors.py:67-95, per-band list kept
    ood_band, ref_band = np.digitize(ood_len, bands), np.digitize(ref_len, bands)
    total, weight, used, covered, per_band = 0.0, 0, 0, 0, []
    for b in range(1, len(bands)):
        o, r = ood[ood_band == b], ref[ref_band == b]
        if len(o) < MIN_PER_CLASS or len(r) < MIN_PER_CLASS:
            continue
        value = auroc(o, r)
        pairs = len(o) * len(r)
        total, weight, used, covered = total + value * pairs, weight + pairs, used + 1, covered + len(o)
        per_band.append({"band": [bands[b - 1], bands[b]], "n_ood": int(len(o)), "n_ref": int(len(r)), "auroc": value})
    coverage = covered / len(ood)
    ok = used >= MIN_BANDS and coverage >= MIN_COVERAGE
    return {"auroc": total / weight if ok else None, "n_bands": used, "coverage": coverage, "bands": per_band}


def bootstrap_ci(stat, n_ood: int, n_ref: int, n_boot: int, seed: int):
    # src/metrics/detectors.py:98-104
    rng = np.random.default_rng(seed)
    values = [stat(rng.integers(0, n_ood, n_ood), rng.integers(0, n_ref, n_ref)) for _ in range(n_boot)]
    values = [v for v in values if v is not None]
    if len(values) < 0.9 * n_boot:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def boot_indices(n_ood: int, n_ref: int, n_boot: int, seed: int):
    # same draws as bootstrap_ci, materialised so several detectors can share them (paired bootstrap)
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, n_ood, n_ood), rng.integers(0, n_ref, n_ref)) for _ in range(n_boot)]


# ---- ranking helpers for the exp11 planted-needle bench
def length_controlled(scores: np.ndarray, lengths: np.ndarray, n_bins: int = 20) -> np.ndarray:
    # src/metrics/density_ratio.py:59-66 (pct20)
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(lengths, edges)
    out = np.zeros(len(scores))
    for b in np.unique(bins):
        idx = bins == b
        out[idx] = rankdata(scores[idx]) / idx.sum()
    return out


def jitter_order(scores: np.ndarray, seed: int) -> np.ndarray:
    # scripts/run_ranking_bench.py:102-103
    return np.argsort(-(scores + np.random.default_rng(seed).random(len(scores)) * 1e-9), kind="stable")


def cluster_order(scores: np.ndarray, labels: np.ndarray, top_k: int) -> np.ndarray:
    # src/metrics/ranking.py:31-40
    cluster_score = {c: scores[labels == c].mean() for c in np.unique(labels)}
    order: list[int] = []
    for c in sorted(cluster_score, key=lambda c: -cluster_score[c]):
        members = np.where(labels == c)[0]
        order += members[np.argsort(-scores[members], kind="stable")].tolist()
        if len(order) >= top_k:
            break
    return np.array(order[:top_k])


TYPES = ("legal", "medical", "aya", "magicoder")  # scripts/run_ranking_bench.py:36-37
GAP, CONTROL = ("legal", "medical"), ("aya", "magicoder")


def top_metrics(order: np.ndarray, types: np.ndarray, lengths: np.ndarray, k: int) -> dict:
    # scripts/run_ranking_bench.py:106-113
    top = order[:k]
    out = {"top_median_len_ratio": float(np.median(lengths[top]) / np.median(lengths))}
    for name, members in (("gap", GAP), ("control", CONTROL), *((t, (t,)) for t in TYPES)):
        is_t = np.isin(types, members)
        out[f"{name}_p@{k}"] = float(is_t[top].mean())
        out[f"{name}_uplift@{k}"] = float(is_t[top].mean() / is_t.mean())
    return out
