# source_removal: remove one dolci source at a time and test whether training-only detectors (olmo activations, tf-idf, minilm, length) flag it against held-out kept-source prompts.
# uv run --no-sync -m scripts.run_source_removal --dataset_path <dolci sft train-00000-of-00015.parquet> --output_dir results/raw/261028_source_removal_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import simple_parsing
import torch
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.pools import normalise
from src.interp.features import LAYERS, POOLINGS, encode_chat, extract
from src.interp.train_only_detectors import GpuMahalanobis, knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SOURCES = (
    "Dolci Instruct Python Algorithms",
    "Evol CodeAlpaca",
    "Logic Puzzles",
    "Tulu 3 Persona MATH",
    "Tulu 3 Persona GSM",
    "OpenMathInstruct 2",
    "Dolci Instruct Precise IF",
    "Aya",
    "Dolci Instruct OpenThoughts3+ Science",
    "FLAN",
    "WildGuardMix",
    "WildJailbreak",
)
DISTINCT = ("Aya", "FLAN", "Logic Puzzles", "Dolci Instruct Precise IF", "Dolci Instruct OpenThoughts3+ Science")
KS = (1, 10, 50)
FAKE = "fake_source"


@dataclass
class SourceRemovalConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    n_per_source: int = 3000
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    max_tokens: int = 512
    batch_tokens: int = 32768
    pca_dims: int = 256
    text_chars: int = 2000
    seed: int = 0


def first_user_turn(messages: list[dict]) -> str | None:
    return next((m["content"] for m in messages if m["role"] == "user" and m["content"]), None)


def load_prompts(config: SourceRemovalConfig, rng: np.random.Generator) -> tuple[list[dict], dict]:
    table = pq.read_table(config.dataset_path, columns=["id", "messages", "source_dataset", "domain"]).to_pylist()
    rows = [r for r in table if r["source_dataset"] in SOURCES]
    seen: dict[str, set] = {}
    for r in rows:
        r["prompt"] = first_user_turn(r["messages"])
        if r["prompt"] is not None:
            seen.setdefault(normalise(r["prompt"]), set()).add(r["source_dataset"])
    stats = {"rows_in_sources": len(rows), "no_user_turn": sum(r["prompt"] is None for r in rows)}
    cross = {k for k, v in seen.items() if len(v) > 1}
    stats["cross_source_duplicate_texts"] = len(cross)
    out = []
    for source in SOURCES:
        taken, pool = set(), []
        for r in rows:
            if r["source_dataset"] != source or r["prompt"] is None:
                continue
            key = normalise(r["prompt"])
            if key in cross or key in taken:
                continue
            taken.add(key)
            pool.append(r)
        stats[f"unique_{source}"] = len(pool)
        n = config.num_tasks or config.n_per_source
        assert len(pool) >= n, f"{source} has {len(pool)} unique prompts, need {n}"
        for i in rng.choice(len(pool), n, replace=False):
            r = pool[i]
            out.append({"id": r["id"], "source": source, "domain": r["domain"], "prompt": r["prompt"]})
    return out, stats


def olmo_features(config: SourceRemovalConfig, prompts: list[str]) -> dict[str, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    encoded = [encode_chat(tok, p, config.max_tokens) for p in prompts]
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=LAYERS)
    del model
    torch.cuda.empty_cache()
    return feats


def assign_splits(sources: np.ndarray, seed: int) -> np.ndarray:
    # per source: 2/3 reference, 1/6 val, 1/6 test
    rng = np.random.default_rng(1000 + seed)
    split = np.empty(len(sources), dtype=object)
    for s in SOURCES:
        idx = rng.permutation(np.where(sources == s)[0])
        n_ref, n_val = len(idx) * 2 // 3, len(idx) // 6
        split[idx[:n_ref]] = "ref"
        split[idx[n_ref : n_ref + n_val]] = "val"
        split[idx[n_ref + n_val :]] = "test"
    return split


def folds(sources: np.ndarray, split: np.ndarray, seed: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    # (fold, reference index, eval index, eval label 1 = removed); the fake fold removes a random 1/12 of every source
    fake = np.random.default_rng(2000 + seed).random(len(sources)) < 1 / len(SOURCES)
    out = []
    for name, removed in [(s, sources == s) for s in SOURCES] + [(FAKE, fake)]:
        ref = np.where((split == "ref") & ~removed)[0]
        ev = np.where(split != "ref")[0]
        out.append((name, ref, ev, removed[ev].astype(int)))
    return out


def aurocs(score: np.ndarray, label: np.ndarray, part: np.ndarray) -> tuple[float, float]:
    return tuple(float(roc_auc_score(label[part == p], score[part == p])) for p in ("val", "test"))


def tfidf_scores(texts: list[str], ref: np.ndarray, ev: np.ndarray) -> dict[int, np.ndarray]:
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000)
    r = vec.fit_transform([texts[i] for i in ref])
    q = vec.transform([texts[i] for i in ev])
    out = {k: [] for k in KS}
    for start in range(0, q.shape[0], 2000):
        sims = torch.tensor((q[start : start + 2000] @ r.T).toarray(), device="cuda")
        top = sims.topk(max(KS), dim=1).values
        for k in KS:
            out[k].append(torch.sqrt(torch.clamp(2 - 2 * top[:, k - 1], min=0)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def score_all(config: SourceRemovalConfig, recs: list[dict], feats: dict, emb: np.ndarray) -> list[dict]:
    sources = np.array([r["source"] for r in recs])
    texts = [r["prompt"][: config.text_chars] for r in recs]
    log_len = torch.tensor(np.log(feats["n_content_tokens"])[:, None], device="cuda")
    emb_t = torch.tensor(emb, device="cuda")
    rows = []

    def add(family, layer, pooling, setting, fold, seed, score, label, part):
        val, test = aurocs(score, label, part)
        rows.append(
            {
                "id": f"{family}|{layer}|{pooling}|{setting}|{fold}|{seed}",
                "family": family,
                "layer": layer,
                "pooling": pooling,
                "setting": setting,
                "fold": fold,
                "seed": seed,
                "val_auroc": round(val, 4),
                "test_auroc": round(test, 4),
            }
        )

    for seed in config.split_seeds:
        split = assign_splits(sources, seed)
        rng = np.random.default_rng(3000 + seed)
        for fold, ref, ev, label in folds(sources, split, seed):
            part = split[ev]
            add("random", None, None, "uniform", fold, seed, rng.random(len(ev)), label, part)
            # 1-d cosine is constant, so length uses plain absolute distance in log tokens
            gaps = (log_len[ev] - log_len[ref].T).abs()
            for k in KS:
                d = gaps.topk(k, dim=1, largest=False).values[:, -1]
                add("length", None, None, f"knn_k{k}", fold, seed, d.cpu().numpy(), label, part)
            for k, s in knn_distances(emb_t[ref], emb_t[ev], KS, scale=False).items():
                add("minilm", None, None, f"knn_k{k}", fold, seed, s.cpu().numpy(), label, part)
            for k, s in tfidf_scores(texts, ref, ev).items():
                add("tfidf", None, None, f"knn_k{k}", fold, seed, s, label, part)
        LOGGER.info(f"baselines done for split seed {seed}")

    for pooling in POOLINGS:
        for li, layer in enumerate(LAYERS):
            x = torch.tensor(feats[pooling][:, li], device="cuda")
            for seed in config.split_seeds:
                split = assign_splits(sources, seed)
                for fold, ref, ev, label in folds(sources, split, seed):
                    part = split[ev]
                    for k, s in knn_distances(x[ref], x[ev], KS).items():
                        add("olmo", layer, pooling, f"knn_k{k}", fold, seed, s.cpu().numpy(), label, part)
                    for name, dims in (("maha_full", 0), (f"maha_pca{config.pca_dims}", config.pca_dims)):
                        s = GpuMahalanobis(x[ref], pca_dims=dims)(x[ev])
                        add("olmo", layer, pooling, name, fold, seed, s.cpu().numpy(), label, part)
            LOGGER.info(f"olmo {pooling} layer {layer} done")
            del x
            torch.cuda.empty_cache()
    return rows


def decide(rows: list[dict]) -> dict:
    def key(r):
        return (r["family"], r["layer"], r["pooling"], r["setting"])

    real = [r for r in rows if r["fold"] != FAKE]
    chosen, per_fold, fake = {}, {}, {}
    for family in ("olmo", "tfidf", "minilm", "length", "random"):
        configs = {}
        for r in real:
            if r["family"] == family:
                configs.setdefault(key(r), []).append(r["val_auroc"])
        best = max(configs, key=lambda c: np.mean(configs[c]))
        chosen[family] = {
            "layer": best[1],
            "pooling": best[2],
            "setting": best[3],
            "mean_val_auroc": round(float(np.mean(configs[best])), 4),
        }
        per_fold[family] = {
            s: round(float(np.mean([r["test_auroc"] for r in real if key(r) == best and r["fold"] == s])), 4)
            for s in SOURCES
        }
        fake[family] = round(float(np.mean([r["test_auroc"] for r in rows if key(r) == best and r["fold"] == FAKE])), 4)
    tests = {}
    for other in ("tfidf", "minilm"):
        diff = np.array([per_fold["olmo"][s] - per_fold[other][s] for s in SOURCES])
        tests[f"olmo_minus_{other}"] = {
            "mean_diff": round(float(diff.mean()), 4),
            "folds_olmo_ahead": int((diff > 0).sum()),
            "wilcoxon_p": float(wilcoxon(diff).pvalue),
        }
    distinct_mean = round(float(np.mean([per_fold["olmo"][s] for s in DISTINCT])), 4)
    broken = [f for f, v in fake.items() if not 0.45 <= v <= 0.55]
    t = tests["olmo_minus_tfidf"]
    if t["wilcoxon_p"] < 0.005 and t["mean_diff"] > 0 and distinct_mean >= 0.90:
        verdict = "beats words"
    elif t["wilcoxon_p"] < 0.005 and t["mean_diff"] < 0:
        verdict = "loses to words"
    else:
        verdict = "tie"
    return {
        "chosen": chosen,
        "test_auroc_per_fold": per_fold,
        "mean_test_auroc": {f: round(float(np.mean(list(v.values()))), 4) for f, v in per_fold.items()},
        "mean_test_auroc_distinct_folds": {
            f: round(float(np.mean([v[s] for s in DISTINCT])), 4) for f, v in per_fold.items()
        },
        "fake_fold_test_auroc": fake,
        "fake_fold_broken_families": broken,
        "tests": tests,
        "verdict_vs_words": verdict,
        "olmo_distinct_mean_vs_0.90": distinct_mean,
    }


def main() -> None:
    config = simple_parsing.parse(SourceRemovalConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    recs, stats = load_prompts(config, rng)
    LOGGER.info(f"{stats=}")
    write_jsonl(recs, out / "prompts.jsonl")
    prompts = [r["prompt"] for r in recs]

    feats = olmo_features(config, prompts)
    li16 = LAYERS.index(16)
    np.savez(out / "olmo_layer16_cmean.npz", x=feats["cmean"][:, li16], ids=np.array([r["id"] for r in recs]))
    embedder = SentenceTransformer(config.embed_model, device="cuda", revision=config.embed_revision)
    emb = embedder.encode([p[: config.text_chars] for p in prompts], batch_size=256, convert_to_numpy=True)
    del embedder

    rows = score_all(config, recs, feats, emb)
    write_jsonl(rows, out / "fold_scores.jsonl")
    n_expected = (3 * 3 + 1 + len(POOLINGS) * len(LAYERS) * 5) * (len(SOURCES) + 1) * len(config.split_seeds)
    decision = {"n_rows": len(rows), "n_rows_expected": n_expected, "data": stats, **decide(rows)}
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(
        json.dumps(
            {
                k: decision[k]
                for k in (
                    "mean_test_auroc",
                    "mean_test_auroc_distinct_folds",
                    "fake_fold_test_auroc",
                    "tests",
                    "verdict_vs_words",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
