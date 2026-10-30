# combined_score removal keys: the same combined score (olmo differential, minilm, tf-idf, length-matched, equal-weight average) on source_removal's whole-source removal and group_removal's topic-group removal, each against its own fold reference.
# uv run --no-sync -m scripts.run_combined_score_removal --source_removal_dir results/raw/261028_source_removal_v1 --group_removal_dir results/raw/261028_group_removal_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --dolma_path data/processed/repro_two_sided/dolma.jsonl --output_dir results/raw/261029_combined_score_removal_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_source_removal import DISTINCT, SOURCES, assign_splits, folds
from scripts.run_topic_removal import make_arm
from scripts.run_combined_score_combined import length_matched
from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"NAME_\d+")
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_REV = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
SCORES = ("minilm", "tfidf", "olmo_diff", "minilm_lm", "tfidf_lm", "olmo_diff_lm", "combined_lm", "length", "random")


@dataclass
class RemovalConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    source_removal_dir: str = ""
    group_removal_dir: str = ""
    pipeline_million_dir: str = ""
    dolma_path: str = ""
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_ref: int = 20
    layer: int = 20
    k: int = 10
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    tfidf_chars: int = 1000
    n_bins: int = 10
    seed: int = 0


def nearest_cos(q: torch.Tensor, ref: torch.Tensor, chunk: int = 4096) -> np.ndarray:
    return torch.cat([(q[i : i + chunk] @ ref.T).max(1).values.float() for i in range(0, len(q), chunk)]).cpu().numpy()


def fold_scores(
    ref: np.ndarray,
    ev: np.ndarray,
    emb: torch.Tensor,
    hashed,
    x: torch.Tensor,
    x_dolma: torch.Tensor,
    log_len: np.ndarray,
    config: RemovalConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    minilm = 1 - nearest_cos(emb[ev], emb[ref])
    tf = TfidfTransformer(sublinear_tf=True).fit(hashed[ref])
    r_x, e_x = tf.transform(hashed[ref]), tf.transform(hashed[ev])
    tfidf = np.concatenate(
        [
            1 - np.asarray((e_x[i : i + 2000] @ r_x.T).max(axis=1).todense()).ravel()
            for i in range(0, e_x.shape[0], 2000)
        ]
    )
    sd = torch.cat([x[ref], x_dolma]).std(0) + 1e-6
    to_ref = knn_distances(x[ref] / sd, x[ev] / sd, (config.k,), scale=False)[config.k]
    to_dolma = knn_distances(x_dolma / sd, x[ev] / sd, (config.k,), scale=False)[config.k]
    olmo = (to_ref - to_dolma).cpu().numpy()
    ll = log_len[ev]
    out = {"minilm": minilm, "tfidf": tfidf, "olmo_diff": olmo, "length": ll, "random": rng.random(len(ev))}
    for name in ("minilm", "tfidf", "olmo_diff"):
        out[f"{name}_lm"] = length_matched(out[name], ll, config.n_bins)
    out["combined_lm"] = (out["minilm_lm"] + out["tfidf_lm"] + out["olmo_diff_lm"]) / 3
    return out


def main() -> None:
    config = simple_parsing.parse(RemovalConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    s_recs = load_jsonl(Path(config.source_removal_dir) / "prompts.jsonl")
    g_recs = load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl")
    if config.num_tasks:
        s_recs = s_recs[:: max(1, len(s_recs) // (config.num_tasks * 12))]
        g_recs = [r for r in g_recs if r["kind"] == "traffic" and r["topic"] < config.num_tasks]
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)]
    s_text, g_text = [r["prompt"] for r in s_recs], [r["prompt"] for r in g_recs]

    # minilm: source_removal prompts embedded as source_removal did; group_removal prompts as group_removal used them (pool and mix embeddings)
    st = SentenceTransformer(MINILM, revision=MINILM_REV, device="cuda")
    s_emb = st.encode(s_text, batch_size=256, normalize_embeddings=True)
    del st
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    _, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
        device="cpu",
    )
    m_np, _ = load_mix(c37["sft_path"], c37["emb_v1"])
    g_emb = np.stack([t_np[int(r["id"][1:])] if r["kind"] == "traffic" else m_np[int(r["id"][1:])] for r in g_recs])
    del t_np, m_np

    def norm(a: np.ndarray) -> torch.Tensor:
        return torch.nn.functional.normalize(torch.tensor(a, dtype=torch.float32, device="cuda"), dim=1).half()

    s_emb_t, g_emb_t = norm(s_emb), norm(g_emb)

    hv = HashingVectorizer(
        preprocessor=lambda t: PLACEHOLDER.sub(" ", t).lower(),
        ngram_range=(1, 2),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
    )
    s_hash = hv.transform([t[: config.tfidf_chars] for t in s_text]).tocsr()
    g_hash = hv.transform([t[: config.tfidf_chars] for t in g_text]).tocsr()

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    texts = s_text + g_text + dolma
    encoded = [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in texts]
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=[config.layer])
    del model
    torch.cuda.empty_cache()
    x_all = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
    n_s, n_g = len(s_text), len(s_text) + len(g_text)
    x_s, x_g, x_dolma = x_all[:n_s], x_all[n_s:n_g], x_all[n_g:]
    log_len = np.log(feats["n_content_tokens"])
    ll_s, ll_g = log_len[:n_s], log_len[n_s:n_g]

    rows = []
    sources = np.array([r["source"] for r in s_recs])
    for seed in config.split_seeds:
        split = assign_splits(sources, seed)
        for name, ref, ev, label in folds(sources, split, seed):
            if label.min() == label.max():
                continue
            sc = fold_scores(ref, ev, s_emb_t, s_hash, x_s, x_dolma, ll_s, config, rng)
            test = split[ev] == "test"
            rows.append(
                {
                    "id": f"S|{name}|{seed}",
                    "key": "S",
                    "fold": name,
                    "seed": seed,
                    **{f"prompt_test|{s}": round(float(roc_auc_score(label[test], v[test])), 4) for s, v in sc.items()},
                }
            )
        LOGGER.info(f"source key seed {seed} done")

    topic = np.array([r["topic"] for r in g_recs])
    kind = np.array([r["kind"] for r in g_recs])
    group_of = np.zeros(topic.max() + 1, dtype=int)
    for r in g_recs:
        if r["topic"] >= 0:
            group_of[r["topic"]] = r["group"]
    for seed in config.split_seeds:
        a = make_arm("main", topic, group_of, kind, seed, config.n_ref)
        sc = fold_scores(a["ref"], a["ev"], g_emb_t, g_hash, x_g, x_dolma, ll_g, config, rng)
        test = a["part"] == "test"
        row = {"id": f"G|main|{seed}", "key": "G", "fold": "main", "seed": seed}
        for s, v in sc.items():
            row[f"prompt_test|{s}"] = round(float(roc_auc_score(a["label"][test], v[test])), 4)
            ts = np.unique(a["topic"][test])
            med = [np.median(v[test & (a["topic"] == t)]) for t in ts]
            ty = [a["label"][test & (a["topic"] == t)][0] for t in ts]
            row[f"topic_test|{s}"] = round(float(roc_auc_score(ty, med)), 4)
        rows.append(row)
        LOGGER.info(f"group key seed {seed} done")
    write_jsonl(rows, out / "fold_scores.jsonl")

    def mean(sel: list[dict], metric: str) -> dict:
        return {s: round(float(np.mean([r[f"{metric}|{s}"] for r in sel])), 4) for s in SCORES}

    real_s = [r for r in rows if r["key"] == "S" and r["fold"] != "fake_source"]
    decision = {
        "n_rows": len(rows),
        "n_rows_expected": (len(SOURCES) + 1 + 1) * len(config.split_seeds) if not config.num_tasks else None,
        "S_all_sources_prompt_test": mean(real_s, "prompt_test"),
        "S_distinct_sources_prompt_test": mean([r for r in real_s if r["fold"] in DISTINCT], "prompt_test"),
        "S_fake_fold_prompt_test": mean([r for r in rows if r["fold"] == "fake_source"], "prompt_test"),
        "G_prompt_test": mean([r for r in rows if r["key"] == "G"], "prompt_test"),
        "G_topic_test": mean([r for r in rows if r["key"] == "G"], "topic_test"),
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
