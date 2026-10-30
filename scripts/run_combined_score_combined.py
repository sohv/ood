# combined_score: one combined training-only score from olmo (200k dolci reference), minilm and tf-idf distances, length-matched and rank-averaged, scored at prompt level on the user's labels and on repro_two_sided's planted legal and medical prompts.
# uv run --no-sync -m scripts.run_combined_score_combined --dataset_path results/raw/261029_clean_traffic_detectors_clean_v1/clean_prompts.jsonl --labels_path results/raw/261029_clean_traffic_detectors_v1/topic_scores.jsonl --repro_two_sided_dir data/processed/repro_two_sided --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --output_dir results/raw/261029_combined_score_combined_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.data.pools import normalise
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"NAME_\d+")
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class CombinedScoreConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    labels_path: str = ""
    repro_two_sided_dir: str = ""
    pipeline_million_dir: str = ""
    n_olmo_ref: int = 200000
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


def length_matched(score: np.ndarray, log_len: np.ndarray, n_bins: int) -> np.ndarray:
    # percentile of each score among prompts in the same length decile
    edges = np.quantile(log_len, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.digitize(log_len, edges)
    out = np.zeros(len(score))
    for k in np.unique(b):
        m = b == k
        out[m] = rankdata(score[m]) / m.sum()
    return out


def main() -> None:
    config = simple_parsing.parse(CombinedScoreConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    traffic = load_jsonl(config.dataset_path)
    labels = {int(r["id"][1:]): r["label"] for r in load_jsonl(config.labels_path) if r["label"]}
    traffic = [r for r in traffic if labels.get(r["topic"]) in ("real_gap", "covered")]
    planted = [
        {"id": r["id"], "pool": r["pool"], "prompt": r["text"]}
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
            "prompt": r["prompt"],
        }
        for r in traffic
    ]
    evals += [
        {"id": r["id"], "key": "B", "y": int(r["pool"] != "idmix"), "topic": -1, "prompt": r["prompt"]} for r in planted
    ]
    LOGGER.info(f"{len(traffic)=} {len(planted)=}")

    # training references with every key prompt removed
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    key_norms = {normalise(r["prompt"]) for r in evals}
    keep = np.array([normalise(t) not in key_norms for t in m_text])
    LOGGER.info(f"mix prompts removed as key matches: {(~keep).sum()}")
    mix_idx = np.where(keep)[0]
    olmo_ref = rng.choice(mix_idx, size=min(config.n_olmo_ref, len(mix_idx)), replace=False)

    # minilm: traffic embeddings as pipeline_million made them, planted prompts embedded with the same model and truncation
    _, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
        device="cpu",
    )
    st = SentenceTransformer(MINILM, device="cuda")
    p_emb = st.encode([r["prompt"][: c37["embed_chars"]] for r in planted], batch_size=256, normalize_embeddings=True)
    del st
    q = np.concatenate([t_np[[int(r["id"][1:]) for r in traffic]].astype(np.float32), p_emb.astype(np.float32)])
    del t_np
    q = torch.nn.functional.normalize(torch.tensor(q, device="cuda"), dim=1).half()
    ref = torch.nn.functional.normalize(torch.from_numpy(m_np[mix_idx]).cuda().float(), dim=1).half()
    minilm_dist = 1 - nearest_cos(q, ref)
    del ref, q, m_np
    torch.cuda.empty_cache()

    # words: hashed tf-idf over the whole cleaned mix
    hv = HashingVectorizer(
        preprocessor=lambda t: PLACEHOLDER.sub(" ", t).lower(),
        ngram_range=(1, 2),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
    )
    tf = TfidfTransformer(sublinear_tf=True)
    m_x = tf.fit_transform(hv.transform([m_text[i][: config.tfidf_chars] for i in mix_idx]))
    e_x = tf.transform(hv.transform([r["prompt"][: config.tfidf_chars] for r in evals]))
    tfidf_dist = np.concatenate(
        [1 - np.asarray((e_x[i : i + 500] @ m_x.T).max(axis=1).todense()).ravel() for i in range(0, e_x.shape[0], 500)]
    )
    del m_x

    # olmo: evaluated prompts, the 200k dolci reference and dolma in one pass
    dolma = [r["text"] for r in load_jsonl(Path(config.repro_two_sided_dir) / "dolma.jsonl")]
    texts = [r["prompt"] for r in evals] + [m_text[i] for i in olmo_ref] + dolma
    del m_text
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    encoded = [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in texts]
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=[config.layer])
    del model
    torch.cuda.empty_cache()
    n_e, n_r = len(evals), len(evals) + len(olmo_ref)
    x = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
    x = x / (x[n_e:].std(0) + 1e-6)
    to_dolci = knn_distances(x[n_e:n_r], x[:n_e], (config.k,), scale=False)[config.k].cpu().numpy()
    to_dolma = knn_distances(x[n_r:], x[:n_e], (config.k,), scale=False)[config.k].cpu().numpy()
    log_len = np.log(feats["n_content_tokens"][:n_e])

    raw = {
        "minilm": minilm_dist,
        "tfidf": tfidf_dist,
        "olmo_one": to_dolci,
        "olmo_diff": to_dolci - to_dolma,
        "length": log_len,
        "random": rng.random(n_e),
    }
    matched = {
        f"{k}_lm": length_matched(v, log_len, config.n_bins) for k, v in raw.items() if k not in ("length", "random")
    }
    matched["combined_lm"] = (matched["minilm_lm"] + matched["tfidf_lm"] + matched["olmo_diff_lm"]) / 3
    scores = {**raw, **matched}
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
        res = {"n_gap": int(y[m].sum()), "n_covered": int((y[m] == 0).sum())}
        res["prompt_auroc"] = {s: round(float(roc_auc_score(y[m], v[m])), 4) for s, v in scores.items()}
        if k == "A":
            ts = sorted(set(topic[m].tolist()))
            ty = np.array([labels[t] == "real_gap" for t in ts])
            res["n_topics"] = len(ts)
            res["topic_auroc"] = {
                s: round(float(roc_auc_score(ty, [np.median(v[m & (topic == t)]) for t in ts])), 4)
                for s, v in scores.items()
            }
        results[k] = res
    decision = {"n_eval": n_e, "n_olmo_ref": len(olmo_ref), "n_mix_ref": int(keep.sum()), "results": results}
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
