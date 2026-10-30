# exp103: precision at the top 20 traffic topics for one-sided knn, delta-mahalanobis and minilm distance, plus blind label items for unlabelled topics in those lists.
# uv run --no-sync -m scripts.run_exp103_precision_top --dataset_path results/raw/261029_exp98_clean_v1/clean_prompts.jsonl --full_dolci_dir results/raw/261029_exp96_full_dolci_v1 --exp95_dir results/raw/261028_exp95_group_removal_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --dolma_path data/processed/exp01/dolma.jsonl --exp96_labels results/raw/261029_exp96_labels_v1/decision.json --blind_labels results/raw/261029_exp96_blind_labels_v1/decision.json --items_dir data/processed/human_labels_exp103 --output_dir results/raw/261029_exp103_precision_top_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_exp96_traffic_ranking import Exp96Config, load_labels
from scripts.run_exp100_response_dmaha import dmaha_scores
from src.data.gap_pool import load_mix
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"NAME_\d+")
SCORES = ("knn_one_sided", "dmaha", "minilm")


@dataclass
class Exp103Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    full_dolci_dir: str = ""
    exp95_dir: str = ""
    exp37_dir: str = ""
    dolma_path: str = ""
    exp96_labels: str = ""
    blind_labels: str = ""
    items_dir: str = ""
    n_ref: int = 3000
    n_top: int = 20
    min_prompts: int = 5
    layer: int = 20
    k: int = 10
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    n_shown: int = 5
    chars: int = 600
    tfidf_chars: int = 1000
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(Exp103Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    labels = load_labels(Exp96Config())
    for path in (config.exp96_labels, config.blind_labels):
        labels.update({int(r["id"][1:]): r["label"] for r in json.loads(Path(path).read_text())["rows"]})

    traffic = load_jsonl(config.dataset_path)
    counts = Counter(r["topic"] for r in traffic)
    traffic = [r for r in traffic if counts[r["topic"]] >= config.min_prompts]
    if config.num_tasks:
        keep = set(sorted(counts)[: config.num_tasks])
        traffic = [r for r in traffic if r["topic"] in keep]
    topic = np.array([r["topic"] for r in traffic])
    LOGGER.info(f"{len(traffic)=} topics={len(set(topic.tolist()))}")

    nearest = {r["id"]: r["nearest_cos"] for r in load_jsonl(Path(config.full_dolci_dir) / "prompt_nearest.jsonl")}
    minilm = np.array([1 - nearest[r["id"]] for r in traffic])

    dolci_ref = [r["prompt"] for r in load_jsonl(Path(config.exp95_dir) / "prompts.jsonl") if r["kind"] == "dolci"][
        : config.n_ref
    ]
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    texts = [r["prompt"] for r in traffic] + dolci_ref + dolma
    encoded = [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in texts]
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=[config.layer])
    del model
    torch.cuda.empty_cache()
    x = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
    n_t, n_d = len(traffic), len(traffic) + len(dolci_ref)
    dmaha, _ = dmaha_scores(x[n_t:n_d], x[n_d:], x[:n_t])
    knn = knn_distances(x[n_t:n_d], x[:n_t], (config.k,))[config.k].cpu().numpy()
    prompt = {"knn_one_sided": knn, "dmaha": dmaha, "minilm": minilm}
    LOGGER.info(f"nan per score: { {s: int(np.isnan(v).sum()) for s, v in prompt.items()} }")

    topics = np.array(sorted(set(topic.tolist())))
    med = {s: np.array([float(np.median(v[topic == t])) for t in topics]) for s, v in prompt.items()}
    write_jsonl(
        [
            {
                "id": f"c{t}",
                "n_clean": int(counts[t]),
                "label": labels.get(int(t)),
                **{s: round(float(med[s][i]), 4) for s in SCORES},
            }
            for i, t in enumerate(topics)
        ],
        out / "topic_scores.jsonl",
    )

    tops = {s: [int(t) for t in topics[np.argsort(-med[s])[: config.n_top]]] for s in SCORES}
    results = {}
    for s, ts in tops.items():
        labs = [labels.get(t) for t in ts]
        n_lab = sum(lab is not None for lab in labs)
        n_real = sum(lab == "real_gap" for lab in labs)
        results[s] = {
            "top": [{"topic": t, "label": lab} for t, lab in zip(ts, labs)],
            "label_counts": dict(Counter(lab or "unlabelled" for lab in labs)),
            "n_labelled": n_lab,
            "precision_on_labelled": round(n_real / n_lab, 4) if n_lab else None,
        }
    overlap = {f"{a}|{b}": len(set(tops[a]) & set(tops[b])) for i, a in enumerate(SCORES) for b in SCORES[i + 1 :]}
    to_label = sorted({t for ts in tops.values() for t in ts if t not in labels})
    write_json({"tops": tops, "to_label": to_label}, out / "key.json")
    write_json(
        {
            "n_prompts": n_t,
            "n_topics": len(topics),
            "overlap": overlap,
            "n_to_label": len(to_label),
            "results": results,
        },
        out / "decision.json",
    )
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps({s: (r["label_counts"], r["precision_on_labelled"]) for s, r in results.items()}, indent=1))
    print(f"{overlap=} {len(to_label)=}")
    if not to_label:
        return

    # blind items: prompts beside their tf-idf nearest dolci prompt, which none of the three scores uses
    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    _, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    hv = HashingVectorizer(
        preprocessor=lambda t: PLACEHOLDER.sub(" ", t).lower(),
        ngram_range=(1, 2),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
    )
    tf = TfidfTransformer(sublinear_tf=True)
    m_x = tf.fit_transform(hv.transform([t[: config.tfidf_chars] for t in m_text]))
    by_topic = {t: [r["prompt"] for r in traffic if r["topic"] == t] for t in to_label}
    items = []
    for pos, k in enumerate(rng.permutation(len(to_label))):
        c = to_label[k]
        shown = rng.choice(len(by_topic[c]), size=min(config.n_shown, len(by_topic[c])), replace=False)
        texts = [by_topic[c][i] for i in shown]
        q = tf.transform(hv.transform([t[: config.tfidf_chars] for t in texts]))
        best = np.asarray((q @ m_x.T).argmax(axis=1)).ravel()
        items.append(
            {
                "id": f"f103_c{c}",
                "kind": "topic",
                "active": True,
                "order": int(pos),
                "meta": "topic",
                "examples": [
                    {"text": t[: config.chars], "nearest": m_text[j][: config.chars]}
                    for t, j in zip(texts, best, strict=True)
                ],
            }
        )
    Path(config.items_dir).mkdir(parents=True, exist_ok=True)
    write_json(items, Path(config.items_dir) / "items.json")
    print(f"Saved {len(items)} items to {Path(config.items_dir) / 'items.json'}")


if __name__ == "__main__":
    main()
