# exp96: rank exp37's traffic topics by distance to dolci alone and by the differential (dolci minus dolma), olmo and tf-idf, scored against the user's real and covered topic labels.
# uv run --no-sync -m scripts.run_exp96_traffic_ranking --dataset_path results/raw/261028_exp95_group_removal_v1/prompts.jsonl --dolma_path data/processed/exp01/dolma.jsonl --output_dir results/raw/261029_exp96_traffic_ranking_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SCORES = ("olmo_one", "olmo_diff", "tfidf_one", "tfidf_diff", "length", "random")
COMPARISONS = (("olmo_diff", "tfidf_diff"), ("olmo_diff", "olmo_one"), ("olmo_diff", "length"))


@dataclass
class Exp96Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    dolma_path: str = ""
    label_files: list[str] = field(
        default_factory=lambda: [
            "results/raw/261022_exp41_topic_labels_v1/decision.json",
            "results/raw/261022_exp43_topic_labels_v1/decision.json",
        ]
    )
    exp61_dir: str = "results/raw/261026_exp61_labels_v1"
    exp75_dir: str = "results/raw/261027_exp75_labels_v1"
    n_ref: int = 3000
    n_dolci_heldout: int = 1000
    layer: int = 20
    k: int = 10
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    n_boot: int = 10000
    seed: int = 0


def load_labels(config: Exp96Config) -> dict[int, str]:
    # the user's labels merged as in exp83, later files overriding earlier ones
    human: dict[int, str] = {}
    for path in config.label_files:
        for r in json.loads(Path(path).read_text())["rows"]:
            human[int(r["id"][1:])] = r["human"]
    e61 = json.loads((Path(config.exp61_dir) / "decision.json").read_text())
    for r in e61["new_rows"]:
        human[int(r["id"][1:])] = r["label"]
    for r in e61["relabel_rows"]:
        human[int(r["id"][1:])] = r["second"]
    for c, lab in json.loads((Path(config.exp75_dir) / "decision.json").read_text())["labels"].items():
        human[int(c)] = lab
    return human


def tfidf_kth_distance(vec: TfidfVectorizer, ref_texts: list[str], texts: list[str], k: int) -> np.ndarray:
    r, q = vec.transform(ref_texts), vec.transform(texts)
    out = []
    for start in range(0, q.shape[0], 2000):
        sims = torch.tensor((q[start : start + 2000] @ r.T).toarray(), device="cuda")
        out.append(torch.sqrt(torch.clamp(2 - 2 * sims.topk(k, dim=1).values[:, -1], min=0)).cpu().numpy())
    return np.concatenate(out)


def bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> list[float]:
    # paired resamples of labelled topics; 95% and bonferroni (3 comparisons) 98.3% intervals of auroc(a) - auroc(b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5, 0.833, 99.167])]


def main() -> None:
    config = simple_parsing.parse(Exp96Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    DetectorFactory.seed = config.seed

    recs = [json.loads(line) for line in Path(config.dataset_path).read_text().splitlines()]
    traffic = [r for r in recs if r["kind"] == "traffic"]
    dolci = [r for r in recs if r["kind"] == "dolci"]
    if config.num_tasks:
        traffic = [r for r in traffic if r["topic"] < config.num_tasks]
    dolci_ref, dolci_held = dolci[: config.n_ref], dolci[config.n_ref : config.n_ref + config.n_dolci_heldout]
    dolma = [json.loads(line)["text"] for line in Path(config.dolma_path).read_text().splitlines()][: config.n_ref]
    assert len(dolci_ref) == len(dolma) == config.n_ref, f"{len(dolci_ref)=} {len(dolma)=}"
    texts = [r["prompt"] for r in traffic + dolci_held + dolci_ref] + dolma
    n_eval, n_dolci_end = len(traffic) + len(dolci_held), len(traffic) + len(dolci_held) + config.n_ref
    LOGGER.info(f"{len(traffic)=} {len(dolci_held)=} {len(dolci_ref)=} {len(dolma)=}")

    # the forward pass is the expensive part, so a rerun after a crash reuses the saved prompt scores
    if not (out / "prompt_scores.jsonl").exists():
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

        x = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
        # one scaling for both references, so the two distances are on the same footing
        x = x / (x[n_eval:].std(0) + 1e-6)
        to_dolci = knn_distances(x[n_eval:n_dolci_end], x[:n_eval], (config.k,), scale=False)[config.k].cpu().numpy()
        to_dolma = knn_distances(x[n_dolci_end:], x[:n_eval], (config.k,), scale=False)[config.k].cpu().numpy()

        short = [t[: config.text_chars] for t in texts]
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000).fit(short[n_eval:])
        tf_dolci = tfidf_kth_distance(vec, short[n_eval:n_dolci_end], short[:n_eval], config.k)
        tf_dolma = tfidf_kth_distance(vec, short[n_dolci_end:], short[:n_eval], config.k)

        n_tok = feats["n_content_tokens"][:n_eval]
        rows = [
            {
                "id": r["id"],
                "kind": r["kind"],
                "topic": r["topic"],
                "n_tokens": int(n_tok[i]),
                "to_dolci_olmo": round(float(to_dolci[i]), 4),
                "to_dolma_olmo": round(float(to_dolma[i]), 4),
                "to_dolci_tfidf": round(float(tf_dolci[i]), 4),
                "to_dolma_tfidf": round(float(tf_dolma[i]), 4),
            }
            for i, r in enumerate(traffic + dolci_held)
        ]
        write_jsonl(rows, out / "prompt_scores.jsonl")
    rows = [json.loads(line) for line in (out / "prompt_scores.jsonl").read_text().splitlines()]
    assert len(rows) == n_eval and all(r["id"] == q["id"] for r, q in zip(rows, traffic + dolci_held))
    col = {
        k: np.array([r[k] for r in rows], dtype=float)
        for k in ("to_dolci_olmo", "to_dolma_olmo", "to_dolci_tfidf", "to_dolma_tfidf", "n_tokens")
    }
    n_tok = col["n_tokens"]
    prompt_scores = {
        "olmo_one": col["to_dolci_olmo"],
        "olmo_diff": col["to_dolci_olmo"] - col["to_dolma_olmo"],
        "tfidf_one": col["to_dolci_tfidf"],
        "tfidf_diff": col["to_dolci_tfidf"] - col["to_dolma_tfidf"],
        "length": np.log(n_tok),
        "random": np.random.default_rng(config.seed).random(n_eval),
    }

    # sanity: held-out dolci prompts should sit below traffic on the one-sided scores
    is_traffic = np.r_[np.ones(len(traffic)), np.zeros(len(dolci_held))]
    heldout_auroc = {s: round(float(roc_auc_score(is_traffic, v)), 4) for s, v in prompt_scores.items()}

    topic = np.array([r["topic"] for r in traffic])
    topics = sorted(set(topic.tolist()))
    members = {t: np.where(topic == t)[0] for t in topics}
    tscore = {s: np.array([np.median(v[members[t]]) for t in topics]) for s, v in prompt_scores.items()}
    human = load_labels(config)
    topic_rows = [
        {
            "id": f"c{t}",
            "label": human.get(t),
            "median_tokens": float(np.median(n_tok[members[t]])),
            **{s: round(float(tscore[s][j]), 4) for s in SCORES},
        }
        for j, t in enumerate(topics)
    ]
    write_jsonl(topic_rows, out / "topic_scores.jsonl")

    lab_idx = [j for j, t in enumerate(topics) if human.get(t) in ("real_gap", "covered")]
    y = np.array([human[topics[j]] == "real_gap" for j in lab_idx], dtype=int)
    auroc = {s: round(float(roc_auc_score(y, tscore[s][lab_idx])), 4) for s in SCORES}
    cis = {
        f"{a}_minus_{b}": {
            "diff": round(auroc[a] - auroc[b], 4),
            **dict(
                zip(
                    ("ci95_lo", "ci95_hi", "ci983_lo", "ci983_hi"),
                    bootstrap(y, tscore[a][lab_idx], tscore[b][lab_idx], config.n_boot, config.seed),
                )
            ),
        }
        for a, b in COMPARISONS
    }

    def lang(i: int) -> str:
        # per-prompt boundary: langdetect raises on text with no letter features, which is a property of the prompt
        try:
            return detect(traffic[i]["prompt"][:500])
        except LangDetectException:
            LOGGER.warning(f"langdetect found no features in {traffic[i]['id']}")
            return "undetected"

    def lang_share(idx: np.ndarray) -> float:
        langs = [lang(i) for i in idx]
        return round(sum(l not in ("en", "undetected") for l in langs) / len(langs), 3)

    top20 = {}
    for s in ("olmo_one", "olmo_diff", "tfidf_one", "tfidf_diff"):
        order = np.argsort(-tscore[s])[:20]
        top20[s] = [
            {
                "rank": r + 1,
                "topic": int(topics[j]),
                "score": round(float(tscore[s][j]), 4),
                "label": human.get(topics[j]),
                "median_tokens": float(np.median(n_tok[members[topics[j]]])),
                "non_english_share": lang_share(members[topics[j]]),
                "example": traffic[members[topics[j]][0]]["prompt"][:300],
            }
            for r, j in enumerate(order)
        ]
    write_json(top20, out / "top20.json")

    decision = {
        "n_traffic": len(traffic),
        "n_topics": len(topics),
        "n_dolci_heldout": len(dolci_held),
        "n_nan": int(sum(np.isnan(v).sum() for v in prompt_scores.values())),
        "labelled": dict(Counter(human[topics[j]] for j in lab_idx)),
        "auroc_real_vs_covered": auroc,
        "comparisons": cis,
        "traffic_vs_heldout_dolci_prompt_auroc": heldout_auroc,
        "labelled_in_top20": {s: sum(r["label"] is not None for r in v) for s, v in top20.items()},
        "real_in_top20": {s: sum(r["label"] == "real_gap" for r in v) for s, v in top20.items()},
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
