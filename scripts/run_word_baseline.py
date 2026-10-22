# exp47: neighbour-count gap score against a tf-idf word gap score, on exp44's planted gaps and on the user's topic labels.
# uv run -m scripts.run_word_baseline --run_dir results/raw/261022_exp47_word_baseline_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp44_dir results/raw/261022_exp44_planted_gaps_v2 --label_dirs results/raw/261022_exp41_topic_labels_v1/labels results/raw/261022_exp43_topic_labels_v1/labels --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class WordBaselineConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp44_dir: str = ""
    label_dirs: list[str] = field(default_factory=list)
    seed: int = 0
    k: int = 50
    chars: int = 1000
    max_features: int = 200_000
    min_df: int = 5
    top_n: int = 293
    better_by: int = 3


def word_logits(t_text: list[str], m_text: list[str], fold_t: np.ndarray, fold_m: np.ndarray, config) -> np.ndarray:
    # two-fold tf-idf logistic regression, traffic 1 vs mix 0, each traffic prompt scored out of fold
    vec = TfidfVectorizer(
        sublinear_tf=True, min_df=config.min_df, max_features=config.max_features, ngram_range=(1, 2), dtype=np.float32
    )
    x = vec.fit_transform([t[: config.chars] for t in t_text] + [t[: config.chars] for t in m_text])
    y = np.r_[np.ones(len(t_text)), np.zeros(len(m_text))]
    fold = np.r_[fold_t, fold_m]
    out = np.empty(len(t_text))
    for k in (0, 1):
        clf = SGDClassifier(
            loss="log_loss", class_weight="balanced", alpha=1e-6, max_iter=5, tol=None, random_state=config.seed
        )
        clf.fit(x[fold != k], y[fold != k])
        test = np.where(fold_t == k)[0]
        out[test] = clf.decision_function(x[test])
    return out


def main():
    config = simple_parsing.parse(WordBaselineConfig)
    assert config.run_dir and config.exp37_dir and config.exp44_dir, "run_dir, exp37_dir, exp44_dir required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    ratio37 = np.load(Path(config.exp37_dir) / "ratio.npy")
    m_np, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    t_text = [r["text"] for r in rows]
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    fold_t, fold_m = (t_hash & 1).astype(np.int64), (m_hash & 1).astype(np.int64)
    clusters = {c["id"]: c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}

    # exp44 v2's deletion: every mix row among the k nearest mix rows of any planted prompt
    planted = [p["cluster"] for p in json.loads((Path(config.exp44_dir) / "decision.json").read_text())["planted"]]
    m_gpu = torch.from_numpy(m_np).cuda()
    keep = np.ones(len(m_np), dtype=bool)
    for c in planted:
        q = torch.from_numpy(t_np[members[c]]).cuda()
        idx = torch.cat(
            [(q[s : s + 1024] @ m_gpu.T).topk(config.k, dim=1).indices.flatten() for s in range(0, len(q), 1024)]
        )
        keep[np.unique(idx.cpu().numpy())] = False
    LOGGER.info(f"planted {planted}, deleted {int((~keep).sum())} mix rows")
    assert int((~keep).sum()) == 61769, "deletion must match exp44 v2"
    m_abl = torch.from_numpy(m_np[keep]).cuda()
    del m_gpu
    ratio, near = density_ratios_batched(
        t_np,
        t_hash,
        np.arange(len(rows)),
        torch.from_numpy(t_np).cuda(),
        torch.from_numpy(t_hash).cuda(),
        m_abl,
        torch.from_numpy(m_hash[keep]).cuda(),
        k=config.k,
        return_nearest_mix=True,
    )
    knn_rank = sorted(order, key=lambda c: (-float(np.median(ratio[members[c]])), float(np.median(near[members[c]]))))
    LOGGER.info("neighbour count on the ablated mix done")

    w_abl = word_logits(t_text, [m_text[i] for i in np.where(keep)[0]], fold_t, fold_m[keep], config)
    word_rank = sorted(order, key=lambda c: -float(np.median(w_abl[members[c]])))
    LOGGER.info("word score on the ablated mix done")

    pos = {
        "neighbour_count": {c: knn_rank.index(c) + 1 for c in planted},
        "words": {c: word_rank.index(c) + 1 for c in planted},
    }
    in_top = {a: sum(p <= config.top_n for p in ps.values()) for a, ps in pos.items()}
    lead = in_top["neighbour_count"] - in_top["words"]
    verdict = (
        "neighbour_count_beats_words"
        if lead >= config.better_by
        else "words_better"
        if -lead >= config.better_by
        else "words_as_good"
    )

    # secondary: user labels against both scores on the unablated mix
    w_full = word_logits(t_text, m_text, fold_t, fold_m, config)
    lab = {}
    for d in config.label_dirs:
        for p in Path(d).glob("*.json"):
            lab[int(p.stem.split("_c")[1])] = json.loads(p.read_text())["verdict"]
    lab = {c: v for c, v in lab.items() if v in ("real_gap", "partly", "covered") and c in members}
    ys = np.array([v != "covered" for v in lab.values()])
    auc = {
        "neighbour_count": float(roc_auc_score(ys, [np.median(ratio37[members[c]]) for c in lab])),
        "words": float(roc_auc_score(ys, [np.median(w_full[members[c]]) for c in lab])),
    }

    decision = {
        "verdict": verdict,
        "planted_in_top": in_top,
        "top_n": config.top_n,
        "planted_positions": {a: {str(c): p for c, p in ps.items()} for a, ps in pos.items()},
        "label_auc": {a: round(v, 4) for a, v in auc.items()},
        "n_labelled": len(lab),
        "n_labelled_uncovered": int(ys.sum()),
        "mix_rows_deleted": int((~keep).sum()),
    }
    write_json(decision, run / "decision.json")
    print(f"{in_top=} {verdict=}")
    print(f"positions {decision['planted_positions']}")
    print(f"label auc {decision['label_auc']} over {len(lab)} topics")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
