# grouping_test: plants tag-free gaps in the dolci sample and asks which prompt grouping ranks the affected traffic prompts first by score change, under delta-knn and a tf-idf word baseline.
# uv run --no-sync -m scripts.run_grouping_test --gap_list_v2_dir results/raw/261030_gap_list_v2 --olmo_clusters_v2_dir results/raw/261030_olmo_clusters_v2 --tags_dir data/processed/kind_groups --chat_path data/processed/chat_kinds/chat_kinds.jsonl --output_dir results/raw/261030_grouping_test_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from scripts.run_gap_list_v2 import length_pct, template_key
from src.data.io import load_jsonl
from src.interp.kmeans import spherical_kmeans
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PRIMARY = ("pasted_document", "roleplay", "code", "not_english", "jailbreak", "creative")
SHARES = (0.01, 0.05, 0.10)


@dataclass
class GroupingTestConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    olmo_clusters_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    tags_dir: str = ""
    chat_path: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    n_mix: int = 200000
    k: int = 100
    lost_share: float = 0.5
    length_adjust: bool = False
    chat_lost_share: float = 0.8
    min_answer: int = 100
    n_dolci_clusters: int = 200
    n_cluster_draws: int = 5
    clusters_per_draw: int = 5
    n_random_small: int = 3
    min_cluster: int = 10
    min_kind_part: int = 5
    min_split: int = 20
    prefix_chars: int = 40
    chunk: int = 2000
    seed: int = 0


def topk_idx(ref: torch.Tensor, x: torch.Tensor, k: int, chunk: int) -> np.ndarray:
    ref, x = torch.nn.functional.normalize(ref, dim=1), torch.nn.functional.normalize(x, dim=1)
    return torch.cat([(x[s : s + chunk] @ ref.T).topk(k, dim=1).indices for s in range(0, len(x), chunk)]).cpu().numpy()


def tfidf_topk(r, q, k: int, chunk: int, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    # k-th nearest cosine distance and neighbour indices in tf-idf space, against the kept reference rows only
    rr = r if keep is None else r[np.where(keep)[0]]
    dist, idx = [], []
    for s in range(0, q.shape[0], chunk):
        sims = torch.tensor((q[s : s + chunk] @ rr.T).toarray(), device="cuda")
        top = sims.topk(k, dim=1)
        dist.append(torch.sqrt(torch.clamp(2 - 2 * top.values[:, -1], min=0)).cpu().numpy())
        idx.append(top.indices.cpu().numpy())
    return np.concatenate(dist), np.concatenate(idx)


def main() -> None:
    config = simple_parsing.parse(GroupingTestConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    torch.backends.cudnn.deterministic = True

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    tags = load_jsonl(Path(config.tags_dir) / "traffic_tags.jsonl")
    assert [t["id"] for t in tags] == [r["id"] for r in scores]
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_t, n_mix, n_ref, n_h = rows["sizes"].tolist()
    assert n_mix == config.n_mix
    len_t, len_h = rows["lengths"][:n_t], rows["lengths"][n_t + n_mix + n_ref :]
    topic = np.array([r["topic"] for r in scores])
    keys = [template_key(p, config.prefix_chars) for p in prompts]

    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=n_mix, replace=False))
    dolci_texts = [mix_text[i] for i in sample]
    del mix_text
    pos = {int(m): i for i, m in enumerate(sample)}
    chat_rows = np.array([pos[r["mix_index"]] for r in load_jsonl(config.chat_path)])

    feats = np.load(config.feats_path)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix = x[:n_t], x[n_t : n_t + n_mix]
    xdolma, xh = x[n_t + n_mix : n_t + n_mix + n_ref], x[n_t + n_mix + n_ref :]
    sd = xmix.std(0) + 1e-6
    xt_s, xh_s, xmix_s = xt / sd, xh / sd, xmix / sd
    dolma_t = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()
    dolma_h = knn_distances(xdolma, xh, (config.k,))[config.k].cpu().numpy()
    del x, xt, xh, xdolma, xmix

    # groupings
    cl = np.load(Path(config.olmo_clusters_v2_dir) / "cluster_labels.npy")
    sizes = np.bincount(cl)
    big = np.where(sizes >= config.min_cluster)[0]
    cent = torch.nn.functional.normalize(
        torch.stack([xt_s[torch.tensor(np.where(cl == c)[0], device="cuda")].mean(0) for c in big]), dim=1
    )
    small = np.where(sizes[cl] < config.min_cluster)[0]
    g2 = cl.copy()
    if len(small):
        near = (torch.nn.functional.normalize(xt_s[torch.tensor(small, device="cuda")], dim=1) @ cent.T).argmax(1)
        g2[small] = big[near.cpu().numpy()]
    primary = np.array([next((k for k in PRIMARY if t[k] is True), "other") for t in tags], dtype=object)
    part = pd.Series(list(zip(topic, primary)))
    counts = part.map(part.value_counts())
    g3 = np.array(
        [f"{t}|{p}" if c >= config.min_kind_part else f"{t}|other" for (t, p), c in zip(part, counts, strict=True)]
    )
    g4 = np.array([str(t) for t in topic], dtype=object)
    for t in np.unique(topic):
        idx = np.where(topic == t)[0]
        if len(idx) >= config.min_split:
            lab, _ = spherical_kmeans(xt_s[torch.tensor(idx, device="cuda")], 2, n_init=3, seed=config.seed)
            g4[idx] = [f"{t}|{v}" for v in lab.cpu().numpy()]
    groupings = {
        "topics": topic,
        "olmo_clusters_min10": g2,
        "topic_x_kind": g3,
        "topic_x_olmo2": g4,
        "prompts": np.arange(n_t),
    }
    group_info = {
        g: {"n_groups": int(len(np.unique(v))), "median_size": float(np.median(pd.Series(v).value_counts()))}
        for g, v in groupings.items()
    }
    LOGGER.info(f"{group_info=}")

    # neighbours on the full sample, for the right answers
    nn_olmo = topk_idx(xmix_s, xt_s, config.k, config.chunk)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000)
    r_tfidf = vec.fit_transform(dolci_texts)
    q_tfidf = vec.transform(prompts)
    _, nn_tfidf = tfidf_topk(r_tfidf, q_tfidf, config.k, config.chunk)

    labels, _ = spherical_kmeans(xmix_s, config.n_dolci_clusters, n_init=1, seed=config.seed)
    dcl = labels.cpu().numpy()
    planted = {"chat": chat_rows}
    for s in range(config.n_cluster_draws):
        pick = np.random.default_rng(s).choice(config.n_dolci_clusters, size=config.clusters_per_draw, replace=False)
        planted[f"clusters{s}"] = np.where(np.isin(dcl, pick))[0]
    mean_b = int(np.mean([len(v) for k, v in planted.items() if k != "chat"]))
    controls = {"random_chat": np.random.default_rng(100).choice(n_mix, size=len(chat_rows), replace=False)}
    for s in range(config.n_random_small):
        controls[f"random_small{s}"] = np.random.default_rng(200 + s).choice(n_mix, size=mean_b, replace=False)

    def arm_scores(removed: np.ndarray | None) -> dict[str, np.ndarray]:
        keep = np.ones(n_mix, dtype=bool)
        if removed is not None:
            keep[removed] = False
        ref = xmix_s[torch.tensor(keep, device="cuda")]
        s_t = knn_distances(ref, xt_s, (config.k,), scale=False)[config.k].cpu().numpy() - dolma_t
        tf, _ = tfidf_topk(r_tfidf, q_tfidf, config.k, config.chunk, keep)
        if not config.length_adjust:
            return {"delta_knn": s_t, "tfidf": tf}
        s_h = knn_distances(ref, xh_s, (config.k,), scale=False)[config.k].cpu().numpy() - dolma_h
        return {"delta_knn": length_pct(s_t, len_t, s_h, len_h), "tfidf": tf}

    def group_score(score: np.ndarray, groups: np.ndarray) -> pd.Series:
        frame = pd.DataFrame({"g": groups, "key": keys, "s": score})
        return frame.groupby(["g", "key"])["s"].median().groupby(level=0).median()

    def rank_metrics(change: np.ndarray, groups: np.ndarray, answer: np.ndarray) -> dict:
        # groups ranked by their score change; every prompt inherits its group's score for the auroc
        gs = group_score(change, groups).sort_values(ascending=False, kind="stable")
        size = pd.Series(groups).value_counts().loc[gs.index].cumsum().to_numpy()
        per_prompt = gs.loc[groups].to_numpy()
        out_ = {"auroc": round(float(roc_auc_score(answer, per_prompt)), 4)}
        for share in SHARES:
            sel = np.isin(groups, gs.index[: int((size < share * n_t).sum()) + 1])
            rec = answer[sel].sum() / answer.sum()
            out_[str(share)] = {
                "recall": round(float(rec), 4),
                "recall_of_max": round(float(rec / min(1.0, sel.sum() / answer.sum())), 4),
                "precision": round(float(answer[sel].mean()), 4),
                "share_selected": round(float(sel.mean()), 4),
            }
        return out_

    LOGGER.info("scoring full arm")
    full = arm_scores(None)
    arms = {name: arm_scores(rm) for name, rm in (planted | controls).items()}
    results, skipped = [], []
    for name, rm in planted.items():
        removed = np.zeros(n_mix, dtype=bool)
        removed[rm] = True
        ctrl = "random_chat" if name == "chat" else "random_small0"
        lost = config.chat_lost_share if name == "chat" else config.lost_share
        for space, nn in (("olmo", nn_olmo), ("tfidf", nn_tfidf)):
            answer = removed[nn].mean(1) >= lost
            if answer.sum() < config.min_answer:
                skipped.append({"planted": name, "answer_space": space, "n_answer": int(answer.sum())})
                continue
            for det in ("delta_knn", "tfidf"):
                for g, groups in groupings.items():
                    results.append(
                        {
                            "id": f"{name}|{space}|{det}|{g}",
                            "planted": name,
                            "answer_space": space,
                            "detector": det,
                            "grouping": g,
                            "n_removed": int(len(rm)),
                            "n_answer": int(answer.sum()),
                            "base_rate": round(float(answer.mean()), 4),
                            "removed": rank_metrics(arms[name][det] - full[det], groups, answer),
                            "random": rank_metrics(arms[ctrl][det] - full[det], groups, answer),
                        }
                    )
        LOGGER.info(f"{name}: done, n_removed {len(rm)}")
    write_jsonl(results, out / "results.jsonl")

    summary = {}
    for space in ("olmo", "tfidf"):
        for det in ("delta_knn", "tfidf"):
            for g in groupings:
                sel = [r for r in results if r["answer_space"] == space and r["detector"] == det and r["grouping"] == g]
                chat = [r for r in sel if r["planted"] == "chat"]
                b = [r for r in sel if r["planted"] != "chat"]
                summary[f"{space}|{det}|{g}"] = {
                    "chat_auroc": chat[0]["removed"]["auroc"] if chat else None,
                    "chat_auroc_random": chat[0]["random"]["auroc"] if chat else None,
                    "chat_precision": {s: chat[0]["removed"][str(s)]["precision"] for s in SHARES} if chat else None,
                    "chat_recall_of_max": {s: chat[0]["removed"][str(s)]["recall_of_max"] for s in SHARES}
                    if chat
                    else None,
                    "n_cluster_draws": len(b),
                    "clusters_auroc_mean": round(float(np.mean([r["removed"]["auroc"] for r in b])), 4) if b else None,
                    "clusters_auroc_random_mean": round(float(np.mean([r["random"]["auroc"] for r in b])), 4)
                    if b
                    else None,
                    "clusters_recall_of_max_mean": {
                        s: round(float(np.mean([r["removed"][str(s)]["recall_of_max"] for r in b])), 4) for s in SHARES
                    }
                    if b
                    else None,
                }
    decision = {
        "n_traffic": n_t,
        "groupings": group_info,
        "n_removed": {k: int(len(v)) for k, v in (planted | controls).items()},
        "skipped_small_answer": skipped,
        "n_rows": len(results),
        "n_rows_expected": (len(planted) * 2 - len(skipped)) * 2 * len(groupings),
        "summary": summary,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'results.jsonl'} and {out / 'decision.json'}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
