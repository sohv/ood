# exp94 and exp95: remove half of exp37's traffic topics (or of n_groups groups of neighbouring topics) from a training-only reference and test whether olmo, tf-idf, minilm and length detectors flag them.
# uv run --no-sync -m scripts.run_exp94_topic_removal --dataset_path results/raw/261021_exp37_gap_pipeline_v2 --output_dir results/raw/261028_exp94_topic_removal_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

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

from src.data.gap_pool import load_mix, load_traffic_pool
from src.interp.features import LAYERS, POOLINGS, encode_chat, extract
from src.interp.train_only_detectors import GpuMahalanobis, knn_distances, tfidf_knn_distances
from src.metrics.gap_pipeline import spherical_kmeans
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
KS = (1, 10, 50)
ARMS = ("main", "dolci", "fake")
EXP93_SETTING = ("olmo", 20, "cmean", "knn_k50")


@dataclass
class Exp94Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    n_per_topic: int = 30
    n_ref: int = 20
    n_dolci: int = 20000
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    max_tokens: int = 512
    batch_tokens: int = 32768
    pca_dims: int = 256
    text_chars: int = 2000
    n_boot: int = 10000
    n_groups: int = 0
    seed: int = 0


def build_sample(config: Exp94Config, rng: np.random.Generator) -> tuple[list[dict], np.ndarray, dict]:
    c37 = json.loads((Path(config.dataset_path) / "config.json").read_text())
    rows, t_emb, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.dataset_path) / "labels.npy")
    assert len(labels) == len(rows), f"{len(labels)} labels for {len(rows)} traffic rows"
    n_topics = labels.max() + 1 if not config.num_tasks else config.num_tasks
    group = np.arange(n_topics)
    if config.n_groups:
        # neighbouring topics: spherical k-means on the topics' normalised mean minilm embeddings
        sums = np.zeros((labels.max() + 1, t_emb.shape[1]))
        np.add.at(sums, labels, t_emb.astype(np.float64))
        cent = torch.nn.functional.normalize(torch.tensor(sums[:n_topics], dtype=torch.float32), dim=1)
        group, _ = spherical_kmeans(cent, config.n_groups, 50, config.seed)
    recs, emb = [], []
    for t in range(n_topics):
        for i in rng.choice(np.where(labels == t)[0], config.n_per_topic, replace=False):
            recs.append(
                {"id": f"t{i}", "topic": int(t), "group": int(group[t]), "kind": "traffic", "prompt": rows[i]["text"]}
            )
            emb.append(t_emb[i])
    m_emb, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    n_dolci = config.n_dolci if not config.num_tasks else config.num_tasks * 10
    for i in rng.choice(len(m_text), n_dolci, replace=False):
        recs.append({"id": f"m{i}", "topic": -1, "group": -1, "kind": "dolci", "prompt": m_text[i]})
        emb.append(m_emb[i])
    return recs, np.array(emb, dtype=np.float32), {int(t): int(g) for t, g in enumerate(group)}


def olmo_features(config: Exp94Config, prompts: list[str]) -> dict[str, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    encoded = [encode_chat(tok, p[: 4 * config.text_chars], config.max_tokens) for p in prompts]
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=LAYERS)
    del model
    torch.cuda.empty_cache()
    return feats


def make_arm(arm: str, topic: np.ndarray, group_of: np.ndarray, kind: np.ndarray, seed: int, n_ref: int) -> dict:
    # reference index, eval index, eval label (1 = should flag), eval topic, eval part; removal and val/test go by group
    rng = np.random.default_rng(1000 + seed)
    n_topics, n_groups = topic.max() + 1, group_of.max() + 1
    removed_group = np.zeros(n_groups, bool)
    removed_group[rng.permutation(n_groups)[: n_groups // 2]] = True
    part_group = np.where(rng.random(n_groups) < 0.5, "val", "test")
    removed_topic, part_topic = removed_group[group_of], part_group[group_of]
    in_ref = np.zeros(len(topic), bool)
    held = np.zeros(len(topic), bool)
    for t in range(n_topics):
        idx = rng.permutation(np.where(topic == t)[0])
        if removed_topic[t]:
            held[idx[n_ref:]] = True
        else:
            in_ref[idx[:n_ref]] = True
            held[idx[n_ref:]] = True
    if arm == "dolci":
        in_ref |= kind == "dolci"
    ev = np.where(held & (kind == "traffic"))[0]
    label = removed_topic[topic[ev]].astype(int)
    if arm == "fake":
        # only kept topics; half of each topic's held-out prompts are called removed at random
        ev = ev[label == 0]
        label = (np.random.default_rng(2000 + seed).random(len(ev)) < 0.5).astype(int)
    return {
        "ref": np.where(in_ref)[0],
        "ev": ev,
        "label": label,
        "topic": topic[ev],
        "part": part_topic[topic[ev]],
        "group": group_of[topic[ev]],
    }


def topic_medians(score: np.ndarray, topic: np.ndarray, label: np.ndarray, part: np.ndarray, group: np.ndarray) -> dict:
    out = {}
    for t in np.unique(topic):
        m = topic == t
        out[int(t)] = (float(np.median(score[m])), int(label[m][0]), str(part[m][0]), int(group[m][0]))
    return out


def evaluate(score: np.ndarray, a: dict, arm: str) -> tuple[dict, dict | None]:
    res = {}
    for p in ("val", "test"):
        m = a["part"] == p
        res[f"prompt_{p}"] = float(roc_auc_score(a["label"][m], score[m]))
    if arm == "fake":
        return res, None
    tm = topic_medians(score, a["topic"], a["label"], a["part"], a["group"])
    for p in ("val", "test"):
        vals = [(s, y) for s, y, q, _ in tm.values() if q == p]
        res[f"topic_{p}"] = float(roc_auc_score([y for _, y in vals], [s for s, _ in vals]))
    return res, tm


def score_all(config: Exp94Config, recs: list[dict], feats: dict, emb: np.ndarray) -> tuple[list[dict], dict]:
    topic = np.array([r["topic"] for r in recs])
    kind = np.array([r["kind"] for r in recs])
    texts = [r["prompt"][: config.text_chars] for r in recs]
    log_len = torch.tensor(np.log(feats["n_content_tokens"])[:, None], device="cuda")
    emb_t = torch.tensor(emb, device="cuda")
    group_of = np.zeros(topic.max() + 1, dtype=int)
    for r in recs:
        if r["topic"] >= 0:
            group_of[r["topic"]] = r["group"]
    arms = {(arm, s): make_arm(arm, topic, group_of, kind, s, config.n_ref) for arm in ARMS for s in config.split_seeds}
    rows, topic_scores = [], {}

    def add(key: tuple, arm: str, seed: int, score: np.ndarray) -> None:
        res, tm = evaluate(score, arms[(arm, seed)], arm)
        family, layer, pooling, setting = key
        rows.append(
            {
                "id": f"{family}|{layer}|{pooling}|{setting}|{arm}|{seed}",
                "family": family,
                "layer": layer,
                "pooling": pooling,
                "setting": setting,
                "arm": arm,
                "seed": seed,
                **{k: round(v, 4) for k, v in res.items()},
            }
        )
        if tm is not None:
            topic_scores[(key, arm, seed)] = tm

    for (arm, seed), a in arms.items():
        ref, ev = a["ref"], a["ev"]
        add(("random", None, None, "uniform"), arm, seed, np.random.default_rng(3000 + seed).random(len(ev)))
        gaps = (log_len[ev] - log_len[ref].T).abs()
        for k in KS:
            add(
                ("length", None, None, f"knn_k{k}"),
                arm,
                seed,
                gaps.topk(k, dim=1, largest=False).values[:, -1].cpu().numpy(),
            )
        del gaps
        for k, s in knn_distances(emb_t[ref], emb_t[ev], KS, scale=False).items():
            add(("minilm", None, None, f"knn_k{k}"), arm, seed, s.cpu().numpy())
        for k, s in tfidf_knn_distances([texts[i] for i in ref], [texts[i] for i in ev], KS).items():
            add(("tfidf", None, None, f"knn_k{k}"), arm, seed, s)
        LOGGER.info(f"baselines done for {arm} seed {seed}")

    for pooling in POOLINGS:
        for li, layer in enumerate(LAYERS):
            x = torch.tensor(feats[pooling][:, li], device="cuda")
            for (arm, seed), a in arms.items():
                for k, s in knn_distances(x[a["ref"]], x[a["ev"]], KS).items():
                    add(("olmo", layer, pooling, f"knn_k{k}"), arm, seed, s.cpu().numpy())
                for name, dims in (("maha_full", 0), (f"maha_pca{config.pca_dims}", config.pca_dims)):
                    add(
                        ("olmo", layer, pooling, name),
                        arm,
                        seed,
                        GpuMahalanobis(x[a["ref"]], pca_dims=dims)(x[a["ev"]]).cpu().numpy(),
                    )
            LOGGER.info(f"olmo {pooling} layer {layer} done")
            del x
            torch.cuda.empty_cache()

    # near-copies: held-out kept prompts whose nearest reference prompt has tf-idf cosine above 0.9 (main arm, seed 0)
    a = arms[("main", config.split_seeds[0])]
    d1 = tfidf_knn_distances([texts[i] for i in a["ref"]], [texts[i] for i in a["ev"]], (1,))[1]
    near = {
        "kept_share": float((d1[a["label"] == 0] < np.sqrt(0.2)).mean()),
        "removed_share": float((d1[a["label"] == 1] < np.sqrt(0.2)).mean()),
    }
    return rows, {"topic_scores": topic_scores, "near_copies": near}


def bootstrap_diff(ta: dict, tb: dict, n_boot: int, seed: int) -> list[float]:
    # paired over test groups (a group is one topic in exp94): auroc(a) - auroc(b) on the resampled groups' topics
    test = [t for t, v in ta.items() if v[2] == "test"]
    y = np.array([ta[t][1] for t in test])
    sa, sb = np.array([ta[t][0] for t in test]), np.array([tb[t][0] for t in test])
    grp = np.array([ta[t][3] for t in test])
    members = [np.where(grp == g)[0] for g in np.unique(grp)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    return [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]


def decide(rows: list[dict], extra: dict, config: Exp94Config) -> dict:
    def key(r):
        return (r["family"], r["layer"], r["pooling"], r["setting"])

    chosen, summary = {}, {}
    for family in ("olmo", "tfidf", "minilm", "length", "random"):
        configs = {}
        for r in rows:
            if r["family"] == family and r["arm"] == "main":
                configs.setdefault(key(r), []).append(r["topic_val"])
        chosen[family] = max(configs, key=lambda c: np.mean(configs[c]))
    chosen["olmo_exp93_setting"] = EXP93_SETTING
    for name, k in chosen.items():
        summary[name] = {"setting": list(k)}
        for arm in ARMS:
            sel = [r for r in rows if key(r) == k and r["arm"] == arm]
            for m in ("prompt_test", "topic_test"):
                if m in sel[0]:
                    summary[name][f"{arm}_{m}"] = round(float(np.mean([r[m] for r in sel])), 4)
    ts = extra["topic_scores"]
    cis = {}
    for other in ("tfidf", "minilm"):
        cis[f"olmo_minus_{other}"] = {
            s: bootstrap_diff(ts[(chosen["olmo"], "main", s)], ts[(chosen[other], "main", s)], config.n_boot, s)
            for s in config.split_seeds
        }
    ci = cis["olmo_minus_tfidf"].values()
    verdict = (
        "beats words" if all(lo > 0 for lo, _ in ci) else "loses to words" if all(hi < 0 for _, hi in ci) else "tie"
    )
    broken = [n for n, v in summary.items() if not 0.45 <= v["fake_prompt_test"] <= 0.55]
    return {
        "summary": summary,
        "bootstrap_ci_topic_test": cis,
        "verdict_vs_words": verdict,
        "fake_broken": broken,
        "near_copies": extra["near_copies"],
    }


def main() -> None:
    config = simple_parsing.parse(Exp94Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    recs, emb, groups = build_sample(config, np.random.default_rng(config.seed))
    LOGGER.info(f"{len(recs)=} {len(set(groups.values()))=}")
    write_json(groups, out / "groups.json")
    write_jsonl(recs, out / "prompts.jsonl")
    feats = olmo_features(config, [r["prompt"] for r in recs])
    rows, extra = score_all(config, recs, feats, emb)
    write_jsonl(rows, out / "scores.jsonl")
    n_expected = (1 + 3 * 3 + len(POOLINGS) * len(LAYERS) * 5) * len(ARMS) * len(config.split_seeds)
    decision = {
        "n_rows": len(rows),
        "n_rows_expected": n_expected,
        "n_prompts": len(recs),
        **decide(rows, extra, config),
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(
        json.dumps(
            {
                k: decision[k]
                for k in ("summary", "bootstrap_ci_topic_test", "verdict_vs_words", "fake_broken", "near_copies")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
