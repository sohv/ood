# final_detector_tests: scores the final delta-knn (layer 20, k 100, length-matched dolma v1.6) and one-sided knn on must-not-flag and hide-half-the-topics.
# uv run --no-sync -m scripts.run_final_detector_tests --group_removal_dir results/raw/261028_group_removal_v1 --negatives_test_dir results/raw/261029_negatives_test_negatives_v1 --clean_path results/raw/261029_clean_traffic_detectors_clean_v1/clean_prompts.jsonl --xstest_csv data/raw/xstest/xstest_v2_completions_gpt4.csv --dolma_path data/processed/repro_two_sided/dolma.jsonl --output_dir results/raw/261031_final_detector_tests_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

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

from scripts.run_negatives_test_negatives import NEGATIVES, POSITIVES, NegativesTestConfig, load_sets
from scripts.run_topic_removal import make_arm
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class FinalTestsConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    group_removal_dir: str = ""
    negatives_test_dir: str = ""
    clean_path: str = ""
    xstest_csv: str = ""
    dolma_path: str = ""
    ks: list[int] = field(default_factory=lambda: [50, 100])
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_ref: int = 3000
    n_topic_ref: int = 20
    layer: int = 20
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def scores(ref: torch.Tensor, dolma: torch.Tensor, x: torch.Tensor, ks: tuple[int, ...]) -> dict[str, np.ndarray]:
    to_id, to_r = knn_distances(ref, x, ks), knn_distances(dolma, x, ks)
    out = {f"delta_knn_k{k}": (to_id[k] - to_r[k]).cpu().numpy() for k in ks}
    return out | {f"knn_one_sided_k{k}": to_id[k].cpu().numpy() for k in ks}


def main() -> None:
    config = simple_parsing.parse(FinalTestsConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    ks = tuple(config.ks)

    # must-not-flag set and references, built exactly as two_sided_layers
    c102 = NegativesTestConfig(
        group_removal_dir=config.group_removal_dir,
        clean_path=config.clean_path,
        xstest_csv=config.xstest_csv,
        n_ref=config.n_ref,
    )
    sets = load_sets(c102, np.random.default_rng(config.seed))
    near = {r["id"]: r["near_copy"] for r in load_jsonl(Path(config.negatives_test_dir) / "scores.jsonl")}
    eval_rows = [(n, t) for n in sets if n != "ref_dolci" for t in sets[n]]
    neg_text, neg_y = [], []
    for i, (name, t) in enumerate(eval_rows):
        if name in POSITIVES + NEGATIVES and not near[f"{name}_{i}"]:
            neg_text.append(t)
            neg_y.append(int(name in POSITIVES))
    neg_y = np.array(neg_y)
    dolci_ref = sets["ref_dolci"]
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    rng = np.random.default_rng(config.seed)
    dolci_len = np.array(
        [min(len(x), config.max_tokens) for x in tok(dolci_ref, add_special_tokens=False)["input_ids"]]
    )
    dolma_raw = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    ids = tok([t[: 4 * config.text_chars] for t in dolma_raw], add_special_tokens=False)["input_ids"]
    cut = rng.choice(dolci_len, size=len(ids))
    dolma = [tok.decode(x[:n]) for x, n in zip(ids, cut, strict=True)]

    # hide-half-the-topics prompts, traffic rows only
    g_recs = load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl")
    topic = np.array([r["topic"] for r in g_recs])
    kind = np.array([r["kind"] for r in g_recs])
    group_of = np.zeros(topic.max() + 1, dtype=int)
    for r in g_recs:
        if r["topic"] >= 0:
            group_of[r["topic"]] = r["group"]
    is_t = np.where(kind == "traffic")[0]
    pos_t = np.full(len(g_recs), -1)
    pos_t[is_t] = np.arange(len(is_t))
    g_text = [g_recs[i]["prompt"] for i in is_t]
    LOGGER.info(f"{len(neg_text)=} {int(neg_y.sum())=} {len(dolci_ref)=} {len(dolma)=} {len(g_text)=}")

    texts = neg_text + dolci_ref + dolma + g_text
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(
        model,
        [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in texts],
        tok.pad_token_id or 0,
        max_tokens=config.batch_tokens,
        layers=[config.layer],
    )["cmean"]
    del model
    torch.cuda.empty_cache()
    x = torch.tensor(feats[:, 0], device="cuda").float()
    b = np.cumsum([0, len(neg_text), len(dolci_ref), len(dolma), len(g_text)])
    xneg, xdolci, xdolma, xg = x[b[0] : b[1]], x[b[1] : b[2]], x[b[2] : b[3]], x[b[3] : b[4]]

    must_not_flag = {s: round(float(roc_auc_score(neg_y, v)), 4) for s, v in scores(xdolci, xdolma, xneg, ks).items()}
    LOGGER.info(f"{must_not_flag=}")

    topic_rows = []
    for seed in config.split_seeds:
        a = make_arm("main", topic, group_of, kind, seed, config.n_topic_ref)
        ref, ev = pos_t[a["ref"]], pos_t[a["ev"]]
        assert (ref >= 0).all() and (ev >= 0).all()
        test = a["part"] == "test"
        row = {"seed": seed, "n_ref": len(ref), "n_eval": int(test.sum())}
        for s, v in scores(xg[ref], xdolma, xg[ev], ks).items():
            ts = np.unique(a["topic"][test])
            med = [np.median(v[test & (a["topic"] == t)]) for t in ts]
            ty = [a["label"][test & (a["topic"] == t)][0] for t in ts]
            row[f"topic_test|{s}"] = round(float(roc_auc_score(ty, med)), 4)
            row[f"prompt_test|{s}"] = round(float(roc_auc_score(a["label"][test], v[test])), 4)
        topic_rows.append(row)
        LOGGER.info(f"{row=}")
    names = [k.split("|", 1)[1] for k in topic_rows[0] if k.startswith("topic_test|")]
    decision = {
        "n_must_not_flag": len(neg_y),
        "n_must_not_flag_positive": int(neg_y.sum()),
        "must_not_flag_auroc": must_not_flag,
        "hide_half_topics_topic_auroc": {
            s: round(float(np.mean([r[f"topic_test|{s}"] for r in topic_rows])), 4) for s in names
        },
        "hide_half_topics_prompt_auroc": {
            s: round(float(np.mean([r[f"prompt_test|{s}"] for r in topic_rows])), 4) for s in names
        },
        "per_seed": topic_rows,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
