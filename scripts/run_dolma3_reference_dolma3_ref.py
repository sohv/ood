# dolma3_reference: delta-mahalanobis and delta-knn with a sample of olmo 3's own pretraining mix as the reference, against the old dolma v1.6 reference, on the user's labels and source_removal's source removal.
# uv run --no-sync -m scripts.run_dolma3_reference_dolma3_ref --delta_maha_dir results/raw/261029_delta_maha_response_dmaha_v1 --source_removal_dir results/raw/261028_source_removal_v1 --group_removal_dir results/raw/261028_group_removal_v1 --dolma_path data/processed/repro_two_sided/dolma.jsonl --traffic_ranking_labels results/raw/261029_traffic_ranking_labels_v1/decision.json --blind_labels results/raw/261029_traffic_ranking_blind_labels_v1/decision.json --ref_out data/processed/dolma3_reference/dolma3.jsonl --output_dir results/raw/261029_dolma3_reference_dolma3_ref_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
import zstandard
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_source_removal import FAKE, assign_splits, folds
from scripts.run_traffic_ranking import TrafficRankingConfig, load_labels
from scripts.run_delta_maha_response_dmaha import dmaha_scores, length_removed_auroc, topic_medians
from src.data.io import load_jsonl
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
REPO = "allenai/dolma3_mix-150B-1025"


@dataclass
class Dolma3ReferenceConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    delta_maha_dir: str = ""
    source_removal_dir: str = ""
    group_removal_dir: str = ""
    dolma_path: str = ""
    traffic_ranking_labels: str = ""
    blind_labels: str = ""
    ref_out: str = ""
    download_dir: str = "/dev/shm/dolma3"
    n_ref: int = 3000
    n_files: int = 60
    min_chars: int = 200
    min_prompts: int = 5
    layer: int = 20
    k: int = 10
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def sample_dolma3(config: Dolma3ReferenceConfig, rng: np.random.Generator) -> list[dict]:
    # files drawn by compressed size, equal documents per file, so the sample follows the mix's byte shares
    info = HfApi().dataset_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size) for s in info.siblings if s.rfilename.startswith("data/")]
    sizes = np.array([s for _, s in files], dtype=float)
    picked = rng.choice(len(files), size=config.n_files, replace=False, p=sizes / sizes.sum())
    per_file = config.n_ref // config.n_files
    docs = []
    for i in picked:
        name = files[i][0]
        path = hf_hub_download(REPO, name, repo_type="dataset", cache_dir=config.download_dir)
        with open(path, "rb") as f:
            lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8").readlines()
        texts = [t for t in (json.loads(x)["text"] for x in lines) if len(t) >= config.min_chars]
        group = name.split("/")[1].rsplit("-", 1)[0]
        docs += [
            {"id": f"dolma3_{len(docs) + j:05d}", "source": group, "file": name, "text": texts[j]}
            for j in rng.choice(len(texts), size=per_file, replace=False)
        ]
        LOGGER.info(f"{name} {len(texts)=}")
    return docs


def main() -> None:
    config = simple_parsing.parse(Dolma3ReferenceConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    if Path(config.ref_out).exists():
        dolma3 = load_jsonl(config.ref_out)
    else:
        dolma3 = sample_dolma3(config, rng)
        Path(config.ref_out).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(dolma3, Path(config.ref_out))
    share = {
        s: round(sum(d["source"] == s for d in dolma3) / len(dolma3), 4) for s in sorted({d["source"] for d in dolma3})
    }
    LOGGER.info(f"{share=}")

    old = load_labels(TrafficRankingConfig())
    new = {int(r["id"][1:]): r["label"] for r in json.loads(Path(config.traffic_ranking_labels).read_text())["rows"]}
    blind = {int(r["id"][1:]): r["label"] for r in json.loads(Path(config.blind_labels).read_text())["rows"]}
    keys = {"old_117": old, "all_labels": {**old, **new, **blind}}

    saved = np.load(Path(config.delta_maha_dir) / f"olmo_layer{config.layer}_cmean.npz")
    topic_of = {r["id"]: r["topic"] for r in load_jsonl(Path(config.delta_maha_dir) / "prompt_scores.jsonl")}
    length_of = {r["id"]: r["length"] for r in load_jsonl(Path(config.delta_maha_dir) / "prompt_scores.jsonl")}
    t_ids = saved["traffic_ids"].tolist()
    x_t = torch.tensor(saved["traffic"], device="cuda").float()
    x_93 = torch.tensor(saved["source_removal"], device="cuda").float()
    topic = np.array([topic_of[i] for i in t_ids])
    log_len = np.array([length_of[i] for i in t_ids])
    if config.num_tasks:
        dolma3 = dolma3[: config.num_tasks * 10]

    dolci_ref = [
        r["prompt"] for r in load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl") if r["kind"] == "dolci"
    ][: config.n_ref]
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    texts = dolci_ref + dolma + [d["text"] for d in dolma3]
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
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
    )
    del model
    torch.cuda.empty_cache()
    x = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
    n_d, n_m = len(dolci_ref), len(dolci_ref) + len(dolma)
    x_dolci, refs = x[:n_d], {"dolma16": x[n_d:n_m], "dolma3": x[n_m:]}

    def scores(ref_id: torch.Tensor, xq: torch.Tensor) -> dict[str, np.ndarray]:
        to_id = knn_distances(ref_id, xq, (config.k,))[config.k].cpu().numpy()
        out = {"knn_one_sided": to_id}
        for name, xr in refs.items():
            d, m1 = dmaha_scores(ref_id, xr, xq)
            out[f"dmaha_{name}"] = d
            out[f"dknn_{name}"] = to_id - knn_distances(xr, xq, (config.k,))[config.k].cpu().numpy()
        out["maha_one_sided"] = m1
        return out

    prompt = scores(x_dolci, x_t)
    prompt["length"] = log_len
    LOGGER.info(f"nan: { {s: int(np.isnan(v).sum()) for s, v in prompt.items()} }")
    write_jsonl(
        [
            {"id": i, "topic": int(topic[j]), **{s: round(float(v[j]), 4) for s, v in prompt.items()}}
            for j, i in enumerate(t_ids)
        ],
        out / "prompt_scores.jsonl",
    )
    topics = np.array(sorted(set(topic.tolist())))
    tscore = {s: topic_medians(v, topic, topics) for s, v in prompt.items()}
    labels = {}
    for name, human in keys.items():
        mask = np.array([human.get(t) in ("real_gap", "covered") for t in topics])
        y = np.array([human.get(t) == "real_gap" for t in topics[mask]], dtype=int)
        pm = np.array([human.get(t) in ("real_gap", "covered") for t in topic])
        py = np.array([human.get(t) == "real_gap" for t in topic[pm]], dtype=int)
        labels[name] = {
            "n_real": int(y.sum()),
            "n_covered": int(len(y) - y.sum()),
            "topic_auroc": {s: round(float(roc_auc_score(y, v[mask])), 4) for s, v in tscore.items()},
            "topic_auroc_length_removed": {
                s: length_removed_auroc(y, v[mask], tscore["length"][mask][:, None])
                for s, v in tscore.items()
                if s != "length"
            },
            "prompt_auroc": {s: round(float(roc_auc_score(py, v[pm])), 4) for s, v in prompt.items()},
        }

    sources = np.array([r["source"] for r in load_jsonl(Path(config.source_removal_dir) / "prompts.jsonl")])
    assert len(sources) == len(x_93), f"{len(sources)=} {len(x_93)=}"
    rows = []
    for seed in (0, 1, 2):
        split = assign_splits(sources, seed)
        for fold, ref, ev, label in folds(sources, split, seed):
            test = split[ev] == "test"
            for method, s in scores(x_93[ref], x_93[ev]).items():
                rows.append(
                    {
                        "id": f"{method}|{fold}|{seed}",
                        "method": method,
                        "fold": fold,
                        "seed": seed,
                        "test_auroc": round(float(roc_auc_score(label[test], s[test])), 4),
                    }
                )
        LOGGER.info(f"removal seed {seed} done")
    write_jsonl(rows, out / "removal_scores.jsonl")
    methods = sorted({r["method"] for r in rows})
    removal = {
        m: {
            "mean_real_folds": round(
                float(np.mean([r["test_auroc"] for r in rows if r["method"] == m and r["fold"] != FAKE])), 4
            ),
            "fake_fold": round(
                float(np.mean([r["test_auroc"] for r in rows if r["method"] == m and r["fold"] == FAKE])), 4
            ),
            "per_fold": {
                f: round(float(np.mean([r["test_auroc"] for r in rows if r["method"] == m and r["fold"] == f])), 4)
                for f in sorted({r["fold"] for r in rows})
            },
        }
        for m in methods
    }
    decision = {
        "n_prompts": len(t_ids),
        "n_topics": len(topics),
        "dolma3_source_share": share,
        "n_removal_rows": len(rows),
        "labels": labels,
        "source_removal": removal,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(
        json.dumps(
            {"labels": labels, "removal": {m: (v["mean_real_folds"], v["fake_fold"]) for m, v in removal.items()}},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
