# delta_maha: delta-mahalanobis on traffic topics and on source_removal's source removal, plus stefan's response methods (base-minus-post logprob and a dolma-vs-dolci probe on olmo's own replies) on the user's labels.
# uv run --no-sync -m scripts.run_delta_maha_response_dmaha --dataset_path results/raw/261029_clean_traffic_detectors_clean_v1/clean_prompts.jsonl --source_removal_dir results/raw/261028_source_removal_v1 --group_removal_dir results/raw/261028_group_removal_v1 --dolma_path data/processed/repro_two_sided/dolma.jsonl --traffic_ranking_labels results/raw/261029_traffic_ranking_labels_v1/decision.json --blind_labels results/raw/261029_traffic_ranking_blind_labels_v1/decision.json --output_dir results/raw/261029_delta_maha_response_dmaha_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_source_removal import FAKE, SOURCES, assign_splits, folds
from scripts.run_traffic_ranking import TrafficRankingConfig, load_labels
from src.data.io import load_jsonl
from src.interp.features import encode_chat, encode_raw, encode_reply, extract
from src.interp.train_only_detectors import GpuMahalanobis, knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PROMPT_SCORES = ("dmaha", "maha_one_sided", "dknn", "knn_one_sided", "length", "random")
REPLY_SCORES = ("logprob_diff", "probe_reply", "reply_length")


@dataclass
class DeltaMahaConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    base_model_id: str = "allenai/Olmo-3-1025-7B"
    base_revision: str = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
    base_cache_dir: str = "/dev/shm/hf"
    source_removal_dir: str = ""
    group_removal_dir: str = ""
    dolma_path: str = ""
    traffic_ranking_labels: str = ""
    blind_labels: str = ""
    n_ref: int = 3000
    n_dolci_replies: int = 1000
    replies_per_topic: int = 10
    min_prompts: int = 5
    layer: int = 20
    k: int = 10
    max_tokens: int = 512
    max_new_tokens: int = 256
    gen_batch: int = 64
    batch_tokens: int = 32768
    text_chars: int = 2000
    seed: int = 0


def load_model(name: str, revision: str, cache_dir: str | None):
    return (
        AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.bfloat16, cache_dir=cache_dir)
        .cuda()
        .eval()
    )


@torch.no_grad()
def generate(model, tok, prompts: list[str], config: DeltaMahaConfig) -> list[str]:
    # greedy on-policy replies, left-padded batches sorted by length
    tok.padding_side = "left"
    rendered = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    enc = [tok(r, add_special_tokens=False)["input_ids"][-config.max_tokens :] for r in rendered]
    order = np.argsort([len(e) for e in enc])
    out = [""] * len(prompts)
    pad_id = tok.pad_token_id or 0
    for s in tqdm(range(0, len(order), config.gen_batch), desc="generate"):
        b = order[s : s + config.gen_batch]
        width = max(len(enc[i]) for i in b)
        ids = torch.full((len(b), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(b), width), dtype=torch.long)
        for row, i in enumerate(b):
            ids[row, width - len(enc[i]) :] = torch.tensor(enc[i])
            attn[row, width - len(enc[i]) :] = 1
        gen = model.generate(
            input_ids=ids.cuda(),
            attention_mask=attn.cuda(),
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        for row, i in enumerate(b):
            out[i] = tok.decode(gen[row, width:], skip_special_tokens=True).strip()
    return out


@torch.no_grad()
def reply_logprob(model, encoded: list[tuple[list[int], list[bool]]], batch_tokens: int, pad_id: int) -> np.ndarray:
    # mean log-probability of the masked reply tokens, each forced given everything before it
    out = np.zeros(len(encoded))
    order = np.argsort([len(e[0]) for e in encoded])[::-1]
    i = 0
    while i < len(order):
        width = len(encoded[order[i]][0])
        b = order[i : i + max(1, batch_tokens // width)]
        x = torch.full((len(b), width), pad_id, dtype=torch.long)
        m = torch.zeros((len(b), width), dtype=torch.bool)
        a = torch.zeros((len(b), width), dtype=torch.long)
        for row, j in enumerate(b):
            ids, mask = encoded[j]
            x[row, : len(ids)] = torch.tensor(ids)
            m[row, : len(ids)] = torch.tensor(mask)
            a[row, : len(ids)] = 1
        x, m, a = x.cuda(), m.cuda(), a.cuda()
        lp = torch.log_softmax(model(input_ids=x, attention_mask=a).logits[:, :-1].float(), -1)
        lp = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        target = m[:, 1:]
        for row, j in enumerate(b):
            out[j] = lp[row][target[row]].mean().item()
        i += len(b)
    return out


def dmaha_scores(ref_dolci: torch.Tensor, ref_dolma: torch.Tensor, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    # repro_two_sided's dmaha_full: raw mahalanobis to the dolci reference minus to the dolma reference
    to_dolci = GpuMahalanobis(ref_dolci)(x)
    return (to_dolci - GpuMahalanobis(ref_dolma)(x)).cpu().numpy(), to_dolci.cpu().numpy()


def topic_medians(score: np.ndarray, topic: np.ndarray, topics: np.ndarray) -> np.ndarray:
    return np.array([float(np.median(score[topic == t])) for t in topics])


def length_removed_auroc(y: np.ndarray, score: np.ndarray, lengths: np.ndarray) -> float:
    # auroc of the score after a linear fit on topic length features is taken out
    resid = score - LinearRegression().fit(lengths, score).predict(lengths)
    return round(float(roc_auc_score(y, resid)), 4)


def main() -> None:
    config = simple_parsing.parse(DeltaMahaConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    old = load_labels(TrafficRankingConfig())
    new = {int(r["id"][1:]): r["label"] for r in json.loads(Path(config.traffic_ranking_labels).read_text())["rows"]}
    blind = {int(r["id"][1:]): r["label"] for r in json.loads(Path(config.blind_labels).read_text())["rows"]}
    keys = {"old_117": old, "all_labels": {**old, **new, **blind}}
    labelled = {t for t, lab in keys["all_labels"].items() if lab in ("real_gap", "covered")}

    traffic = [r for r in load_jsonl(config.dataset_path) if r["topic"] in labelled]
    by_topic = defaultdict(list)
    for i, r in enumerate(traffic):
        by_topic[r["topic"]].append(i)
    topics = np.array(sorted(t for t, idx in by_topic.items() if len(idx) >= config.min_prompts))
    if config.num_tasks:
        topics = topics[: config.num_tasks]
        traffic = [traffic[i] for t in topics for i in by_topic[t]]
    topic = np.array([r["topic"] for r in traffic])
    reply_idx = np.array(
        [i for t in topics for i in rng.permutation(np.where(topic == t)[0])[: config.replies_per_topic]]
    )
    dolci_all = [
        r["prompt"] for r in load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl") if r["kind"] == "dolci"
    ]
    dolci_ref = dolci_all[: config.n_ref]
    n_dr = config.n_dolci_replies if not config.num_tasks else 100
    dolci_gen = dolci_all[config.n_ref : config.n_ref + n_dr]
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    source_removal = load_jsonl(Path(config.source_removal_dir) / "prompts.jsonl")
    if config.num_tasks:
        source_removal = [r for s in SOURCES for r in [x for x in source_removal if x["source"] == s][:60]]
    LOGGER.info(f"{len(traffic)=} {len(topics)=} {len(reply_idx)=} {len(dolci_gen)=} {len(source_removal)=}")

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    pad_id = tok.pad_token_id or 0

    def clip(t: str) -> str:
        return t[: 4 * config.text_chars]

    # post-trained model: prompt activations for delta-mahalanobis, then on-policy replies and their logprobs
    model = load_model(config.model_id, config.model_revision, None)
    prompt_texts = [r["prompt"] for r in traffic] + dolci_ref + dolma + [r["prompt"] for r in source_removal]
    feats = extract(
        model,
        [encode_chat(tok, clip(t), config.max_tokens) for t in prompt_texts],
        pad_id,
        max_tokens=config.batch_tokens,
        layers=[config.layer],
    )
    x = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
    n_tok = feats["n_content_tokens"][: len(traffic)]
    n_t, n_d, n_m = len(traffic), len(traffic) + len(dolci_ref), len(traffic) + len(dolci_ref) + len(dolma)
    x_t, x_dolci, x_dolma, x_93 = x[:n_t], x[n_t:n_d], x[n_d:n_m], x[n_m:]
    np.savez(
        out / f"olmo_layer{config.layer}_cmean.npz",
        traffic=x_t.cpu().numpy().astype(np.float16),
        traffic_ids=np.array([r["id"] for r in traffic]),
        source_removal=x_93.cpu().numpy().astype(np.float16),
        source_removal_ids=np.array([r["id"] for r in source_removal]),
    )

    gen_prompts = [clip(traffic[i]["prompt"]) for i in reply_idx] + [clip(t) for t in dolci_gen]
    replies = generate(model, tok, gen_prompts, config)
    n_empty = sum(not r for r in replies)
    LOGGER.info(f"{n_empty=} of {len(replies)} replies")
    replies = [r or "." for r in replies]
    write_jsonl(
        [
            {"id": traffic[i]["id"] if j < len(reply_idx) else f"dolci{j - len(reply_idx)}", "prompt": p, "reply": r}
            for j, (i, p, r) in enumerate(zip(list(reply_idx) + [-1] * len(dolci_gen), gen_prompts, replies))
        ],
        out / "replies.jsonl",
    )
    conv = [
        encode_reply(tok, [{"role": "user", "content": p}, {"role": "assistant", "content": r}], 2 * config.max_tokens)
        for p, r in zip(gen_prompts, replies)
    ]
    reply_tokens = np.array([sum(m) for _, m in conv])
    lp_post = reply_logprob(model, conv, config.batch_tokens, pad_id)
    # probe features: the post-trained model's activations on the reply text alone and on dolma passages, as raw text
    raw = [encode_raw(tok, clip(t), config.max_tokens) for t in replies + dolma]
    raw = [(ids, m) if ids else ([tok.eos_token_id], [True]) for ids, m in raw]
    raw_feats = extract(model, raw, pad_id, max_tokens=config.batch_tokens, layers=[config.layer])["cmean"][:, 0]
    del model
    torch.cuda.empty_cache()

    # base model: the same token ids, so the logprob difference is the model change alone
    model = load_model(config.base_model_id, config.base_revision, config.base_cache_dir)
    lp_base = reply_logprob(model, conv, config.batch_tokens, pad_id)
    del model
    torch.cuda.empty_cache()

    n_r = len(reply_idx)
    raw_feats = raw_feats.astype(np.float32)
    dolci_rows = np.arange(n_r, len(replies))
    probe_train = dolci_rows[: len(dolci_rows) // 2]
    dolci_held = dolci_rows[len(dolci_rows) // 2 :]
    dolma_rows = np.arange(len(replies), len(replies) + len(dolma))
    tr = np.r_[probe_train, dolma_rows[: len(dolma_rows) // 2]]
    y_tr = np.r_[np.zeros(len(probe_train)), np.ones(len(dolma_rows) // 2)]
    scaler = StandardScaler().fit(raw_feats[tr])
    probe = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(raw_feats[tr]), y_tr)
    p_dolma = probe.predict_proba(scaler.transform(raw_feats))[:, 1]
    lp_diff = lp_base - lp_post

    held_dolma = dolma_rows[len(dolma_rows) // 2 :]
    sanity = {
        "probe_heldout_dolma_vs_dolci_replies": round(
            float(
                roc_auc_score(
                    np.r_[np.zeros(len(dolci_held)), np.ones(len(held_dolma))], p_dolma[np.r_[dolci_held, held_dolma]]
                )
            ),
            4,
        ),
        "traffic_vs_dolci_replies": {
            name: round(
                float(
                    roc_auc_score(np.r_[np.ones(n_r), np.zeros(len(dolci_held))], s[np.r_[np.arange(n_r), dolci_held]])
                ),
                4,
            )
            for name, s in (("logprob_diff", lp_diff), ("probe_reply", p_dolma), ("reply_length", np.log(reply_tokens)))
        },
        "empty_replies": n_empty,
        "replies_hit_max_new_tokens": int(
            sum(len(tok(r, add_special_tokens=False)["input_ids"]) >= config.max_new_tokens for r in replies)
        ),
    }

    # prompt scores on traffic
    dmaha, maha1 = dmaha_scores(x_dolci, x_dolma, x_t)
    knn_dolci = knn_distances(x_dolci, x_t, (config.k,))[config.k].cpu().numpy()
    knn_dolma = knn_distances(x_dolma, x_t, (config.k,))[config.k].cpu().numpy()
    prompt_scores = {
        "dmaha": dmaha,
        "maha_one_sided": maha1,
        "dknn": knn_dolci - knn_dolma,
        "knn_one_sided": knn_dolci,
        "length": np.log(n_tok),
        "random": rng.random(n_t),
    }
    reply_scores = {
        "logprob_diff": lp_diff[:n_r],
        "probe_reply": p_dolma[:n_r],
        "reply_length": np.log(reply_tokens[:n_r]),
    }
    write_jsonl(
        [
            {"id": r["id"], "topic": int(r["topic"]), **{s: round(float(v[i]), 4) for s, v in prompt_scores.items()}}
            for i, r in enumerate(traffic)
        ],
        out / "prompt_scores.jsonl",
    )
    write_jsonl(
        [
            {
                "id": traffic[i]["id"],
                "topic": int(topic[i]),
                **{s: round(float(v[j]), 4) for s, v in reply_scores.items()},
            }
            for j, i in enumerate(reply_idx)
        ],
        out / "reply_scores.jsonl",
    )

    tscore = {s: topic_medians(v, topic, topics) for s, v in prompt_scores.items()}
    tscore |= {s: topic_medians(v, topic[reply_idx], topics) for s, v in reply_scores.items()}
    len_feats = np.c_[tscore["length"], tscore["reply_length"]]
    results = {}
    for name, human in keys.items():
        mask = np.array([human.get(t) in ("real_gap", "covered") for t in topics])
        y = np.array([human.get(t) == "real_gap" for t in topics[mask]], dtype=int)
        results[name] = {
            "n_real": int(y.sum()),
            "n_covered": int(len(y) - y.sum()),
            "auroc": {s: round(float(roc_auc_score(y, v[mask])), 4) for s, v in tscore.items()},
            "auroc_prompt_and_reply_length_removed": {
                s: length_removed_auroc(y, v[mask], len_feats[mask])
                for s, v in tscore.items()
                if s not in ("length", "reply_length")
            },
        }
    write_jsonl(
        [
            {"id": f"c{t}", "label": keys["all_labels"].get(t), **{s: round(float(v[i]), 4) for s, v in tscore.items()}}
            for i, t in enumerate(topics)
        ],
        out / "topic_scores.jsonl",
    )

    # source_removal source removal with delta-mahalanobis against the same dolma reference
    sources = np.array([r["source"] for r in source_removal])
    removal_rows = []
    for seed in (0, 1, 2):
        split = assign_splits(sources, seed)
        for fold, ref, ev, label in folds(sources, split, seed):
            test = split[ev] == "test"
            d, m1 = dmaha_scores(x_93[ref], x_dolma, x_93[ev])
            k1 = knn_distances(x_93[ref], x_93[ev], (50,))[50].cpu().numpy()
            for method, s in (("dmaha", d), ("maha_one_sided", m1), ("knn50_one_sided", k1)):
                removal_rows.append(
                    {
                        "id": f"{method}|{fold}|{seed}",
                        "method": method,
                        "fold": fold,
                        "seed": seed,
                        "test_auroc": round(float(roc_auc_score(label[test], s[test])), 4),
                    }
                )
    write_jsonl(removal_rows, out / "removal_scores.jsonl")
    removal = {
        method: {
            "mean_real_folds": round(
                float(np.mean([r["test_auroc"] for r in removal_rows if r["method"] == method and r["fold"] != FAKE])),
                4,
            ),
            "fake_fold": round(
                float(np.mean([r["test_auroc"] for r in removal_rows if r["method"] == method and r["fold"] == FAKE])),
                4,
            ),
        }
        for method in ("dmaha", "maha_one_sided", "knn50_one_sided")
    }

    n_nan = int(sum(np.isnan(v).sum() for v in list(prompt_scores.values()) + list(reply_scores.values())))
    decision = {
        "n_prompts": n_t,
        "n_topics": len(topics),
        "n_replies": n_r,
        "n_nan": n_nan,
        "sanity": sanity,
        "labels": results,
        "source_removal": removal,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
