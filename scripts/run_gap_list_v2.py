# gap_list_v2: gap list v2 with NAME_n swapped, head truncation, length-conditional scores, template units and a size-matched line.
# uv run --no-sync -m scripts.run_gap_list_v2 --phase extract --gap_list_dir results/raw/261029_gap_list_v1 --group_removal_dir results/raw/261028_group_removal_v1 --dolma_path data/processed/repro_two_sided/dolma.jsonl --output_dir results/raw/261030_gap_list_v2 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0
# uv run --no-sync -m scripts.run_gap_list_v2 --phase score --gap_list_dir results/raw/261029_gap_list_v1 --output_dir results/raw/261030_gap_list_v2 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.generation.prompt_kinds import NAME_TOKEN, swap_placeholders
from src.interp.features import encode_chat_head, extract
from src.interp.kmeans import spherical_kmeans
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
LEN_BINS = [0, 16, 32, 64, 128, 256, 10**9]
UNIT_BINS = [0, 5, 10, 20, 10**9]
QS = (0.9, 0.95, 0.99)


@dataclass
class GapListV2Config(Config):
    phase: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    gap_list_dir: str = ""
    group_removal_dir: str = ""
    dolma_path: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    n_mix: int = 200000
    n_ref: int = 3000
    n_heldout: int = 30000
    heldout_seed: int = 1
    n_null_groups: int = 1500
    layer: int = 20
    k: int = 100
    max_tokens: int = 512
    text_chars: int = 2000
    batch_tokens: int = 32768
    prefix_chars: int = 40
    n_boot: int = 1000
    seed: int = 0


def template_key(text: str, chars: int) -> str:
    return " ".join(text.lower().split())[:chars]


def extract_phase(config: GapListV2Config, out: Path) -> None:
    traffic = load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")
    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=config.n_mix, replace=False))
    outside = np.setdiff1d(np.arange(len(mix_text)), sample)
    held = np.sort(np.random.default_rng(config.heldout_seed).choice(outside, size=config.n_heldout, replace=False))
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)

    # dolma reference built exactly as in chat_removal_loop
    dolci_ref = [
        r["prompt"] for r in load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl") if r["kind"] == "dolci"
    ][: config.n_ref]
    dolci_len = np.array(
        [min(len(x), config.max_tokens) for x in tok(dolci_ref, add_special_tokens=False)["input_ids"]]
    )
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: config.n_ref]
    ids = tok([t[: 4 * config.text_chars] for t in dolma], add_special_tokens=False)["input_ids"]
    cut = np.random.default_rng(config.seed).choice(dolci_len, size=len(ids))
    dolma_m = [tok.decode(x[:n]) for x, n in zip(ids, cut, strict=True)]

    pools = {
        "traffic": [r["prompt"] for r in traffic],
        "mix": [mix_text[i] for i in sample],
        "dolma": dolma_m,
        "held": [mix_text[i] for i in held],
    }
    del mix_text
    texts = [swap_placeholders(t[: 4 * config.text_chars]) for p in pools.values() for t in p]
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    encoded = [encode_chat_head(tok, t, config.max_tokens) for t in texts]
    LOGGER.info(f"{len(texts)=} {[len(p) for p in pools.values()]=} {max(len(e[0]) for e in encoded)=}")
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=[config.layer])[
        "cmean"
    ][:, 0]
    np.save(config.feats_path, feats)
    np.savez(
        out / "rows.npz",
        lengths=lengths,
        sizes=np.array([len(p) for p in pools.values()]),
        held=held,
    )
    write_jsonl(
        [{"id": f"mix{int(i)}", "text": t[:300]} for i, t in zip(held, pools["held"], strict=True)],
        out / "heldout_texts.jsonl",
    )
    print(f"Features saved to {config.feats_path}, rows to {out / 'rows.npz'}")


def length_pct(score: np.ndarray, length: np.ndarray, ref: np.ndarray, ref_len: np.ndarray) -> np.ndarray:
    # share of held-out dolci prompts in the same token-length bin scoring below each prompt
    b, rb = np.digitize(length, LEN_BINS[1:-1], right=True), np.digitize(ref_len, LEN_BINS[1:-1], right=True)
    out = np.empty(len(score))
    for i in range(len(LEN_BINS) - 1):
        r = np.sort(ref[rb == i])
        out[b == i] = np.searchsorted(r, score[b == i]) / len(r)
    return out


def unit_scores(pct: np.ndarray, keys: list[str]) -> np.ndarray:
    units = defaultdict(list)
    for p, k in zip(pct, keys, strict=True):
        units[k].append(p)
    return np.array([np.median(v) for v in units.values()])


def score_phase(config: GapListV2Config, out: Path) -> None:
    traffic = load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")
    v1_topics = load_jsonl(Path(config.gap_list_dir) / "topics.jsonl")
    rows = np.load(out / "rows.npz")
    n_t, n_mix, n_ref, n_h = rows["sizes"].tolist()
    lengths = rows["lengths"]
    feats = np.load(config.feats_path)
    assert len(feats) == n_t + n_mix + n_ref + n_h and n_t == len(traffic)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix = x[:n_t], x[n_t : n_t + n_mix]
    xdolma, xh = x[n_t + n_mix : n_t + n_mix + n_ref], x[n_t + n_mix + n_ref :]
    sd = xmix.std(0) + 1e-6

    def delta(q: torch.Tensor) -> np.ndarray:
        to_dolci = knn_distances(xmix / sd, q / sd, (config.k,), scale=False)[config.k]
        to_dolma = knn_distances(xdolma, q, (config.k,))[config.k]
        return (to_dolci - to_dolma).cpu().numpy()

    s_t, s_h = delta(xt), delta(xh)
    len_t, len_h = lengths[:n_t], lengths[n_t + n_mix + n_ref :]
    pct_t, pct_h = length_pct(s_t, len_t, s_h, len_h), length_pct(s_h, len_h, s_h, len_h)

    prompts = [r["prompt"] for r in traffic]
    topic = np.array([r["topic"] for r in traffic])
    v1 = np.array([r["delta_knn"] for r in traffic])
    has_name = np.array([bool(NAME_TOKEN.search(p)) for p in prompts])
    keys = [template_key(p, config.prefix_chars) for p in prompts]
    held_texts = [r["text"] for r in load_jsonl(out / "heldout_texts.jsonl")]
    keys_h = [template_key(t, config.prefix_chars) for t in held_texts]

    # null: held-out dolci grouped into topic-like clusters in the detector's own space
    labels, _ = spherical_kmeans(xh / sd, config.n_null_groups, n_init=3, seed=config.seed)
    labels = labels.cpu().numpy()
    del x, xt, xh
    torch.cuda.empty_cache()
    null = []
    for g in range(config.n_null_groups):
        idx = np.where(labels == g)[0]
        if len(idx):
            u = unit_scores(pct_h[idx], [keys_h[i] for i in idx])
            null.append((len(u), float(np.median(u))))
    null_n, null_s = np.array([a for a, _ in null]), np.array([b for _, b in null])
    null_bin = np.digitize(null_n, UNIT_BINS[1:-1], right=True)
    cuts = {q: [round(float(np.quantile(null_s[null_bin == i], q)), 4) for i in range(len(UNIT_BINS) - 1)] for q in QS}
    rng = np.random.default_rng(config.seed)
    indep = {}

    def indep_cut(n: int, q: float) -> float:
        # medians of n random held-out prompts, no clustering
        if n not in indep:
            indep[n] = np.median(rng.choice(pct_h, size=(2000, n)), axis=1)
        return float(np.quantile(indep[n], q))

    topics = np.unique(topic)
    units = {}
    for t in topics:
        idx = np.where(topic == t)[0]
        units[int(t)] = unit_scores(pct_t[idx], [keys[i] for i in idx])
    score = np.array([np.median(units[int(t)]) for t in topics])
    raw = np.array([np.median(s_t[topic == t]) for t in topics])
    boot = np.stack(
        [np.median(rng.choice(units[int(t)], size=(config.n_boot, len(units[int(t)]))), axis=1) for t in topics], 1
    )
    boot_rank = (-boot).argsort(1).argsort(1) + 1
    order = np.argsort(-score)
    v1_rank = {r["topic"]: r["rank"] for r in v1_topics}
    out_rows = []
    for rank, i in enumerate(order, start=1):
        t = int(topics[i])
        idx = np.where(topic == t)[0]
        n_u = len(units[t])
        ub = int(np.digitize(n_u, UNIT_BINS[1:-1], right=True))
        out_rows.append(
            {
                "id": f"c{t}",
                "rank": rank,
                "topic": t,
                "n_prompts": len(idx),
                "n_units": n_u,
                "score": round(float(score[i]), 4),
                "median_delta_knn": round(float(raw[i]), 4),
                "rank_ci90": [int(x) for x in np.percentile(boot_rank[:, i], [5, 95])],
                "v1_rank": v1_rank[t],
                **{f"flag_{q}": bool(score[i] > cuts[q][ub]) for q in QS},
                **{f"flag_indep_{q}": bool(score[i] > indep_cut(n_u, q)) for q in QS},
                "median_tokens": float(np.median(len_t[idx])),
                "name_share": round(float(has_name[idx].mean()), 4),
                "template_share": round(1 - n_u / len(idx), 4),
                "examples": [prompts[j][:300] for j in idx[np.argsort(-pct_t[idx])][:5]],
            }
        )
    write_jsonl(out_rows, out / "topics.jsonl")
    write_jsonl(
        [
            {
                "id": r["id"],
                "topic": r["topic"],
                "delta_knn": round(float(s), 4),
                "length_pct": round(float(p), 4),
                "n_tokens": int(n),
                "v1_delta_knn": r["delta_knn"],
            }
            for r, s, p, n in zip(traffic, s_t, pct_t, len_t, strict=True)
        ],
        out / "prompt_scores.jsonl",
    )

    def top_stats(sel: list[dict]) -> dict:
        return {
            "median_tokens": float(np.median([r["median_tokens"] for r in sel])),
            "name_share": round(float(np.mean([r["name_share"] for r in sel])), 4),
            "template_share": round(float(np.mean([r["template_share"] for r in sel])), 4),
        }

    v1_sorted = sorted(v1_topics, key=lambda r: r["rank"])
    by_topic = {r["topic"]: r for r in out_rows}
    med_tok = np.array([by_topic[int(t)]["median_tokens"] for t in topics])
    name_sh = np.array([by_topic[int(t)]["name_share"] for t in topics])
    templ_sh = np.array([by_topic[int(t)]["template_share"] for t in topics])
    v1_med = np.array([{r["topic"]: r["median_delta_knn"] for r in v1_topics}[int(t)] for t in topics])

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        return round(float(spearmanr(a, b).statistic), 4)

    decision = {
        "n_traffic": n_t,
        "n_heldout": n_h,
        "n_topics": len(topics),
        "n_nan": int(np.isnan(s_t).sum() + np.isnan(s_h).sum()),
        "heldout_per_length_bin": np.bincount(np.digitize(len_h, LEN_BINS[1:-1], right=True)).tolist(),
        "traffic_per_length_bin": np.bincount(np.digitize(len_t, LEN_BINS[1:-1], right=True)).tolist(),
        "null_groups_per_unit_bin": np.bincount(null_bin, minlength=len(UNIT_BINS) - 1).tolist(),
        "cuts_by_unit_bin": {str(q): v for q, v in cuts.items()},
        "n_flagged": {str(q): sum(r[f"flag_{q}"] for r in out_rows) for q in QS},
        "n_flagged_indep_null": {str(q): sum(r[f"flag_indep_{q}"] for r in out_rows) for q in QS},
        "swap_effect": {
            "mean_v2_minus_v1_with_name": round(float((s_t - v1)[has_name].mean()), 4),
            "mean_v2_minus_v1_without_name": round(float((s_t - v1)[~has_name].mean()), 4),
            "n_with_name": int(has_name.sum()),
        },
        "spearman_prompt": {
            "raw_v1_vs_tokens": rho(v1, len_t),
            "raw_v2_vs_tokens": rho(s_t, len_t),
            "pct_v2_vs_tokens": rho(pct_t, len_t),
            "raw_v1_vs_name": rho(v1, has_name),
            "raw_v2_vs_name": rho(s_t, has_name),
            "pct_v2_vs_name": rho(pct_t, has_name),
        },
        "spearman_topic": {
            "v1_vs_tokens": rho(v1_med, med_tok),
            "v2_vs_tokens": rho(score, med_tok),
            "v1_vs_name": rho(v1_med, name_sh),
            "v2_vs_name": rho(score, name_sh),
            "v1_vs_template": rho(v1_med, templ_sh),
            "v2_vs_template": rho(score, templ_sh),
            "v1_vs_v2": rho(v1_med, score),
        },
        "overlap_v1_v2": {
            n: len({r["topic"] for r in v1_sorted[:n]} & {r["topic"] for r in out_rows[:n]}) for n in (20, 100)
        },
        "top100_v1": top_stats([by_topic[r["topic"]] for r in v1_sorted[:100]]),
        "top100_v2": top_stats(out_rows[:100]),
        "all_topics": top_stats(out_rows),
        "top30": [
            {k: r[k] for k in ("rank", "topic", "score", "rank_ci90", "v1_rank", "n_units", "median_tokens")}
            | {"example": r["examples"][0][:150]}
            for r in out_rows[:30]
        ],
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'topics.jsonl'} and {out / 'decision.json'}")
    print(json.dumps({k: v for k, v in decision.items() if k != "top30"}, indent=1))


def main() -> None:
    config = simple_parsing.parse(GapListV2Config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=config.phase == "score")
    setup_logging(out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    if config.phase == "extract":
        write_config_json(config, out)
        extract_phase(config, out)
    elif config.phase == "score":
        score_phase(config, out)
    else:
        raise ValueError(f"unknown phase {config.phase!r}")


if __name__ == "__main__":
    main()
