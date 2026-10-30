# negatives_test: stefan's must-not-flag test: training-only detectors scored on unseen negative sets (code, multilingual, tool use, safety, maths) against planted legal and medical positives, with a roc curve and a traffic-set flag line.
# uv run --no-sync -m scripts.run_negatives_test_negatives --group_removal_dir results/raw/261028_group_removal_v1 --clean_path results/raw/261029_clean_traffic_detectors_clean_v1/clean_prompts.jsonl --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --dolma_path data/processed/repro_two_sided/dolma.jsonl --xstest_csv data/raw/xstest/xstest_v2_completions_gpt4.csv --output_dir results/raw/261029_negatives_test_negatives_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simple_parsing
import torch
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_source_removal import assign_splits
from scripts.run_combined_score_combined import length_matched, nearest_cos
from src.data.gap_pool import load_mix
from src.data.io import load_jsonl
from src.data.pools import normalise
from src.interp.features import encode_chat, extract
from src.interp.train_only_detectors import GpuMahalanobis, knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.plotting import AQUA, BLUE, NEUTRAL, ORANGE, finish, header, style, tidy
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"NAME_\d+")
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MATCHED = {
    "code_other": ("Dolci Instruct Python Algorithms", "Evol CodeAlpaca"),
    "maths_other": ("Tulu 3 Persona MATH", "Tulu 3 Persona GSM", "OpenMathInstruct 2"),
    "multilingual_other": ("Aya",),
    "safety_other": ("WildGuardMix", "WildJailbreak"),
}
MATCHED_DETECTORS = ("delta_maha", "delta_knn", "knn_one_sided", "minilm", "tfidf", "combined")
NEGATIVES = ("code_other", "multilingual_other", "tool_other", "safety_other", "maths_other")
POSITIVES = ("legal", "medical")
DETECTORS = ("delta_maha", "delta_knn", "knn_one_sided", "minilm", "tfidf", "combined_lm", "length")


@dataclass
class NegativesTestConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    group_removal_dir: str = ""
    clean_path: str = ""
    pipeline_million_dir: str = ""
    dolma_path: str = ""
    xstest_csv: str = ""
    source_removal_dir: str = ""
    n_ref: int = 3000
    n_heldout: int = 1000
    n_traffic: int = 5000
    copy_cos: float = 0.95
    line_pct: float = 95.0
    layer: int = 20
    k: int = 10
    max_tokens: int = 512
    batch_tokens: int = 32768
    text_chars: int = 2000
    tfidf_chars: int = 1000
    n_bins: int = 10
    seed: int = 0


def load_sets(config: NegativesTestConfig, rng: np.random.Generator) -> dict[str, list[str]]:
    maths = [
        r["text"] for r in load_jsonl("data/processed/owl_stage1_trained/gen_pool.jsonl") if r["family"] == "maths"
    ]
    maths += [r["text"] for r in load_jsonl("data/processed/owl_stage1_trained/test.jsonl") if r["family"] == "maths"]
    dolci = [r["prompt"] for r in load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl") if r["kind"] == "dolci"]
    clean = [r["prompt"] for r in load_jsonl(config.clean_path)]
    sets = {
        "code_other": [r["text"] for r in load_jsonl("data/processed/repro_controls/magicoder.jsonl")],
        "multilingual_other": [r["text"] for r in load_jsonl("data/processed/repro_controls/bactrian.jsonl")],
        "tool_other": [r["text"] for r in load_jsonl("data/processed/multiturn_tooluse/tool_alone.jsonl")],
        "safety_other": pd.read_csv(config.xstest_csv)["prompt"].drop_duplicates().tolist(),
        "maths_other": list(dict.fromkeys(maths)),
        "legal": [r["text"] for r in load_jsonl("data/processed/density_ratio_needles/needles_legal.jsonl")],
        "medical": [r["text"] for r in load_jsonl("data/processed/density_ratio_needles/needles_medical.jsonl")],
        "heldout_dolci": dolci[config.n_ref : config.n_ref + config.n_heldout],
        "traffic": [clean[i] for i in rng.choice(len(clean), size=min(config.n_traffic, len(clean)), replace=False)],
        "ref_dolci": dolci[: config.n_ref],
    }
    if config.num_tasks:
        sets = {k: v[: config.num_tasks] for k, v in sets.items()}
    return sets


def main() -> None:
    config = simple_parsing.parse(NegativesTestConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    sets = load_sets(config, rng)
    dolma = [r["text"] for r in load_jsonl(config.dolma_path)][: len(sets["ref_dolci"])]
    eval_names = [n for n in sets if n != "ref_dolci"]
    texts = [t for n in eval_names for t in sets[n]]
    names = np.array([n for n in eval_names for _ in sets[n]])

    # training references for minilm and words: the whole mix minus anything matching an evaluated prompt
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    key_norms = {normalise(t) for n in eval_names if n not in ("heldout_dolci",) for t in sets[n]} | {
        normalise(t) for t in sets["heldout_dolci"]
    }
    m_norm = [normalise(t) for t in m_text]
    keep = np.array([t not in key_norms for t in m_norm])
    mix_norms = set(m_norm)
    exact = np.array([normalise(t) in mix_norms for t in texts])

    e93 = load_jsonl(Path(config.source_removal_dir) / "prompts.jsonl")
    src93 = np.array([r["source"] for r in e93])
    ref93 = np.where(assign_splits(src93, 0) == "ref")[0]
    if config.num_tasks:
        ref93 = ref93[:: max(1, len(ref93) // (config.num_tasks * 12))]
    ref93_text, ref93_src = [e93[i]["prompt"] for i in ref93], src93[ref93]

    st = SentenceTransformer(MINILM, device="cuda")
    r93_emb = torch.tensor(
        st.encode([t[: c37["embed_chars"]] for t in ref93_text], batch_size=256, normalize_embeddings=True),
        device="cuda",
    ).half()
    q = st.encode([t[: c37["embed_chars"]] for t in texts], batch_size=256, normalize_embeddings=True)
    del st
    q = torch.tensor(q, device="cuda").half()
    ref = torch.nn.functional.normalize(torch.from_numpy(m_np[keep]).cuda().float(), dim=1).half()
    # near-copy check against the full mix (before removal), then distance to the cleaned mix
    full = torch.nn.functional.normalize(torch.from_numpy(m_np).cuda().float(), dim=1).half()
    near_full = nearest_cos(q, full)
    del full
    minilm = 1 - nearest_cos(q, ref)
    del ref, m_np
    torch.cuda.empty_cache()
    is_copy = exact | (near_full >= config.copy_cos)
    is_copy[np.isin(names, ("heldout_dolci", "traffic"))] = False

    hv = HashingVectorizer(
        preprocessor=lambda t: PLACEHOLDER.sub(" ", t).lower(),
        ngram_range=(1, 2),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
    )
    tf = TfidfTransformer(sublinear_tf=True)
    mix_x = tf.fit_transform(hv.transform([m_text[i][: config.tfidf_chars] for i in np.where(keep)[0]]))
    e_h = hv.transform([t[: config.tfidf_chars] for t in texts]).tocsr()
    e_x = tf.transform(e_h)
    r93_h = hv.transform([t[: config.tfidf_chars] for t in ref93_text]).tocsr()
    tfidf = np.concatenate(
        [
            1 - np.asarray((e_x[i : i + 500] @ mix_x.T).max(axis=1).todense()).ravel()
            for i in range(0, e_x.shape[0], 500)
        ]
    )
    del mix_x, m_text

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    all_texts = texts + sets["ref_dolci"] + dolma + ref93_text
    encoded = [encode_chat(tok, t[: 4 * config.text_chars], config.max_tokens) for t in all_texts]
    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    feats = extract(model, encoded, tok.pad_token_id or 0, max_tokens=config.batch_tokens, layers=[config.layer])
    del model
    torch.cuda.empty_cache()
    n_e, n_r = len(texts), len(texts) + len(sets["ref_dolci"])
    n_m = n_r + len(dolma)
    x93 = torch.tensor(feats["cmean"][n_m:, 0], device="cuda").float()
    x_dolma_raw = torch.tensor(feats["cmean"][n_r:n_m, 0], device="cuda").float()
    x_eval_raw = torch.tensor(feats["cmean"][:n_e, 0], device="cuda").float()
    x = torch.tensor(feats["cmean"][:, 0], device="cuda").float()
    # dolma is x[n_r:n_m]; source_removal reference prompts follow it and must not enter the dolma side
    d_maha = (GpuMahalanobis(x[n_e:n_r])(x[:n_e]) - GpuMahalanobis(x[n_r:n_m])(x[:n_e])).cpu().numpy()
    xs = x / (x[n_e:n_m].std(0) + 1e-6)
    to_dolci = knn_distances(xs[n_e:n_r], xs[:n_e], (config.k,), scale=False)[config.k].cpu().numpy()
    to_dolma = knn_distances(xs[n_r:n_m], xs[:n_e], (config.k,), scale=False)[config.k].cpu().numpy()
    log_len = np.log(feats["n_content_tokens"][:n_e])

    scores = {
        "delta_maha": d_maha,
        "delta_knn": to_dolci - to_dolma,
        "knn_one_sided": to_dolci,
        "minilm": minilm,
        "tfidf": tfidf,
        "length": log_len,
    }
    lm = {k: length_matched(scores[k], log_len, config.n_bins) for k in ("minilm", "tfidf", "delta_knn")}
    scores["combined_lm"] = (lm["minilm"] + lm["tfidf"] + lm["delta_knn"]) / 3
    write_jsonl(
        [
            {
                "id": f"{names[i]}_{i}",
                "set": names[i],
                "near_copy": bool(is_copy[i]),
                **{d: round(float(scores[d][i]), 4) for d in DETECTORS},
            }
            for i in range(n_e)
        ],
        out / "scores.jsonl",
    )

    use = ~is_copy
    pos = np.isin(names, POSITIVES) & use
    neg = np.isin(names, NEGATIVES) & use
    pn = pos | neg
    results, roc_rows = {}, []
    for d in DETECTORS:
        s = scores[d]
        line = float(np.percentile(s[names == "traffic"], config.line_pct))
        fpr, tpr, _ = roc_curve(pos[pn].astype(int), s[pn])
        roc_rows += [[d, round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]
        results[d] = {
            "auroc_pos_vs_neg": round(float(roc_auc_score(pos[pn].astype(int), s[pn])), 4),
            "line_traffic_p95": round(line, 4),
            "flag_rate_above_line": {
                n: round(float((s[(names == n) & use] > line).mean()), 4)
                for n in (*NEGATIVES, *POSITIVES, "heldout_dolci", "traffic")
            },
            "auroc_vs_heldout_dolci": {
                n: round(
                    float(
                        roc_auc_score(
                            np.r_[np.ones(((names == n) & use).sum()), np.zeros((names == "heldout_dolci").sum())],
                            np.r_[s[(names == n) & use], s[names == "heldout_dolci"]],
                        )
                    ),
                    4,
                )
                for n in (*NEGATIVES, *POSITIVES)
            },
        }

    # matched version: the same unseen prompts against a reference with and without their kind's dolci sources
    matched, matched_roc = {}, []
    for kind, removed in MATCHED.items():
        qi = np.where((names == kind) & use)[0]
        per = {d: [] for d in MATCHED_DETECTORS}
        for cond, keep_ref in (("covered", np.ones(len(ref93), bool)), ("removed", ~np.isin(ref93_src, removed))):
            r = np.where(keep_ref)[0]
            mini = 1 - nearest_cos(q[qi], r93_emb[r])
            tf93 = TfidfTransformer(sublinear_tf=True).fit(r93_h[r])
            rx, qx = tf93.transform(r93_h[r]), tf93.transform(e_h[qi])
            words = 1 - np.asarray((qx @ rx.T).max(axis=1).todense()).ravel()
            xr, xq = x93[r], x_eval_raw[qi]
            dm = (GpuMahalanobis(xr)(xq) - GpuMahalanobis(x_dolma_raw)(xq)).cpu().numpy()
            sd = torch.cat([xr, x_dolma_raw]).std(0) + 1e-6
            kr = knn_distances(xr / sd, xq / sd, (config.k,), scale=False)[config.k].cpu().numpy()
            kd = knn_distances(x_dolma_raw / sd, xq / sd, (config.k,), scale=False)[config.k].cpu().numpy()
            for d, v in (
                ("delta_maha", dm),
                ("delta_knn", kr - kd),
                ("knn_one_sided", kr),
                ("minilm", mini),
                ("tfidf", words),
            ):
                per[d].append(v)
        y = np.r_[np.zeros(len(qi)), np.ones(len(qi))]
        pooled = {d: np.concatenate(per[d]) for d in MATCHED_DETECTORS if d != "combined"}
        pooled["combined"] = np.mean([rankdata(pooled[d]) / len(y) for d in ("minilm", "tfidf", "delta_knn")], axis=0)
        matched[kind] = {"n_prompts": len(qi), "removed_sources": list(removed), "auroc_removed_vs_covered": {}}
        for d, v in pooled.items():
            matched[kind]["auroc_removed_vs_covered"][d] = round(float(roc_auc_score(y, v)), 4)
            fpr, tpr, _ = roc_curve(y, v)
            matched_roc += [[kind, d, round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]
        LOGGER.info(f"{kind=} {matched[kind]['auroc_removed_vs_covered']}")

    style()
    fig, axes = plt.subplots(1, len(MATCHED), figsize=(4.2 * len(MATCHED), 4.4), sharey=True)
    mcol = {
        "delta_maha": ORANGE,
        "delta_knn": BLUE,
        "knn_one_sided": AQUA,
        "combined": "#7a5195",
        "minilm": "#444444",
        "tfidf": "#999999",
    }
    for ax, kind in zip(axes, MATCHED):
        for d in MATCHED_DETECTORS:
            r = [(a, b) for k, dd, a, b in matched_roc if k == kind and dd == d]
            ax.plot(
                [a for a, _ in r],
                [b for _, b in r],
                color=mcol[d],
                label=f"{d} {matched[kind]['auroc_removed_vs_covered'][d]:.2f}",
            )
        ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--")
        ax.set_xlabel("False positive rate (same prompts, kind covered)")
        ax.legend(fontsize=7, loc="lower right")
        tidy(ax, grid_axis="both", drop_left=False)
        ax.set_title(kind.replace("_other", ""), fontsize=10)
    axes[0].set_ylabel("True positive rate (same prompts, kind removed)")
    header(
        fig,
        "Matched negatives: the same unseen prompts scored with their kind in and out of the reference\nOlmo-3-7B-DPO layer 20; reference = source_removal's 24,000 source-labelled Dolci prompts",
    )
    print(finish(fig, "roc_matched", matched_roc, ["kind", "detector", "fpr", "tpr"], out / "figures"))

    style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colours = {
        "delta_maha": ORANGE,
        "delta_knn": BLUE,
        "knn_one_sided": AQUA,
        "combined_lm": "#7a5195",
        "minilm": "#444444",
        "tfidf": "#999999",
        "length": NEUTRAL,
    }
    for d in DETECTORS:
        r = [(a, b) for name, a, b in roc_rows if name == d]
        ax.plot(
            [a for a, _ in r],
            [b for _, b in r],
            color=colours[d],
            label=f"{d} (AUROC {results[d]['auroc_pos_vs_neg']:.2f})",
        )
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", label="random")
    ax.set_xlabel("False positive rate (unseen covered-kind prompts flagged)")
    ax.set_ylabel("True positive rate (legal and medical flagged)")
    ax.legend(fontsize=8, loc="lower right")
    tidy(ax, grid_axis="both", drop_left=False)
    header(
        fig,
        "Training-only detectors: legal/medical vs five unseen covered-kind sets\nOlmo-3-7B-DPO layer 20, n per set after near-copy removal",
    )
    print(finish(fig, "roc_negatives", roc_rows, ["detector", "fpr", "tpr"], out / "figures"))

    decision = {
        "n_per_set_used": {n: int(((names == n) & use).sum()) for n in eval_names},
        "near_copy_share": {n: round(float(is_copy[names == n].mean()), 4) for n in (*NEGATIVES, *POSITIVES)},
        "n_nan": int(sum(np.isnan(v).sum() for v in scores.values())),
        "matched": matched,
        "results": results,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
