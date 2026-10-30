# self_nature_a: finds self-nature messages in Olmo's post-training data (SFT and DPO) and in real traffic, labels them, and builds the covered / uncovered cells with their gate.
# uv run -m scripts.run_build_self_nature --phase candidates --output_dir data/processed/self_nature_a --run_dir results/raw/261021_self_nature_a_self_nature_cells_v1 --iterative_discovery_dir results/raw/261019_iterative_discovery_v2 --activations_vs_words_dir results/raw/261019_activations_vs_words_act_vs_word_v1 --seed 0

import asyncio
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.data.gaps import rebuild_activations_vs_words_gaps
from src.data.io import load_jsonl
from src.data.pools import MembershipIndex, load_dolci_dpo, normalise
from src.metrics.coverage import density_ratios
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SELF = re.compile(
    r"""
    \b(who|what)\s+(are|r)\s+(you|u)\b
    |\bwhat\s+(model|version|llm|ai|language\s+model)\s+(are|r)\s+(you|u)\b
    |\bare\s+(you|u)\s+(an?\s+)?(ai|a\.i\.|bot|robot|machine|computer|program|human|person|real|alive|conscious|sentient|self[- ]?aware|chatgpt|gpt[-\s]?\d?|claude|bard|gemini|llama|olmo|t[uü]lu|open[- ]?sourced?|openai)\b
    |\bdo\s+(you|u)\s+(have|feel|think|dream|actually|really|truly|experience|understand|know\s+(what|who)\s+you)\b
    |\b(can|could)\s+(you|u)\s+(feel|think|suffer|love|dream|lie|be\s+conscious|understand)\b
    |\byour\s+(own\s+|true\s+)?(nature|consciousness|existence|feelings|emotions|thoughts|mind|identity|training(\s+data)?|limitations|capabilities|creators?|developers?|name|weights|system\s+prompt|guidelines|license|context\s+window|cutoff)\b
    |\bwho\s+(made|built|trained|created|developed|designed|owns|programmed)\s+(you|u)\b
    |\b(tell\s+me\s+about|describe|introduce)\s+yourself\b
    |\bwhat\s+is\s+it\s+like\s+(to\s+be\s+(you|an\s+ai)|being\s+an\s+ai)\b
    |\bturing\s+test\b
    |\byou('re|\s+are)\s+(just|only)\s+(an?\s+)?(program|machine|language\s+model|llm|chatbot|bot|algorithm|predicting)\b
    |\bwhat\s+can\s+(you|u)\s+do\b
    |\bhow\s+(were|was)\s+(you|u)\s+(trained|made|built|created)\b
    |你是谁|你是什么|你是人|你有意识|你是\s*chatgpt|你叫什么|你是哪个模型|あなたは誰|あなたは\s*ai|君は誰|お前は誰|あなたの名前
    """,
    re.IGNORECASE | re.VERBOSE,
)
LABEL_PROMPT = """Below is a message a user sent to an AI assistant. Answer YES if the user asks or asserts something about the assistant itself: what it is; whether it has consciousness, feelings, experiences, a mind, free will or a self; its identity, name, model, maker or training; its capabilities and limits as an AI; how it experiences things; or presses it to reflect on its own nature (for example "you're just predicting tokens", "do you actually understand?").
Answer NO if a fictional character is being addressed inside a roleplay or story script, if "you" means a human, if the message is a task that mentions AI in the third person (such as an essay about machine consciousness), or if "what can you do" is asked about some tool other than the assistant.

<message>
{message}
</message>

Reply with YES or NO only."""
BANDS = [0, 16, 32, 64, 128, 256, 10**9]
GAP_PATTERNS = {
    "placeholder": re.compile(r"placeholder|NAME_", re.I),
    "creative": re.compile(r"creative|character|story|fiction", re.I),
}


@dataclass
class SelfNatureConfig:
    phase: str = ""
    output_dir: str = ""
    run_dir: str = ""
    iterative_discovery_dir: str = ""
    activations_vs_words_dir: str = ""
    sft_path: str = "data/processed/self_nature_a/dolci_sft_first_user.parquet"
    model_id: str = "openai/gpt-4o-mini"
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    n_nearest: int = 1000
    k: int = 50
    uncovered_ratio: float = 5.0
    covered_ratio: float = 2.0
    max_chars: int = 500
    n_read: int = 20
    # pre-registered extension when the 20k pool gives under 200 traffic hits: english wildchat shards 0-13 and lmsys shards 0-1
    extend_traffic: bool = False
    # where emb_<set>.npy files from an earlier run live; empty means run_dir
    emb_dir: str = ""


TRAFFIC_SETS = ("traffic", "traffic_ext")
EXT_POOL = {"deploy_ext": "deploy", "lmsys_ext": "lmsys"}


def load_pool(iterative_discovery_dir: str) -> list[dict]:
    import json

    c9 = json.loads((Path(iterative_discovery_dir) / "config.json").read_text())
    return load_jsonl(c9["wildchat_pool"]) + load_jsonl(c9["lmsys_pool"])


def load_traffic_ext(pool_norms: set[str]) -> list[dict]:
    from src.data.needles import lmsys_first_turns, wildchat_first_turns

    rows = [{"text": t, "source": "deploy_ext"} for t in wildchat_first_turns(list(range(14)), english_only=True)]
    rows += [{"text": r["text"], "source": "lmsys_ext"} for r in lmsys_first_turns([0, 1], os.environ["HF_TOKEN"])]
    seen, out = set(pool_norms), []
    for r in rows:
        nm = normalise(r["text"])
        if nm and nm not in seen:
            seen.add(nm)
            out.append(r)
    return out


def embed(model: SentenceTransformer, texts: list[str], max_chars: int | None, batch: int = 256) -> np.ndarray:
    # the traffic pool is embedded untruncated so activations_vs_words's clusters rebuild exactly; the millions of mix prompts are cut to save time
    cut = texts if max_chars is None else [t[:max_chars] for t in texts]
    return model.encode(
        cut, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)


def band_of(n: int) -> int:
    return int(np.searchsorted(BANDS, n, side="right") - 1)


def main():
    config = simple_parsing.parse(SelfNatureConfig)
    assert (
        config.phase in ("candidates", "label", "cells") and config.output_dir and config.run_dir
    ), "phase, output_dir, run_dir"
    out, run = Path(config.output_dir), Path(config.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    if config.phase == "candidates":
        assert (
            config.iterative_discovery_dir and config.activations_vs_words_dir
        ), "iterative_discovery_dir and activations_vs_words_dir are required"
        import json

        c9 = json.loads((Path(config.iterative_discovery_dir) / "config.json").read_text())
        sft = pd.read_parquet(config.sft_path)
        sft = sft[sft["first_user"].notna()].reset_index(drop=True)
        dpo = load_dolci_dpo()
        pool = load_pool(config.iterative_discovery_dir)
        sets = {
            "sft": (sft["first_user"].tolist(), sft["source_dataset"].tolist(), sft["id"].tolist()),
            "dpo": (dpo["first_user"].tolist(), dpo["family"].tolist(), dpo["prompt_id"].tolist()),
            "traffic": ([r["text"] for r in pool], [r["pool"] for r in pool], [r["id"] for r in pool]),
        }
        if config.extend_traffic:
            ext = load_traffic_ext({normalise(r["text"]) for r in pool})
            sets["traffic_ext"] = (
                [r["text"] for r in ext],
                [r["source"] for r in ext],
                [f"ext_{i}" for i in range(len(ext))],
            )
            write_jsonl([{"id": f"ext_{i}", **r} for i, r in enumerate(ext)], out / "traffic_ext.jsonl")
        model = SentenceTransformer(c9["embed_model"], device="cuda", revision=c9["embed_model_revision"])
        emb_dir = Path(config.emb_dir) if config.emb_dir else run
        embs = {}
        for name, (texts, _, _) in sets.items():
            path = run / f"emb_{name}.npy" if (run / f"emb_{name}.npy").exists() else emb_dir / f"emb_{name}.npy"
            # the traffic pool is always re-embedded in full precision: activations_vs_words's clusters do not rebuild from a float16 copy
            if path.exists() and name != "traffic":
                embs[name] = np.load(path).astype(np.float32)
                assert len(embs[name]) == len(texts), f"saved {name} embeddings do not match the prompt count"
                LOGGER.info(f"{name}: {len(texts)} saved embeddings loaded")
                continue
            embs[name] = embed(model, texts, None if name == "traffic" else 1000)
            np.save(run / f"emb_{name}.npy", embs[name] if name == "traffic" else embs[name].astype(np.float16))
            LOGGER.info(f"{name}: {len(texts)} prompts embedded")
        # the relative_frequency_gaps self-nature cluster, rebuilt exactly from activations_vs_words, gives the retrieval centroid
        gaps = rebuild_activations_vs_words_gaps(embs["traffic"], sets["traffic"][0], config.activations_vs_words_dir)
        target = [g for g in gaps if g["id"] == "s1_word_g0"]
        assert (
            len(target) == 1 and "nature" in target[0]["name"].lower()
        ), f"self-nature gap not found: {[g['name'] for g in gaps]}"
        cen = embs["traffic"][target[0]["members"]].mean(0)
        cen = cen / np.linalg.norm(cen)
        write_jsonl(
            [
                {
                    "id": g["id"],
                    "name": g["name"],
                    "arm": g["arm"],
                    "seed": g["seed"],
                    "only": g["only"],
                    "members": g["members"],
                }
                for g in gaps
            ],
            out / "activations_vs_words_gaps.jsonl",
        )
        rows = []
        for name, (texts, sources, ids) in sets.items():
            sims = torch.from_numpy(embs[name]).cuda() @ torch.from_numpy(cen).cuda()
            near = set(sims.topk(min(config.n_nearest, len(texts))).indices.cpu().tolist())
            for i, t in enumerate(texts):
                by_regex = bool(SELF.search(t[: config.max_chars]))
                if by_regex or i in near:
                    rows.append(
                        {
                            "id": f"{name}_{i}",
                            "set": name,
                            "index": i,
                            "source": sources[i],
                            "source_id": str(ids[i]),
                            "text": t,
                            "by_regex": by_regex,
                            "by_embed": i in near,
                            "cos_to_centroid": round(float(sims[i]), 4),
                        }
                    )
        write_jsonl(rows, out / "candidates.jsonl")
        counts = {n: {"prompts": len(sets[n][0]), "candidates": sum(r["set"] == n for r in rows)} for n in sets}
        write_json(
            {
                "git_hash": get_git_hash(),
                **asdict(config),
                "counts": counts,
                "self_nature_cluster_size": len(target[0]["members"]),
            },
            out / "build_candidates.json",
        )
        print(f"{counts=}")
        print(f"candidates written to {out / 'candidates.jsonl'}")
        return

    cands = load_jsonl(out / "candidates.jsonl")
    if config.phase == "label":
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        replies = asyncio.run(
            name_clusters(
                client,
                config.model_id,
                [LABEL_PROMPT.format(message=r["text"][:2000]) for r in cands],
                str(out / "label_cache"),
                max_concurrent=32,
            )
        )
        labelled = [{**r, "label": parse_label(x, ["YES", "NO"])} for r, x in zip(cands, replies)]
        write_jsonl(labelled, out / "labelled.jsonl")
        n_yes = {
            n: sum(r["set"] == n and r["label"] == "YES" for r in labelled)
            for n in dict.fromkeys(r["set"] for r in cands)
        }
        n_fail = sum(r["label"] is None for r in labelled)
        print(f"{n_yes=} label_failed={n_fail}")
        print(f"labels written to {out / 'labelled.jsonl'}")
        return

    labelled = load_jsonl(out / "labelled.jsonl")
    yes = [r for r in labelled if r["label"] == "YES"]
    sft = pd.read_parquet(config.sft_path)
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    dpo = load_dolci_dpo()
    pool = load_pool(config.iterative_discovery_dir)
    import json

    built = json.loads((out / "build_candidates.json").read_text())["counts"]
    n_prompts = {"sft": len(sft), "dpo": len(dpo), "traffic": len(pool)}
    if "traffic_ext" in built:
        n_prompts["traffic_ext"] = built["traffic_ext"]["prompts"]
    counts = {}
    for n in n_prompts:
        ys = [r for r in yes if r["set"] == n]
        by_source = pd.Series([r["source"] for r in ys]).value_counts().to_dict() if ys else {}
        counts[n] = {
            "prompts": n_prompts[n],
            "candidates": sum(r["set"] == n for r in labelled),
            "yes": len(ys),
            "yes_per_100k": round(len(ys) / n_prompts[n] * 1e5, 2),
            "label_failed": sum(r["set"] == n and r["label"] is None for r in labelled),
            "yes_by_source": by_source,
        }
    sft_domain = sft.set_index("id")["domain"]
    for r in yes:
        if r["set"] == "sft":
            r["domain"] = str(sft_domain.get(r["source_id"], ""))
    write_jsonl([r for r in yes if r["set"] in ("sft", "dpo")], out / "yes_dolci.jsonl")

    emb_dir = Path(config.emb_dir) if config.emb_dir else run
    emb = {
        n: np.load((run if (run / f"emb_{n}.npy").exists() else emb_dir) / f"emb_{n}.npy").astype(np.float32)
        for n in n_prompts
    }
    norms = {
        "sft": [normalise(t) for t in sft["first_user"]],
        "dpo": [normalise(t) for t in dpo["first_user"]],
        "traffic": [normalise(r["text"]) for r in pool],
    }
    # the mix for coverage: unique first user messages of sft and dpo together
    seen, mix_idx = set(), []
    for n in ("sft", "dpo"):
        for i, nm in enumerate(norms[n]):
            if nm not in seen:
                seen.add(nm)
                mix_idx.append((n, i))
    m_emb = np.concatenate(
        [emb["sft"][[i for n, i in mix_idx if n == "sft"]], emb["dpo"][[i for n, i in mix_idx if n == "dpo"]]]
    )
    m_norm = [norms[n][i] for n, i in mix_idx if n == "sft"] + [norms[n][i] for n, i in mix_idx if n == "dpo"]
    dpo_u = [i for i, nm in enumerate(norms["dpo"]) if nm in seen]
    traffic_yes = [r for r in yes if r["set"] in TRAFFIC_SETS]
    # extended-traffic prompts are scored against the 20k pool as the traffic reference; -1 means no self row to leave out
    q_emb = np.stack([emb[r["set"]][r["index"]] for r in traffic_yes])
    q_norm = [normalise(r["text"]) for r in traffic_yes]
    q_self = [r["index"] if r["set"] == "traffic" else -1 for r in traffic_yes]
    ratio = density_ratios(q_emb, q_norm, q_self, emb["traffic"], norms["traffic"], m_emb, m_norm, k=config.k)
    ratio_dpo = density_ratios(
        q_emb,
        q_norm,
        q_self,
        emb["traffic"],
        norms["traffic"],
        emb["dpo"][dpo_u],
        [norms["dpo"][i] for i in dpo_u],
        k=config.k,
    )
    for r, a, b in zip(traffic_yes, ratio, ratio_dpo):
        r["ratio"], r["ratio_dpo_only"] = round(float(a), 3), round(float(b), 3)
    gaps = load_jsonl(out / "activations_vs_words_gaps.jsonl")
    cl = [g for g in gaps if g["id"] == "s1_word_g0"][0]["members"]
    cl_ratio = density_ratios(
        emb["traffic"][cl],
        [norms["traffic"][i] for i in cl],
        cl,
        emb["traffic"],
        norms["traffic"],
        m_emb,
        m_norm,
        k=config.k,
    )
    del m_emb

    # membership of traffic self-nature prompts in dolci: exact normalised copies against sft+dpo, 13-word shingles against dpo
    shingle_index = MembershipIndex(dpo["first_user"].tolist())
    for r in traffic_yes:
        r["exact_copy_in_dolci"] = normalise(r["text"]) in seen
        r["shingle_member_dpo"] = shingle_index.is_member(r["text"])

    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    for r in yes:
        r["n_tokens"] = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        r["band"] = band_of(r["n_tokens"])
    covered = [r for r in yes if r["set"] not in TRAFFIC_SETS] + [
        r for r in traffic_yes if r["ratio"] <= config.covered_ratio
    ]
    uncovered = [r for r in traffic_yes if r["ratio"] >= config.uncovered_ratio]
    between = [r for r in traffic_yes if config.covered_ratio < r["ratio"] < config.uncovered_ratio]
    # length matching: per band, downsample the larger cell to the smaller cell's count
    cov_m, unc_m = [], []
    for b in range(len(BANDS) - 1):
        c, u = [r for r in covered if r["band"] == b], [r for r in uncovered if r["band"] == b]
        n = min(len(c), len(u))
        cov_m += [c[i] for i in sorted(rng.choice(len(c), size=n, replace=False))] if n else []
        unc_m += [u[i] for i in sorted(rng.choice(len(u), size=n, replace=False))] if n else []
    # other topics, covered: ordinary pool prompts, never a candidate and never in a confirmed activations_vs_words gap, 2 per matched prompt, same band (and pool for traffic prompts)
    cand_idx = {r["index"] for r in labelled if r["set"] == "traffic"}
    gap_idx = {i for g in gaps for i in g["members"]}
    ordinary = [i for i in range(len(pool)) if i not in cand_idx and i not in gap_idx]
    ord_tok = {i: len(tok(pool[i]["text"], add_special_tokens=False)["input_ids"]) for i in ordinary}
    buckets: dict[tuple[str, int], list[int]] = {}
    for i in ordinary:
        buckets.setdefault((pool[i]["pool"], band_of(ord_tok[i])), []).append(i)
    for v in buckets.values():
        rng.shuffle(v)
    pools_cycle = ["deploy", "lmsys"]
    other_cov, short = [], 0
    for j, r in enumerate(cov_m + unc_m):
        src = EXT_POOL.get(r["source"], r["source"]) if r["set"] in TRAFFIC_SETS else pools_cycle[j % 2]
        for _ in range(2):
            b = buckets.get((src, r["band"]), [])
            if b:
                i = b.pop()
                other_cov.append(
                    {
                        "id": f"other_cov_{i}",
                        "cell": "other_covered",
                        "set": "traffic",
                        "index": i,
                        "source": pool[i]["pool"],
                        "text": pool[i]["text"],
                        "n_tokens": ord_tok[i],
                        "band": band_of(ord_tok[i]),
                        "matched_to": r["id"],
                    }
                )
            else:
                short += 1
    other_unc = []
    for key, pat in GAP_PATTERNS.items():
        members = list(
            dict.fromkeys(i for g in gaps if g["arm"] == "word" and pat.search(g["name"]) for i in g["members"])
        )
        for i in members:
            other_unc.append(
                {
                    "id": f"other_unc_{key}_{i}",
                    "cell": "other_uncovered",
                    "gap": key,
                    "set": "traffic",
                    "index": i,
                    "source": pool[i]["pool"],
                    "text": pool[i]["text"],
                }
            )
    for r in other_unc:
        r["n_tokens"] = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        r["band"] = band_of(r["n_tokens"])

    def strip(r: dict, cell: str) -> dict:
        keep = (
            "id",
            "set",
            "index",
            "source",
            "source_id",
            "text",
            "n_tokens",
            "band",
            "ratio",
            "ratio_dpo_only",
            "domain",
            "exact_copy_in_dolci",
            "shingle_member_dpo",
        )
        return {"cell": cell, **{k: r[k] for k in keep if k in r}}

    cells = (
        [strip(r, "self_covered") for r in cov_m] + [strip(r, "self_uncovered") for r in unc_m] + other_cov + other_unc
    )
    write_jsonl(cells, run / "cells.jsonl")
    n_small = min(len(cov_m), len(unc_m))
    gate = "pass" if n_small >= 100 else "pass_flagged_small" if n_small >= 50 else "stop_cells_too_small"
    sample = {}
    for name, rs in (("self_covered", cov_m), ("self_uncovered", unc_m)):
        pick = rng.choice(len(rs), size=min(config.n_read, len(rs)), replace=False) if rs else []
        sample[name] = [rs[i]["text"][:300] for i in pick]
    (run / "read_sample.md").write_text(
        "\n\n".join(f"## {k}\n\n" + "\n".join(f"- {t!r}" for t in v) for k, v in sample.items())
    )
    decision = {
        "counts": counts,
        "traffic_ratio_distribution": {
            "covered_le_2": sum(r["ratio"] <= config.covered_ratio for r in traffic_yes),
            "between": len(between),
            "uncovered_ge_5": len(uncovered),
            "median_ratio": float(np.median(ratio)) if len(ratio) else None,
            "median_ratio_dpo_only": float(np.median(ratio_dpo)) if len(ratio_dpo) else None,
        },
        "relative_frequency_gaps_cluster_ratio_sft_plus_dpo_mix": float(np.median(cl_ratio)),
        "traffic_yes_copies_in_dolci": {
            "exact": sum(r["exact_copy_in_dolci"] for r in traffic_yes),
            "shingle_dpo": sum(r["shingle_member_dpo"] for r in traffic_yes),
        },
        "cells_before_matching": {
            "self_covered": len(covered),
            "self_covered_from_dolci": sum(r["set"] not in TRAFFIC_SETS for r in covered),
            "self_uncovered": len(uncovered),
        },
        "cells_after_matching": {
            "self_covered": len(cov_m),
            "self_uncovered": len(unc_m),
            "other_covered": len(other_cov),
            "other_covered_short": short,
            "other_uncovered": {k: sum(r["gap"] == k for r in other_unc) for k in GAP_PATTERNS},
        },
        "length_median_tokens": {
            "self_covered": float(np.median([r["n_tokens"] for r in cov_m])) if cov_m else None,
            "self_uncovered": float(np.median([r["n_tokens"] for r in unc_m])) if unc_m else None,
        },
        "self_covered_sources_after_matching": pd.Series([r["set"] for r in cov_m]).value_counts().to_dict()
        if cov_m
        else {},
        "sft_yes_domains": pd.Series([r.get("domain") for r in yes if r["set"] == "sft"]).value_counts().to_dict(),
        "gate": gate,
        "read_sample": sample,
    }
    write_json(decision, run / "decision.json")
    for n, c in counts.items():
        print(
            f"{n}: {c['prompts']} prompts, {c['candidates']} candidates, {c['yes']} yes ({c['yes_per_100k']} per 100k), label failed {c['label_failed']}"
        )
        print(f"  by source: {c['yes_by_source']}")
    print(f"traffic ratios: {decision['traffic_ratio_distribution']}")
    print(
        f"relative_frequency_gaps cluster ratio with sft+dpo mix: {decision['relative_frequency_gaps_cluster_ratio_sft_plus_dpo_mix']:.2f}"
    )
    print(f"cells before matching {decision['cells_before_matching']}, after {decision['cells_after_matching']}")
    print(f"gate: {gate}")
    print(f"cells written to {run / 'cells.jsonl'}; read sample at {run / 'read_sample.md'}")


if __name__ == "__main__":
    main()
