# exp21: blind llm check of whether each exp20 gap is really missing from the mix, with known-answer control sets.
# uv run -m scripts.run_coverage_check --run_dir results/raw/261020_exp21_coverage_check_v1 --exp20_dir results/raw/261019_exp20_act_vs_word_v1 --banks_dir data/processed/exp06 --idmix_path data/processed/exp01/idmix.jsonl --model_id openai/gpt-4.1 --seed 0

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo, normalise
from src.metrics.discovery import centroid, cluster, match_gap, name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("act", "word", "random")
LABELS = ["COVERED", "PARTLY", "MISSING"]

JUDGE_PROMPT = """Below are {n} messages that users sent to an AI chatbot. Under each message are the most similar prompts found in a training dataset (by meaning and by shared words).

{blocks}

Question: does the training dataset, judging from the prompts shown, contain requests of the same kind (same topic and same task) as most of these messages?
Reply in exactly this format and nothing else:
REASON: <one sentence>
ANSWER: <COVERED, PARTLY or MISSING>"""


@dataclass
class CoverageCheckConfig:
    run_dir: str = ""
    exp20_dir: str = ""
    banks_dir: str = ""
    idmix_path: str = ""
    model_id: str = "openai/gpt-4.1"
    seed: int = 0
    n_shown: int = 10
    n_embed_neighbours: int = 3
    n_word_neighbours: int = 2
    max_chars: int = 300
    n_control_sets: int = 3


def shorten(text: str, n: int) -> str:
    return " ".join(text.split())[:n]


def main():
    config = simple_parsing.parse(CoverageCheckConfig)
    assert config.run_dir and config.exp20_dir and config.banks_dir and config.idmix_path, "input paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    e20 = Path(config.exp20_dir)
    c20 = json.loads((e20 / "config.json").read_text())
    c9 = c20["exp09_config"]
    texts = [r["text"] for r in load_jsonl(c9["wildchat_pool"])] + [r["text"] for r in load_jsonl(c9["lmsys_pool"])]
    embedder = SentenceTransformer(c9["embed_model"], device="cuda", revision=c9["embed_model_revision"])

    def encode(t: list[str]) -> np.ndarray:
        return embedder.encode(
            t, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    emb = encode(texts)

    # rebuild exp20's clusters and gaps, and check them against what exp20 stored
    reviews = {(r["arm"], r["round"]): np.array(r["rows"]) for r in load_jsonl(e20 / "reviews.jsonl")}
    stored = {c["id"]: c for c in load_jsonl(e20 / "clusters.jsonl")}
    d20 = json.loads((e20 / "decision.json").read_text())
    sets = []
    for s in c20["cluster_seeds"]:
        gaps = {arm: [] for arm in ARMS}
        for arm in ARMS:
            for rnd in range(1, c20["rounds"] + 1):
                rows = reviews[(arm, rnd)]
                labels = cluster(emb[rows], c20["n_clusters"], s)
                for c in range(c20["n_clusters"]):
                    m = rows[labels == c]
                    st = stored[f"s{s}_{arm}_r{rnd}_c{c}"]
                    assert st["size"] == len(m) and st["examples"] == [
                        " ".join(texts[i].split())[:200] for i in m[:5]
                    ], st["id"]
                    if not st["confirmed"]:
                        continue
                    if st["gap_index"] == len(gaps[arm]):
                        gaps[arm].append(
                            {
                                "centroid": centroid(emb[m]),
                                "members": list(m),
                                "name": st["name"],
                                "first_cluster": st["id"],
                            }
                        )
                    else:
                        gaps[arm][st["gap_index"]]["members"] += list(m)
        word_cents = [g["centroid"] for g in gaps["word"]]
        act_cents = [g["centroid"] for g in gaps["act"]]
        for arm in ARMS:
            for k, g in enumerate(gaps[arm]):
                only = None
                if arm == "act":
                    only = match_gap(g["centroid"], word_cents, c20["same_gap_cos"]) is None
                elif arm == "word":
                    only = match_gap(g["centroid"], act_cents, c20["same_gap_cos"]) is None
                sets.append(
                    {
                        "id": f"s{s}_{arm}_g{k}",
                        "kind": "gap",
                        "seed": s,
                        "arm": arm,
                        "name": g["name"],
                        "only": only,
                        "members": [texts[i] for i in dict.fromkeys(g["members"])],
                    }
                )
        stored_seed = d20["per_seed"][str(s)]
        assert {a: len(gaps[a]) for a in ARMS} == stored_seed["distinct_gaps"], f"seed {s} gap counts differ from exp20"
        assert (
            sum(x["only"] for x in sets if x["seed"] == s and x["arm"] == "act") == stored_seed["act_unique"]
        ), f"seed {s} act-only differs"
        assert (
            sum(x["only"] for x in sets if x["seed"] == s and x["arm"] == "word") == stored_seed["word_unique"]
        ), f"seed {s} word-only differs"
    print(f"rebuilt {sum(x['kind'] == 'gap' for x in sets)} gaps, all matching exp20")

    expected = {"legal": "MISSING", "medical": "MISSING", "magicoder": "COVERED", "aya": "COVERED", "idmix": "COVERED"}
    banks = {
        t: [r["text"] for r in load_jsonl(Path(config.banks_dir) / f"needles_{t}.jsonl")]
        for t in ("legal", "medical", "magicoder", "aya")
    }
    banks["idmix"] = [r["text"] for r in load_jsonl(config.idmix_path)]
    for t, bank in banks.items():
        perm = rng.permutation(len(bank))
        for j in range(config.n_control_sets):
            pick = perm[j * config.n_shown : (j + 1) * config.n_shown]
            sets.append(
                {
                    "id": f"ctrl_{t}_{j}",
                    "kind": "control",
                    "control_type": t,
                    "expected": expected[t],
                    "members": [bank[i] for i in pick],
                }
            )

    dolci = load_dolci_dpo()["first_user"].drop_duplicates().tolist()
    dolci_norm = [normalise(t) for t in dolci]
    ref = torch.from_numpy(encode(dolci)).cuda()
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000).fit(dolci)
    dolci_tfidf = vec.transform(dolci)
    prompts = []
    for st in sets:
        shown = (
            st["members"]
            if len(st["members"]) <= config.n_shown
            else [st["members"][i] for i in np.sort(rng.choice(len(st["members"]), config.n_shown, replace=False))]
        )
        e = torch.from_numpy(encode(shown)).cuda()
        embed_rank = (e @ ref.T).topk(config.n_embed_neighbours + 5, dim=1).indices.cpu().numpy()
        word_rank = np.argsort(-(vec.transform(shown) @ dolci_tfidf.T).toarray(), axis=1)[
            :, : config.n_word_neighbours + 5
        ]
        blocks = []
        st["shown"] = []
        for j, msg in enumerate(shown):
            own = normalise(msg)
            nb = [k for k in embed_rank[j] if dolci_norm[k] != own][: config.n_embed_neighbours]
            nb += [k for k in word_rank[j] if dolci_norm[k] != own and k not in nb][: config.n_word_neighbours]
            st["shown"].append(
                {
                    "message": shorten(msg, config.max_chars),
                    "neighbours": [shorten(dolci[k], config.max_chars) for k in nb],
                }
            )
            blocks.append(
                f"MESSAGE {j + 1}: {shorten(msg, config.max_chars)}\n"
                + "\n".join(f"  similar training prompt: {shorten(dolci[k], config.max_chars)}" for k in nb)
            )
        prompts.append(JUDGE_PROMPT.format(n=len(shown), blocks="\n\n".join(blocks)))

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(name_clusters(client, config.model_id, prompts, str(run / "judge_cache")))
    for st, reply in zip(sets, replies):
        answer = reply.split("ANSWER:")[-1] if reply else None
        st["label"] = parse_label(answer, LABELS)
        st["judge"] = reply
        st["real"] = st["label"] == "MISSING"

    ctrl = [x for x in sets if x["kind"] == "control"]
    gap_ctrl = [x for x in ctrl if x["expected"] == "MISSING"]
    cov_ctrl = [x for x in ctrl if x["expected"] == "COVERED"]
    n_gap_ok = sum(x["label"] == "MISSING" for x in gap_ctrl)
    n_cov_ok = sum(x["label"] is not None and x["label"] != "MISSING" for x in cov_ctrl)
    is_valid = n_gap_ok >= 5 and n_cov_ok >= 8

    per_seed = {}
    for s in c20["cluster_seeds"]:
        g = [x for x in sets if x["kind"] == "gap" and x["seed"] == s]
        real = {a: sum(x["real"] for x in g if x["arm"] == a) for a in ARMS}
        act_only_real = sum(x["real"] and x["only"] for x in g if x["arm"] == "act")
        word_only_real = sum(x["real"] and x["only"] for x in g if x["arm"] == "word")
        if act_only_real >= 2 and real["act"] >= 2 * real["random"] and real["act"] >= real["random"] + 2:
            outcome = "adds_something"
        elif act_only_real == 0 or real["act"] <= real["random"] + 1:
            outcome = "adds_nothing"
        else:
            outcome = "ambiguous"
        per_seed[str(s)] = {
            "real_gaps": real,
            "all_gaps": {a: sum(x["arm"] == a for x in g) for a in ARMS},
            "act_only_real": act_only_real,
            "word_only_real": word_only_real,
            "outcome": outcome,
        }
    outcomes = [v["outcome"] for v in per_seed.values()]
    verdict = next((o for o in ("adds_something", "adds_nothing") if outcomes.count(o) >= 2), "ambiguous")
    if not is_valid:
        verdict = f"no_verdict_judge_invalid ({verdict})"

    decision = {
        "verdict": verdict,
        "judge_valid": is_valid,
        "controls": {x["id"]: x["label"] for x in ctrl},
        "gap_controls_missing_of_6": n_gap_ok,
        "covered_controls_not_missing_of_9": n_cov_ok,
        "per_seed": per_seed,
        "gap_labels": {
            x["id"]: {"arm": x["arm"], "name": x["name"], "only": x["only"], "label": x["label"]}
            for x in sets
            if x["kind"] == "gap"
        },
        "null_labels": sum(x["label"] is None for x in sets),
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [{k: v for k, v in x.items() if k != "members"} | {"n_members": len(x["members"])} for x in sets],
        run / "sets.jsonl",
    )
    sheet = [x for x in sets if x["kind"] == "gap"]
    pick = rng.choice(len(sheet), size=min(20, len(sheet)), replace=False)
    write_jsonl(
        [
            {"id": sheet[i]["id"], "shown": sheet[i]["shown"], "judge_label": sheet[i]["label"], "human_label": None}
            for i in pick
        ],
        run / "human_review_sheet.jsonl",
    )
    for x in ctrl:
        print(f"{x['id']}: expected {x['expected']}, judge {x['label']}")
    for s, v in per_seed.items():
        print(f"seed {s}: {v}")
    print(f"{verdict=}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
