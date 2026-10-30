# builds behaviour_in_loop_gaps's prompts: iterative_discovery v2 gap groups (rebuilt and checked), matched ordinary-traffic controls, idmix and toxicchat jailbreaks.
# uv run -m scripts.run_build_gap_pools --iterative_discovery_dir results/raw/261019_iterative_discovery_v2 --output_dir data/processed/behaviour_in_loop_gaps --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.data.gaps import draw_controls, merge_groups, rebuild_clusters
from src.data.io import load_jsonl
from src.data.sharechat import normalise
from src.interp.features import render_user
from src.metrics.detectors import LENGTH_BANDS
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

TOXICCHAT_REVISION = "29df8e4dba60e1f4af4b4075c0705c5b313548a8"


@dataclass
class GapPoolsConfig:
    iterative_discovery_dir: str = ""
    output_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    tier1: list[str] = field(
        default_factory=lambda: ["dr_top_r2_c5", "dknn_top_r1_c4", "dknn_top_r1_c7", "dknn_top_r2_c3"]
    )
    controls_per_row: int = 2
    n_idmix: int = 300
    min_tier2_size: int = 15
    max_prompt_tokens: int = 6144


def main():
    config = simple_parsing.parse(GapPoolsConfig)
    assert config.iterative_discovery_dir and config.output_dir, "iterative_discovery_dir and output_dir are required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    e9 = Path(config.iterative_discovery_dir)
    c9 = json.loads((e9 / "config.json").read_text())

    wc, lm = load_jsonl(c9["wildchat_pool"]), load_jsonl(c9["lmsys_pool"])
    texts = [r["text"] for r in wc] + [r["text"] for r in lm]
    pool_rows = load_jsonl(e9 / "pool_rows.jsonl")
    assert len(pool_rows) == len(texts) and [r["id"] for r in pool_rows] == [f"pool_{i:05d}" for i in range(len(texts))]
    source = np.array([r["source"] for r in pool_rows])
    band = np.digitize([r["n_tokens"] for r in pool_rows], LENGTH_BANDS)

    embedder = SentenceTransformer(c9["embed_model"], device="cuda", revision=c9["embed_model_revision"])
    emb = embedder.encode(
        texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    reviews = load_jsonl(e9 / "reviews.jsonl")
    members = rebuild_clusters(emb, reviews, c9["n_clusters"], c9["seed"])
    stored = {c["id"]: c for c in load_jsonl(e9 / "clusters.jsonl")}
    assert set(stored) == set(members), "rebuilt cluster ids differ from iterative_discovery's"
    for cid, rows in members.items():
        examples = [" ".join(texts[i].split())[:200] for i in rows[:5]]
        assert (
            stored[cid]["size"] == len(rows) and stored[cid]["examples"] == examples
        ), f"{cid} does not match iterative_discovery"
    print(f"all {len(members)} rebuilt clusters match iterative_discovery v2")

    assert all(stored[c]["confirmed"] for c in config.tier1), "a tier-1 cluster is not confirmed in iterative_discovery"
    assert {c for c, s in stored.items() if s["confirmed"] and s["kind"] == "top"} == set(
        config.tier1
    ), "tier-1 list is not every confirmed top cluster"
    tier2_ids = [
        c
        for c, s in stored.items()
        if s["kind"] == "top" and s["coherent"] and not s["confirmed"] and s["size"] >= config.min_tier2_size
    ]
    groups, taken = [], set()
    for tier, ids in (("t1", config.tier1), ("t2", sorted(tier2_ids))):
        for k, g in enumerate(merge_groups(members, ids, emb, c9["same_gap_cos"])):
            rows = [int(i) for c in g for i in members[c] if int(i) not in taken]
            rows = list(dict.fromkeys(rows))
            taken.update(rows)
            groups.append(
                {
                    "group": f"{tier}_g{k}",
                    "tier": tier,
                    "clusters": g,
                    "names": [stored[c]["name"] for c in g],
                    "rows": rows,
                }
            )
    groups = [g for g in groups if g["rows"]]

    top_reviewed = np.zeros(len(texts), dtype=bool)
    for rev in reviews:
        if rev["kind"] == "top":
            top_reviewed[rev["rows"]] = True
    eligible = ~top_reviewed
    prompts = []
    for g in groups:
        rows = np.array(g["rows"])
        assert top_reviewed[rows].all(), "a gap row was never in a top review"
        ctrl = draw_controls(rows, source, band, eligible, config.controls_per_row, rng)
        g["n_rows"], g["n_controls"] = len(rows), int(ctrl.size)
        for r, cs in zip(rows, ctrl):
            prompts.append(
                {
                    "id": f"pool_{r:05d}",
                    "text": texts[r],
                    "role": "gap",
                    "group": g["group"],
                    "tier": g["tier"],
                    "source": str(source[r]),
                }
            )
            for c in cs:
                prompts.append(
                    {
                        "id": f"pool_{c:05d}",
                        "text": texts[c],
                        "role": "control",
                        "group": g["group"],
                        "tier": g["tier"],
                        "matched_to": f"pool_{r:05d}",
                        "source": str(source[c]),
                    }
                )

    idmix = load_jsonl("data/processed/repro_two_sided/idmix.jsonl")
    for i in np.sort(rng.choice(len(idmix), size=config.n_idmix, replace=False)):
        prompts.append(
            {
                "id": f"idmix_{idmix[i]['id']}",
                "text": idmix[i]["text"],
                "role": "idmix",
                "group": "idmix",
                "tier": "ref",
                "source": "dolci",
            }
        )
    tc = pd.read_csv(
        hf_hub_download(
            "lmsys/toxic-chat",
            "data/0124/toxic-chat_annotation_all.csv",
            repo_type="dataset",
            revision=TOXICCHAT_REVISION,
        )
    )
    tc = tc[tc["jailbreaking"] == 1]
    tc = tc[~tc["user_input"].map(normalise).duplicated()]
    for r in tc.itertuples():
        prompts.append(
            {
                "id": f"tc_{r.conv_id}",
                "text": r.user_input,
                "role": "toxic",
                "group": "toxicchat_jailbreak",
                "tier": "ref",
                "source": "toxicchat",
            }
        )

    assert len({p["id"] for p in prompts}) == len(prompts), "duplicate prompt ids"
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    n_tok = [len(tokenizer(render_user(tokenizer, p["text"]), add_special_tokens=False)["input_ids"]) for p in prompts]
    too_long = [p["id"] for p, n in zip(prompts, n_tok) if n > config.max_prompt_tokens]
    assert not too_long, f"prompts over {config.max_prompt_tokens} tokens: {too_long}"
    for p, n in zip(prompts, n_tok):
        p["n_prompt_tokens"] = n
    write_jsonl(prompts, out / "prompts.jsonl")
    write_json([{k: v for k, v in g.items() if k != "rows"} for g in groups], out / "groups.json")
    counts = {r: sum(p["role"] == r for p in prompts) for r in ("gap", "control", "idmix", "toxic")}
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "toxicchat_revision": TOXICCHAT_REVISION,
            "counts": counts,
            "n_prompts": len(prompts),
        },
        out / "build.json",
    )
    for g in groups:
        print(g["group"], g["n_rows"], g["names"])
    print(f"{counts=}, {len(prompts)} prompts written to {out / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
