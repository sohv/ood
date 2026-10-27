# exp64 v2: splits three found gaps into request kinds with an LLM, adds 25, 50 and 100 percent of the largest kind to the mix side, and checks which scores release it while keeping its siblings and unrelated gaps.
# uv run -m scripts.run_subcluster_addone_v2 --run_dir results/raw/261027_exp64_subcluster_addone_v2 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp41_dir results/raw/261022_exp41_gap_loop_v1 --exp61_dir results/raw/261026_exp61_labels_v1 --exp64_dir results/raw/261027_exp64_subcluster_addone_v1 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.generation.cache import cached_llm_call
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

SPLIT_PROMPT = """Below are {n} messages that users sent to an AI chatbot, numbered 1 to {n}, separated by lines of ---.

{messages}

These messages were grouped together as one topic. Sort them into distinct KINDS OF REQUEST: a kind is what the user is asking the assistant to do, at a grain where someone collecting training data would write different examples for different kinds (for example "translate a sentence into French" and "solve a maths word problem in French" are different kinds; two maths problems with different numbers are the same kind). Use at most {max_kinds} kinds. If the messages are all one kind of request, return one kind. Every message belongs to exactly one kind.
Reply with JSON only, in exactly this shape and nothing else:
{{"kinds": [{{"name": "<at most 6 words>", "description": "<one sentence>", "indices": [<message numbers>]}}]}}"""


@dataclass
class AddOneV2Config:
    run_dir: str = ""
    exp37_dir: str = ""
    exp41_dir: str = ""
    exp61_dir: str = ""
    exp64_dir: str = ""
    model_id: str = "anthropic/claude-sonnet-4.6"
    split_cache_dir: str = "cache/exp64v2_split"
    seed: int = 0
    n_topics: int = 3
    max_kinds: int = 3
    n_sample: int = 60
    sample_chars: int = 300
    holdout: float = 0.2
    doses: list[int] = field(default_factory=lambda: [25, 50, 100])
    k_small: int = 50
    k_large: int = 200
    n_examples: int = 5
    max_change: float = 0.10
    line_sample: int = 50000
    n_random_subsets: int = 1000
    subset_size: int = 500
    ratio_multiple: float = 3.0
    num_tasks: int | None = None


def mix_share(labels: np.ndarray, m_gpu: torch.Tensor, t_np: np.ndarray, n_labels: int) -> np.ndarray:
    # every mix prompt goes to its nearest traffic centroid; share = mix rate / (mix rate + traffic rate)
    cent = np.zeros((n_labels, t_np.shape[1]), dtype=np.float32)
    np.add.at(cent, labels, t_np.astype(np.float32))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    cent_gpu = torch.from_numpy(cent).cuda()
    assign = np.empty(len(m_gpu), dtype=np.int64)
    for s in range(0, len(m_gpu), 65536):
        assign[s : s + 65536] = (m_gpu[s : s + 65536].float() @ cent_gpu.T).argmax(1).cpu().numpy()
    rate_t = np.bincount(labels, minlength=n_labels) / len(labels)
    rate_m = np.bincount(assign, minlength=n_labels) / len(m_gpu)
    return rate_m / (rate_m + rate_t + 1e-12)


async def split_kinds(client, config: AddOneV2Config, texts: list[str], cache_dir: Path) -> list[dict]:
    messages = "\n---\n".join(f"{i + 1}. {t[: config.sample_chars]}" for i, t in enumerate(texts))
    prompt = SPLIT_PROMPT.format(n=len(texts), messages=messages, max_kinds=config.max_kinds)
    reply = await cached_llm_call(
        client, config.model_id, [{"role": "user", "content": prompt}], cache_dir=str(cache_dir)
    )
    body = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.MULTILINE).strip()
    kinds = json.loads(body)["kinds"]
    # a message listed under two kinds stays with the kind that lists it first
    taken: set[int] = set()
    for k in kinds:
        k["indices"] = [i for i in k["indices"] if 1 <= i <= len(texts) and i not in taken and not taken.add(i)]
    kinds = [k for k in kinds if k["indices"]]
    return kinds


def main():
    config = simple_parsing.parse(AddOneV2Config)
    assert config.run_dir and config.exp37_dir and config.exp41_dir and config.exp61_dir and config.exp64_dir
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])[:2]
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    line41 = json.loads((Path(config.exp41_dir) / "decision.json").read_text())["ratio_line"]
    n_clusters = int(labels.max()) + 1

    e64 = json.loads((Path(config.exp64_dir) / "decision.json").read_text())
    chosen = list(e64["chosen"])[: config.n_topics]
    unrelated = list(e64["unrelated"])
    doses = config.doses
    if config.num_tasks:
        chosen, doses = chosen[:1], [50]
    LOGGER.info(f"chosen {chosen}; doses {doses}; {len(unrelated)} unrelated control topics")

    # kinds of request from the LLM on a sample, then every member to the nearest kind centroid
    members = {c: np.where(labels == c)[0] for c in chosen}
    kinds_info: dict[int, list[dict]] = {}
    parts: dict[int, list[np.ndarray]] = {}
    for c in chosen:
        sample = rng.choice(members[c], size=min(config.n_sample, len(members[c])), replace=False)
        kinds = asyncio.run(
            split_kinds(client, config, [rows[i]["norm"] for i in sample], Path(config.split_cache_dir))
        )
        LOGGER.info(f"topic c{c}: {len(kinds)} kinds: {[k['name'] for k in kinds]}")
        if len(kinds) < 2:
            kinds_info[c] = [
                {"name": k["name"], "description": k["description"], "sample_size": len(k["indices"])} for k in kinds
            ]
            parts[c] = []
            continue
        cent = np.stack([t_np[sample[np.array(k["indices"]) - 1]].astype(np.float32).mean(0) for k in kinds])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        assign = (t_np[members[c]].astype(np.float32) @ cent.T).argmax(1)
        order = np.argsort([-(assign == j).sum() for j in range(len(kinds))])
        parts[c] = [members[c][assign == j] for j in order]
        kinds_info[c] = [
            {
                "name": kinds[j]["name"],
                "description": kinds[j]["description"],
                "sample_size": len(kinds[j]["indices"]),
                "size": int((assign == j).sum()),
                "role": "added" if rank == 0 else "sibling",
                "examples": [rows[i]["norm"][:300] for i in members[c][assign == j][: config.n_examples]],
            }
            for rank, j in enumerate(order)
        ]
    tested = [c for c in chosen if parts[c]]
    single_kind = [c for c in chosen if not parts[c]]
    LOGGER.info(f"tested {tested}; single-kind {single_kind}")

    held: dict[int, np.ndarray] = {}
    pool: dict[int, np.ndarray] = {}
    for c in tested:
        perm = rng.permutation(parts[c][0])
        n_held = max(1, round(config.holdout * len(perm)))
        held[c], pool[c] = perm[:n_held], perm[n_held:]
        LOGGER.info(f"topic c{c}: part sizes {[len(p) for p in parts[c]]}; held out {n_held}; add pool {len(pool[c])}")

    scored = np.unique(np.concatenate([members[c] for c in chosen] + [np.where(labels == c)[0] for c in unrelated]))
    t_gpu, th_gpu = torch.from_numpy(t_np).cuda(), torch.from_numpy(t_hash).cuda()
    m_gpu, mh_gpu = torch.from_numpy(m_np).cuda(), torch.from_numpy(m_hash).cuda()
    line_sample = rng.choice(len(rows), size=config.line_sample, replace=False)

    def ratios(m_emb: torch.Tensor, m_h: torch.Tensor, k: int, idx: np.ndarray = scored) -> np.ndarray:
        return density_ratios_batched(t_np[idx], t_hash[idx], idx, t_gpu, th_gpu, m_emb, m_h, k=k)

    def gap_line(k: int) -> float:
        # exp41's rule at this k: 3 times the median of random-subset medians, subsets drawn from a scored traffic sample
        r = ratios(m_gpu, mh_gpu, k, line_sample)
        sub_rng = np.random.default_rng(config.seed)
        medians = [
            float(np.median(r[sub_rng.choice(len(r), size=config.subset_size, replace=False)]))
            for _ in range(config.n_random_subsets)
        ]
        return config.ratio_multiple * float(np.median(medians))

    lines = {f"ratio_k{k}": gap_line(k) for k in (config.k_small, config.k_large)}
    LOGGER.info(f"gap lines {lines}; exp41's line at k=50 was {line41}")

    grp: dict[str, np.ndarray] = {}
    for c in tested:
        for j, part in enumerate(parts[c]):
            grp[f"c{c}_kind{j}"] = held[c] if j == 0 else part
    for c in single_kind:
        grp[f"c{c}"] = members[c]
    for c in unrelated:
        grp[f"c{c}"] = np.where(labels == c)[0]
    pos = {int(i): n for n, i in enumerate(scored)}

    def summarise(ratio_scored: np.ndarray) -> dict[str, float]:
        return {name: float(np.median(ratio_scored[[pos[int(i)] for i in idx]])) for name, idx in grp.items()}

    # one centroid per kind for the mix share
    lab2 = labels.copy()
    share_ids = {}
    nxt = n_clusters
    for c in tested:
        for j, part in enumerate(parts[c]):
            lab2[part] = nxt
            share_ids[f"c{c}_kind{j}"] = nxt
            nxt += 1
    for name in grp:
        if name not in share_ids:
            share_ids[name] = int(name[1:])

    results = {
        "lines": lines,
        "exp41_line": line41,
        "chosen": chosen,
        "tested": tested,
        "single_kind": single_kind,
        "unrelated": unrelated,
        "doses": doses,
        "kinds": {f"c{c}": kinds_info[c] for c in chosen},
        "held_out_rows": {f"c{c}": len(held[c]) for c in tested},
        "added_rows": {},
        "conditions": {},
    }

    def condition(name: str, m_emb: torch.Tensor, m_h: torch.Tensor) -> None:
        share = mix_share(lab2, m_emb, t_np, int(lab2.max()) + 1)
        results["conditions"][name] = {
            f"ratio_k{config.k_small}": summarise(ratios(m_emb, m_h, config.k_small)),
            f"ratio_k{config.k_large}": summarise(ratios(m_emb, m_h, config.k_large)),
            "mix_share": {g: float(share[i]) for g, i in share_ids.items()},
        }
        LOGGER.info(f"condition {name} scored")

    condition("before", m_gpu, mh_gpu)
    for c in tested:
        for dose in doses:
            # copies of a share of the add pool join the mix with fresh hashes, so they count as mix neighbours
            add = pool[c][: round(len(pool[c]) * dose / 100)]
            results["added_rows"][f"c{c}_d{dose}"] = len(add)
            add_emb = t_gpu[torch.from_numpy(add).cuda()]
            add_hash = torch.from_numpy(-(add.astype(np.int64)) - 1).cuda()
            condition(f"add_c{c}_d{dose}", torch.cat([m_gpu, add_emb]), torch.cat([mh_gpu, add_hash]))

    # the pass table per score and dose: A below the line, siblings above, unrelated median change under max_change
    table = {}
    for score in (f"ratio_k{config.k_small}", f"ratio_k{config.k_large}"):
        before = results["conditions"]["before"][score]
        table[score] = {}
        for dose in doses:
            per_topic = {}
            for c in tested:
                after = results["conditions"][f"add_c{c}_d{dose}"][score]
                unrel = [f"c{u}" for u in unrelated]
                change = float(np.median([abs(after[u] - before[u]) / max(before[u], 1e-9) for u in unrel]))
                siblings = [f"c{c}_kind{j}" for j in range(1, len(parts[c]))]
                row = {
                    "a_before": before[f"c{c}_kind0"],
                    "a_after": after[f"c{c}_kind0"],
                    "a_released": after[f"c{c}_kind0"] < lines[score],
                    "siblings_before": [before[s] for s in siblings],
                    "siblings_after": [after[s] for s in siblings],
                    "siblings_kept": all(after[s] >= lines[score] for s in siblings),
                    "unrelated_median_change": change,
                    "unrelated_stable": change < config.max_change,
                }
                row["pass"] = row["a_released"] and row["siblings_kept"] and row["unrelated_stable"]
                per_topic[f"c{c}"] = row
            rows_t = list(per_topic.values())
            if all(r["pass"] for r in rows_t):
                verdict = "pass"
            elif all(r["a_released"] for r in rows_t) and not all(r["siblings_kept"] for r in rows_t):
                verdict = "too_coarse"
            elif not all(r["a_released"] for r in rows_t):
                verdict = "too_insensitive"
            else:
                verdict = "unrelated_moved"
            per_topic["verdict"] = verdict
            table[score][f"d{dose}"] = per_topic
        passing = [d for d in doses if table[score][f"d{d}"]["verdict"] == "pass"]
        table[score]["smallest_passing_dose"] = min(passing) if passing else None
    results["table"] = table
    results["mix_share_changes"] = {
        f"c{c}_d{dose}": {
            g: {
                "before": results["conditions"]["before"]["mix_share"][g],
                "after": results["conditions"][f"add_c{c}_d{dose}"]["mix_share"][g],
            }
            for g in share_ids
            if g.startswith(f"c{c}_")
        }
        for c in tested
        for dose in doses
    }
    out = write_json(results, run / "decision.json")

    for c in chosen:
        print(f"c{c} kinds: " + "; ".join(f"{k['name']} ({k.get('size', k['sample_size'])})" for k in kinds_info[c]))
    for score, t in table.items():
        print(f"{score}: smallest passing dose {t['smallest_passing_dose']}")
        for dose in doses:
            td = t[f"d{dose}"]
            print(f"  dose {dose}%: verdict {td['verdict']}")
            for c in tested:
                r = td[f"c{c}"]
                print(
                    f"    c{c}: A {r['a_before']:.1f} -> {r['a_after']:.1f} (released {r['a_released']}); siblings {[round(x, 1) for x in r['siblings_before']]} -> {[round(x, 1) for x in r['siblings_after']]} (kept {r['siblings_kept']}); unrelated change {r['unrelated_median_change']:.3f}"
                )
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
