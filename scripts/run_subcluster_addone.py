# add_one_test: splits three found gaps into subclusters, adds half of the largest subcluster to the mix side, and checks which scores release it while keeping its siblings and unrelated gaps.
# uv run -m scripts.run_subcluster_addone --run_dir results/raw/261027_add_one_test_subcluster_addone_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --precision_at_depth_dir results/raw/261026_precision_at_depth_labels_v1 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.cluster import KMeans

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class AddOneConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    deep_gap_loop_dir: str = ""
    precision_at_depth_dir: str = ""
    seed: int = 0
    n_topics: int = 3
    n_sub: int = 3
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


def main():
    config = simple_parsing.parse(AddOneConfig)
    assert config.run_dir and config.pipeline_million_dir and config.deep_gap_loop_dir and config.precision_at_depth_dir
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])[:2]
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    line = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())["ratio_line"]
    n_clusters = int(labels.max()) + 1

    # the three largest clusters labelled real gap in precision_at_depth's new draw
    e61 = json.loads((Path(config.precision_at_depth_dir) / "decision.json").read_text())
    new_ids = [int(r["id"][1:]) for r in e61["new_rows"]]
    real = [c for c in new_ids if e61["new_rows"][new_ids.index(c)]["label"] == "real_gap"]
    chosen = sorted(real, key=lambda c: -clusters[c]["size"])[: config.n_topics]
    if config.num_tasks:
        chosen = chosen[:1]
    random_arm = [c for c in clusters if clusters[c]["arm"] == "random"]
    unrelated = [c for c in new_ids if c not in chosen] + random_arm
    LOGGER.info(f"chosen {chosen}; {len(unrelated)} unrelated control topics")

    # subclusters by k-means on the member embeddings; the largest is the one added
    members = {c: np.where(labels == c)[0] for c in chosen}
    subs: dict[int, list[np.ndarray]] = {}
    held: dict[int, np.ndarray] = {}
    added: dict[int, np.ndarray] = {}
    for c in chosen:
        km = KMeans(n_clusters=config.n_sub, random_state=config.seed, n_init=10).fit(
            t_np[members[c]].astype(np.float32)
        )
        parts = sorted((members[c][km.labels_ == j] for j in range(config.n_sub)), key=len, reverse=True)
        subs[c] = parts
        perm = rng.permutation(parts[0])
        added[c], held[c] = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        LOGGER.info(
            f"topic c{c}: subcluster sizes {[len(p) for p in parts]}; added {len(added[c])}, held out {len(held[c])}"
        )

    # scored prompts: every member of the chosen topics plus the controls
    scored = np.unique(np.concatenate([members[c] for c in chosen] + [np.where(labels == c)[0] for c in unrelated]))
    t_gpu, th_gpu = torch.from_numpy(t_np).cuda(), torch.from_numpy(t_hash).cuda()
    m_gpu, mh_gpu = torch.from_numpy(m_np).cuda(), torch.from_numpy(m_hash).cuda()

    line_sample = rng.choice(len(rows), size=config.line_sample, replace=False)

    def ratios(m_emb: torch.Tensor, m_h: torch.Tensor, k: int, idx: np.ndarray = scored) -> np.ndarray:
        return density_ratios_batched(t_np[idx], t_hash[idx], idx, t_gpu, th_gpu, m_emb, m_h, k=k)

    def gap_line(k: int) -> float:
        # deep_gap_loop's rule at this k: 3 times the median of random-subset medians, subsets drawn from a scored traffic sample
        r = ratios(m_gpu, mh_gpu, k, line_sample)
        sub_rng = np.random.default_rng(config.seed)
        medians = [
            float(np.median(r[sub_rng.choice(len(r), size=config.subset_size, replace=False)]))
            for _ in range(config.n_random_subsets)
        ]
        return config.ratio_multiple * float(np.median(medians))

    lines = {f"ratio_k{k}": gap_line(k) for k in (config.k_small, config.k_large)}
    LOGGER.info(f"gap lines {lines}; deep_gap_loop's line at k=50 was {line}")

    def groups(c_added: int | None) -> dict[str, np.ndarray]:
        # score groups: the chosen topics' subclusters (A held out) and whole unrelated topics
        g = {}
        for c in chosen:
            for j, part in enumerate(subs[c]):
                g[f"c{c}_sub{j}"] = held[c] if j == 0 else part
        for c in unrelated:
            g[f"c{c}"] = np.where(labels == c)[0]
        return g

    def summarise(ratio_scored: np.ndarray, grp: dict[str, np.ndarray]) -> dict[str, float]:
        pos = {int(i): n for n, i in enumerate(scored)}
        return {name: float(np.median(ratio_scored[[pos[int(i)] for i in idx]])) for name, idx in grp.items()}

    def sub_labels() -> tuple[np.ndarray, dict[str, int]]:
        # relabel the chosen topics' members by subcluster so the mix share has one centroid per subcluster
        lab = labels.copy()
        ids = {}
        nxt = n_clusters
        for c in chosen:
            for j, part in enumerate(subs[c]):
                lab[part] = nxt
                ids[f"c{c}_sub{j}"] = nxt
                nxt += 1
        for c in unrelated:
            ids[f"c{c}"] = c
        return lab, ids

    lab2, share_ids = sub_labels()
    grp = groups(None)
    results = {"lines": lines, "deep_gap_loop_line": line, "chosen": chosen, "unrelated": unrelated, "conditions": {}}
    results["subclusters"] = {
        f"c{c}": [
            {
                "index": j,
                "size": len(p),
                "role": "added" if j == 0 else "sibling",
                "examples": [rows[i]["norm"][:300] for i in p[: config.n_examples]],
            }
            for j, p in enumerate(subs[c])
        ]
        for c in chosen
    }
    results["added_rows"] = {f"c{c}": len(added[c]) for c in chosen}
    results["held_out_rows"] = {f"c{c}": len(held[c]) for c in chosen}

    def condition(name: str, m_emb: torch.Tensor, m_h: torch.Tensor) -> None:
        share = mix_share(lab2, m_emb, t_np, int(lab2.max()) + 1)
        results["conditions"][name] = {
            f"ratio_k{config.k_small}": summarise(ratios(m_emb, m_h, config.k_small), grp),
            f"ratio_k{config.k_large}": summarise(ratios(m_emb, m_h, config.k_large), grp),
            "mix_share": {g: float(share[i]) for g, i in share_ids.items()},
        }
        LOGGER.info(f"condition {name} scored")

    condition("before", m_gpu, mh_gpu)
    for c in chosen:
        # copies of half of subcluster A join the mix with fresh hashes, so they count as mix neighbours
        add_emb = t_gpu[torch.from_numpy(added[c]).cuda()]
        add_hash = torch.from_numpy(-(added[c].astype(np.int64)) - 1).cuda()
        condition(f"add_c{c}", torch.cat([m_gpu, add_emb]), torch.cat([mh_gpu, add_hash]))

    # the pass table: A below the line, siblings above, unrelated median change under max_change
    table = {}
    for score in (f"ratio_k{config.k_small}", f"ratio_k{config.k_large}"):
        before = results["conditions"]["before"][score]
        table[score] = {}
        for c in chosen:
            after = results["conditions"][f"add_c{c}"][score]
            unrel = [f"c{u}" for u in unrelated]
            change = float(np.median([abs(after[u] - before[u]) / max(before[u], 1e-9) for u in unrel]))
            siblings = [f"c{c}_sub{j}" for j in range(1, config.n_sub)]
            row = {
                "a_before": before[f"c{c}_sub0"],
                "a_after": after[f"c{c}_sub0"],
                "a_released": after[f"c{c}_sub0"] < lines[score],
                "siblings_before": [before[s] for s in siblings],
                "siblings_after": [after[s] for s in siblings],
                "siblings_kept": all(after[s] >= lines[score] for s in siblings),
                "unrelated_median_change": change,
                "unrelated_stable": change < config.max_change,
            }
            row["pass"] = row["a_released"] and row["siblings_kept"] and row["unrelated_stable"]
            table[score][f"c{c}"] = row
        rows_t = list(table[score].values())
        n_pass = sum(r["pass"] for r in rows_t)
        if n_pass == len(rows_t):
            verdict = "pass"
        elif all(r["a_released"] for r in rows_t) and not all(r["siblings_kept"] for r in rows_t):
            verdict = "too_coarse"
        elif not all(r["a_released"] for r in rows_t):
            verdict = "too_insensitive"
        else:
            verdict = "unrelated_moved"
        table[score]["verdict"] = verdict
    results["table"] = table
    results["mix_share_changes"] = {
        f"c{c}": {
            g: {
                "before": results["conditions"]["before"]["mix_share"][g],
                "after": results["conditions"][f"add_c{c}"]["mix_share"][g],
            }
            for g in share_ids
            if g.startswith(f"c{c}_")
        }
        for c in chosen
    }
    out = write_json(results, run / "decision.json")

    for score, t in table.items():
        print(f"{score}: verdict {t['verdict']}")
        for c in chosen:
            r = t[f"c{c}"]
            print(
                f"  c{c}: A {r['a_before']:.1f} -> {r['a_after']:.1f} (released {r['a_released']}); siblings {[round(x, 1) for x in r['siblings_before']]} -> {[round(x, 1) for x in r['siblings_after']]} (kept {r['siblings_kept']}); unrelated change {r['unrelated_median_change']:.3f}"
            )
    for c in chosen:
        ch = results["mix_share_changes"][f"c{c}"]
        print(f"mix share c{c}: " + "; ".join(f"{g} {v['before']:.3f} -> {v['after']:.3f}" for g, v in ch.items()))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
