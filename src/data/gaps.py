# rebuilds iterative_discovery v2's review clusters from its stored rows, merges them into gap groups, and draws matched controls.

import json
from pathlib import Path

import numpy as np

from src.data.io import load_jsonl
from src.metrics.discovery import centroid, cluster, match_gap


def rebuild_clusters(emb: np.ndarray, reviews: list[dict], n_clusters: int, seed: int) -> dict[str, np.ndarray]:
    # same call iterative_discovery v2 made: k-means on each review group's rows, members kept in review order
    out = {}
    for rev in reviews:
        rows = np.array(rev["rows"])
        labels = cluster(emb[rows], n_clusters, seed)
        for c in range(n_clusters):
            out[f"{rev['arm']}_{rev['kind']}_r{rev['round']}_c{c}"] = rows[labels == c]
    return out


def merge_groups(members: dict[str, np.ndarray], ids: list[str], emb: np.ndarray, threshold: float) -> list[list[str]]:
    # single-linkage on centroid cosine, iterative_discovery's same-gap rule
    cents = {i: centroid(emb[members[i]]) for i in ids}
    parent = {i: i for i in ids}

    def find(i: str) -> str:
        while parent[i] != i:
            i = parent[i]
        return i

    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1 :]:
            if float(cents[a] @ cents[b]) >= threshold:
                parent[find(b)] = find(a)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def draw_controls(
    gap_rows: np.ndarray,
    source: np.ndarray,
    band: np.ndarray,
    eligible: np.ndarray,
    per_row: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # per_row controls per gap row, same source and length band, without replacement across all calls sharing eligible
    out = np.empty((len(gap_rows), per_row), dtype=int)
    for k, r in enumerate(gap_rows):
        pool = np.where(eligible & (source == source[r]) & (band == band[r]))[0]
        assert len(pool) >= per_row, f"only {len(pool)} controls for row {r} (source {source[r]}, band {band[r]})"
        pick = rng.choice(pool, size=per_row, replace=False)
        eligible[pick] = False
        out[k] = pick
    return out


def rebuild_activations_vs_words_gaps(emb: np.ndarray, texts: list[str], e20_dir: str) -> list[dict]:
    # activations_vs_words's distinct gaps per seed and arm, with members and the activation-only / word-only flags, checked against activations_vs_words
    e20 = Path(e20_dir)
    c20 = json.loads((e20 / "config.json").read_text())
    reviews = {(r["arm"], r["round"]): np.array(r["rows"]) for r in load_jsonl(e20 / "reviews.jsonl")}
    stored = {c["id"]: c for c in load_jsonl(e20 / "clusters.jsonl")}
    d20 = json.loads((e20 / "decision.json").read_text())
    arms = ("act", "word", "random")
    out = []
    for s in c20["cluster_seeds"]:
        gaps = {arm: [] for arm in arms}
        for arm in arms:
            for rnd in range(1, c20["rounds"] + 1):
                rows = reviews[(arm, rnd)]
                labels = cluster(emb[rows], c20["n_clusters"], s)
                for c in range(c20["n_clusters"]):
                    m = rows[labels == c]
                    st = stored[f"s{s}_{arm}_r{rnd}_c{c}"]
                    examples = [" ".join(texts[i].split())[:200] for i in m[:5]]
                    assert (
                        st["size"] == len(m) and st["examples"] == examples
                    ), f"{st['id']} differs from activations_vs_words"
                    if not st["confirmed"]:
                        continue
                    if st["gap_index"] == len(gaps[arm]):
                        gaps[arm].append({"centroid": centroid(emb[m]), "members": list(m), "name": st["name"]})
                    else:
                        gaps[arm][st["gap_index"]]["members"] += list(m)
        cents = {arm: [g["centroid"] for g in gaps[arm]] for arm in arms}
        for arm in arms:
            other = {"act": "word", "word": "act"}.get(arm)
            for k, g in enumerate(gaps[arm]):
                only = None if other is None else match_gap(g["centroid"], cents[other], c20["same_gap_cos"]) is None
                out.append(
                    {
                        "id": f"s{s}_{arm}_g{k}",
                        "seed": s,
                        "arm": arm,
                        "name": g["name"],
                        "only": only,
                        "members": list(dict.fromkeys(int(i) for i in g["members"])),
                    }
                )
        want = d20["per_seed"][str(s)]
        assert {a: len(gaps[a]) for a in arms} == want[
            "distinct_gaps"
        ], f"seed {s} gap counts differ from activations_vs_words"
        for arm, key in (("act", "act_unique"), ("word", "word_unique")):
            assert (
                sum(bool(x["only"]) for x in out if x["seed"] == s and x["arm"] == arm) == want[key]
            ), f"seed {s} {key}"
    return out
