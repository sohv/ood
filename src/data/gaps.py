# rebuilds exp09 v2's review clusters from its stored rows, merges them into gap groups, and draws matched controls.

import numpy as np

from src.metrics.discovery import centroid, cluster


def rebuild_clusters(emb: np.ndarray, reviews: list[dict], n_clusters: int, seed: int) -> dict[str, np.ndarray]:
    # same call exp09 v2 made: k-means on each review group's rows, members kept in review order
    out = {}
    for rev in reviews:
        rows = np.array(rev["rows"])
        labels = cluster(emb[rows], n_clusters, seed)
        for c in range(n_clusters):
            out[f"{rev['arm']}_{rev['kind']}_r{rev['round']}_c{c}"] = rows[labels == c]
    return out


def merge_groups(members: dict[str, np.ndarray], ids: list[str], emb: np.ndarray, threshold: float) -> list[list[str]]:
    # single-linkage on centroid cosine, exp09's same-gap rule
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
