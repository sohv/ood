# unit tests for the exp11 ranking variants: spline length removal, per-cluster review order, known-ood gating.
# uv run --no-sync -m pytest tests/test_ranking.py -v

import numpy as np

from src.metrics.ranking import cluster_order, known_gate, nearest_in_log_length, spline_residual


def test_spline_residual_removes_pure_length_ranking():
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 1000, 5000)
    scores = np.log(lengths) * 3 + rng.normal(size=5000) * 0.1
    resid = spline_residual(scores, lengths)
    assert abs(np.corrcoef(resid, np.log(lengths))[0, 1]) < 0.05


def test_nearest_in_log_length_picks_closest_rows():
    ref = np.array([1, 2, 4, 8, 16, 32, 64])
    picked = nearest_in_log_length(ref, np.log(8.0), 3)
    assert sorted(ref[picked].tolist()) == [4, 8, 16]


def test_cluster_order_reviews_whole_clusters_by_mean_score():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.5, 0.6])
    labels = np.array([0, 0, 1, 1, 2, 2])
    order = cluster_order(scores, labels, top_k=4)
    assert order.tolist() == [2, 3, 5, 4]
    assert len(set(order.tolist())) == 4


def test_known_gate_drops_rows_near_the_known_set_only():
    rng = np.random.default_rng(0)
    known = rng.normal(size=(20, 8)) + np.array([10.0] + [0.0] * 7)
    near = known[:5] + rng.normal(size=(5, 8)) * 0.01
    far = rng.normal(size=(5, 8)) - np.array([10.0] + [0.0] * 7)
    drop, tau = known_gate(np.vstack([near, far]), known, scale=np.ones(8))
    assert drop[:5].all() and not drop[5:].any() and tau > 0
