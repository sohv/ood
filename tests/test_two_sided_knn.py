import numpy as np
import scipy.sparse as sp
import torch

from src.metrics.two_sided_knn import knn_counts_dense, knn_counts_sparse, log_ratio, sparse_nearest


def _toy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 8)) + 5
    b = rng.normal(size=(20, 8)) - 5
    x = np.concatenate([a, b]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    side = np.r_[np.zeros(20), np.ones(20)]
    hashes = np.arange(40, dtype=np.int64)
    return x, side, hashes


def test_dense_counts_follow_the_cluster_and_skip_self():
    x, side, hashes = _toy()
    n_t, n_m = knn_counts_dense(
        x, hashes, torch.from_numpy(x), torch.from_numpy(hashes), torch.from_numpy(side), k=5, extra=2
    )
    assert (n_t[:20] == 5).all() and (n_m[:20] == 0).all()
    assert (n_m[20:] == 5).all() and (n_t[20:] == 0).all()


def test_sparse_counts_match_dense_on_the_same_vectors():
    x, side, hashes = _toy()
    xs = sp.csr_matrix(x)
    d = knn_counts_dense(x, hashes, torch.from_numpy(x), torch.from_numpy(hashes), torch.from_numpy(side), 5, 2)
    s = knn_counts_sparse(xs, hashes, xs, hashes, side, 5, 2, chunk=7)
    assert np.array_equal(d[0], s[0]) and np.array_equal(d[1], s[1])


def test_sparse_nearest_puts_self_first():
    x, _, _ = _toy()
    xs = sp.csr_matrix(x)
    idx, sim = sparse_nearest(xs, xs, k=3, chunk=6)
    assert (idx[:, 0] == np.arange(40)).all()
    assert np.all(np.diff(sim, axis=1) <= 1e-6)


def test_log_ratio_is_zero_when_shares_match():
    assert np.isclose(log_ratio(np.array([9.0]), np.array([19.0]), 100, 200)[0], 0.0)
