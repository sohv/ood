# unit tests for exp80's neighbour count with a fair split of ties at the k-th similarity.
# uv run -m pytest tests/test_fair_count.py -v

import numpy as np
import torch

from src.metrics.fair_count import density_ratios_fair
from src.metrics.gap_pipeline import density_ratios_batched


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def run(t, t_hash, m, m_hash, k, extra):
    return density_ratios_fair(
        t, t_hash, np.arange(len(t)), torch.from_numpy(t), torch.from_numpy(t_hash), torch.from_numpy(m),
        torch.from_numpy(m_hash), k=k, extra=extra,
    )  # fmt: skip


def test_without_ties_fair_equals_standard_and_exp37_count():
    rng = np.random.default_rng(0)
    t, m = unit(rng.normal(size=(60, 16))), unit(rng.normal(size=(90, 16)))
    t_hash, m_hash = np.arange(60, dtype=np.int64), np.arange(1000, 1090, dtype=np.int64)
    out = run(t, t_hash, m, m_hash, k=10, extra=20)
    ref = density_ratios_batched(
        t, t_hash, np.arange(60), torch.from_numpy(t), torch.from_numpy(t_hash), torch.from_numpy(m),
        torch.from_numpy(m_hash), k=10, extra=20,
    )  # fmt: skip
    assert np.allclose(out["ratio_fair"], ref) and np.allclose(out["ratio_std"], ref)
    assert out["tie_decided"].sum() == 0


def test_tied_boundary_is_split_by_tied_counts():
    # query 0 has 3 identical traffic copies and 1 identical mix copy (different hashes), all at similarity 1
    base = unit(np.ones((1, 4)))
    other = unit(np.array([[1.0, -1.0, 0.0, 0.0]]))
    t = np.vstack([base, base, base, base, other])
    m = np.vstack([base, other, other])
    t_hash, m_hash = np.arange(5, dtype=np.int64), np.arange(100, 103, dtype=np.int64)
    out = run(t, t_hash, m, m_hash, k=2, extra=3)
    # 2 slots, 4 tied rows (3 traffic, 1 mix): 1.5 traffic and 0.5 mix
    expected = ((1.5 + 1) / 5) / ((0.5 + 1) / 3)
    assert np.isclose(out["ratio_fair"][0], expected)
    assert out["tie_decided"][0]
    # the two tie orders bound the fair split: all slots to traffic (2, 0) or the mix first (1, 1)
    assert np.isclose(out["ratio_traffic_first"][0], ((2 + 1) / 5) / ((0 + 1) / 3))
    assert np.isclose(out["ratio_mix_first"][0], ((1 + 1) / 5) / ((1 + 1) / 3))


def test_self_and_exact_copies_are_left_out():
    rng = np.random.default_rng(1)
    t = unit(rng.normal(size=(20, 8)))
    m = np.vstack([t[:1], unit(rng.normal(size=(30, 8)))])
    t_hash = np.arange(20, dtype=np.int64)
    m_hash = np.r_[t_hash[:1], np.arange(500, 530)].astype(np.int64)
    out = run(t, t_hash, m, m_hash, k=5, extra=10)
    ref = density_ratios_batched(
        t, t_hash, np.arange(20), torch.from_numpy(t), torch.from_numpy(t_hash), torch.from_numpy(m),
        torch.from_numpy(m_hash), k=5, extra=10,
    )  # fmt: skip
    assert np.allclose(out["ratio_std"], ref)
    # t[0] sits on both sides, so for any query whose k-th neighbour it is, traffic and mix tie there
    tied = out["tie_decided"]
    assert np.allclose(out["ratio_fair"][~tied], ref[~tied])
    assert not tied[0]
