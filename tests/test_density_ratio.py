# unit tests for density_ratio_needles density-ratio scoring, length-controlled ranking and precision@k.
# uv run --no-sync -m pytest tests/test_density_ratio.py -v

import numpy as np
import pytest
import torch

from src.metrics.density_ratio import cross_fit_dense, length_controlled, precision_at_k

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a gpu")


@needs_gpu
def test_cross_fit_dense_ranks_rows_absent_from_the_mix_first():
    rng = np.random.default_rng(0)
    mix = rng.normal(size=(2000, 16))
    deploy = rng.normal(size=(2000, 16))
    # a large shift, so the test checks plumbing and cross-fitting rather than the method's power
    deploy[:40, 0] += 8
    scores = cross_fit_dense(deploy, mix, c=1.0, folds=5, seed=0)
    assert precision_at_k(scores, np.arange(2000) < 40, 40) > 0.8


def test_length_controlled_removes_pure_length_ranking():
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 1000, 5000)
    pct = length_controlled(lengths.astype(float), lengths)
    assert abs(np.corrcoef(pct, lengths)[0, 1]) < 0.2


def test_precision_at_k_counts_targets_in_top_k():
    assert precision_at_k(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 0, 1, 0]), 2) == 0.5
