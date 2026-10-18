# unit tests for the exp01 detectors, stratified auroc guard, and pool helpers.
# uv run -m pytest tests/test_detectors.py -v

import numpy as np
import pandas as pd

from src.data.pools import MembershipIndex, allocate_by_share, family_of
from src.metrics.detectors import KNN, Mahalanobis, auroc, partial_spearman, stratified_auroc


def test_auroc_direction_is_ood_higher():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0


def test_mahalanobis_and_knn_flag_shifted_points():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 20))
    near, far = rng.normal(size=(100, 20)), rng.normal(size=(100, 20))
    far[:, 0] += 6
    assert auroc(Mahalanobis(ref)(far), Mahalanobis(ref)(near)) > 0.99
    assert auroc(Mahalanobis(ref, pca_dims=10)(far), Mahalanobis(ref, pca_dims=10)(near)) > 0.9
    assert auroc(Mahalanobis(ref, shrinkage=0.5)(far), Mahalanobis(ref, shrinkage=0.5)(near)) > 0.99
    # cosine knn needs a reference with a mean direction to move away from
    knn = KNN(ref + 5, scale=np.ones(20))
    far[:, 0] += 9
    assert auroc(knn(far + 5), knn(near + 5)) > 0.99


def test_stratified_auroc_removes_pure_length_signal():
    rng = np.random.default_rng(0)
    ood_len, ref_len = rng.integers(16, 512, 400), rng.integers(16, 512, 400)
    result = stratified_auroc(ood_len.astype(float), ref_len.astype(float), ood_len, ref_len)
    assert result["auroc"] is not None and abs(result["auroc"] - 0.5) < 0.1


def test_stratified_auroc_is_unmeasurable_without_length_overlap():
    rng = np.random.default_rng(0)
    result = stratified_auroc(rng.normal(size=100), rng.normal(size=100), np.full(100, 600), np.full(100, 20))
    assert result["auroc"] is None


def test_allocate_by_share_sums_to_total():
    counts = allocate_by_share(pd.Series([7, 2, 1], index=["a", "b", "c"]), 11)
    assert counts.sum() == 11 and counts["a"] >= counts["b"] >= counts["c"]


def test_family_strips_numeric_suffix():
    assert family_of("correct-python-sft-187k-request-70-606") == "correct-python-sft-187k-request"


def test_membership_index_fires_on_known_positive():
    ref = ["the quick brown fox jumps over the lazy dog and then runs far away into the woods tonight"]
    index = MembershipIndex(ref)
    assert index.is_member(
        "THE quick brown   fox jumps over the lazy dog and then runs far away into the woods tonight"
    )
    assert index.is_member("prefix words the quick brown fox jumps over the lazy dog and then runs far")
    assert not index.is_member(
        "an entirely different sentence about cooking pasta with garlic and olive oil for dinner"
    )


def test_partial_spearman_removes_shared_length_driver():
    rng = np.random.default_rng(0)
    length = rng.uniform(1, 100, 2000)
    x, y = length + rng.normal(size=2000), length + rng.normal(size=2000)
    assert abs(partial_spearman(x, y, length)) < 0.1
    assert partial_spearman(x, -x + rng.normal(scale=0.1, size=2000), length) < -0.5
