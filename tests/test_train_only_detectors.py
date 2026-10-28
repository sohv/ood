# the gpu detectors match the project's cpu mahalanobis and knn on random data.

import numpy as np
import torch

from src.interp.train_only_detectors import GpuMahalanobis, knn_distances
from src.metrics.detectors import KNN, Mahalanobis


def test_mahalanobis_matches_cpu() -> None:
    rng = np.random.default_rng(0)
    ref, x = rng.normal(size=(300, 20)), rng.normal(size=(50, 20)) * 1.5
    cpu = Mahalanobis(ref)(x)
    gpu = GpuMahalanobis(torch.tensor(ref))(torch.tensor(x)).numpy()
    assert np.allclose(cpu, gpu, rtol=1e-4)


def test_mahalanobis_pca_matches_cpu_ranking() -> None:
    rng = np.random.default_rng(1)
    ref, x = rng.normal(size=(300, 20)), rng.normal(size=(50, 20))
    cpu = Mahalanobis(ref, pca_dims=5)(x)
    gpu = GpuMahalanobis(torch.tensor(ref), pca_dims=5)(torch.tensor(x)).numpy()
    assert np.allclose(cpu, gpu, rtol=1e-4)


def test_knn_matches_cpu() -> None:
    rng = np.random.default_rng(2)
    ref, x = rng.normal(size=(200, 16)), rng.normal(size=(40, 16))
    sd = ref.std(0, ddof=1) + 1e-6
    for k in (1, 10):
        cpu = KNN(ref, sd, k=k)(x)
        gpu = knn_distances(torch.tensor(ref), torch.tensor(x), ks=(1, 10))[k].numpy()
        assert np.allclose(cpu, gpu, atol=1e-4)


def test_tfidf_knn_copy_is_nearest() -> None:
    from src.interp.train_only_detectors import tfidf_knn_distances

    ref = ["how do i bake bread", "write a poem about rain", "fix my python loop", "bread recipe with yeast"]
    d = tfidf_knn_distances(ref, ["write a poem about rain", "quantum field theory"], ks=(1,), min_df=1)[1]
    assert d[0] < 1e-3 and d[1] > 1.4
