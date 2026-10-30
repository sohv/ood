import torch

from src.interp.kmeans import spherical_kmeans


def test_separates_three_directions():
    gen = torch.Generator().manual_seed(0)
    dirs = torch.eye(8)[:3]
    x = torch.cat([d + 0.05 * torch.randn(50, 8, generator=gen) for d in dirs])
    labels, centres = spherical_kmeans(x, 3, n_init=3, seed=0)
    assert centres.shape == (3, 8)
    groups = [set(labels[i * 50 : (i + 1) * 50].tolist()) for i in range(3)]
    assert all(len(g) == 1 for g in groups)
    assert len(set.union(*groups)) == 3
