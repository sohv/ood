# unit tests for exp37's batched density ratio and spherical k-means against the reference implementations.
# uv run -m pytest tests/test_gap_pipeline.py -v

import numpy as np
import pytest
import torch

from src.metrics.coverage import density_ratios
from src.metrics.gap_pipeline import density_ratios_batched, spherical_kmeans, text_hash

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a gpu")


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@needs_gpu
def test_batched_ratio_matches_reference_with_self_and_copies():
    rng = np.random.default_rng(0)
    t_emb, m_emb = unit(rng.normal(size=(400, 16))), unit(rng.normal(size=(600, 16)))
    t_norm = [f"t{i}" for i in range(400)]
    m_norm = [f"m{i}" for i in range(600)]
    # exact copies: traffic row 5 also appears in the mix and as traffic row 6
    m_norm[10], t_norm[6] = "t5", "t5"
    m_emb[10], t_emb[6] = t_emb[5], t_emb[5]
    q = np.arange(0, 40)
    ref = density_ratios(t_emb[q], [t_norm[i] for i in q], q.tolist(), t_emb, t_norm, m_emb, m_norm, k=10)
    th = torch.tensor([text_hash(x) for x in t_norm]).cuda()
    mh = torch.tensor([text_hash(x) for x in m_norm]).cuda()
    got = density_ratios_batched(
        t_emb[q],
        np.array([text_hash(t_norm[i]) for i in q]),
        q,
        torch.from_numpy(t_emb).cuda(),
        th,
        torch.from_numpy(m_emb).cuda(),
        mh,
        k=10,
        chunk=16,
    )
    assert np.allclose(got, ref)


@needs_gpu
def test_spherical_kmeans_recovers_separated_groups():
    rng = np.random.default_rng(0)
    centres = unit(rng.normal(size=(3, 8)) * 10)
    x = unit(np.vstack([c + rng.normal(size=(100, 8)) * 0.05 for c in centres]))
    labels, _ = spherical_kmeans(torch.from_numpy(x).cuda(), k=3, iters=10, seed=0)
    groups = [set(labels[i * 100 : (i + 1) * 100]) for i in range(3)]
    assert all(len(g) == 1 for g in groups) and len(set.union(*groups)) == 3


def test_text_hash_is_stable_and_nonnegative():
    assert text_hash("abc") == text_hash("abc") and text_hash("abc") != text_hash("abd") and text_hash("x") >= 0
