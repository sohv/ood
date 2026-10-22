# unit tests for exp37's batched density ratio, exp42's neighbour recount and spherical k-means against reference implementations.
# uv run -m pytest tests/test_gap_pipeline.py -v

import numpy as np
import pytest
import torch

from src.metrics.coverage import density_ratios
from src.metrics.gap_pipeline import density_ratios_batched, ratios_from_knn, spherical_kmeans, text_hash, union_knn

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


@needs_gpu
def test_recount_after_moving_rows_matches_full_recompute():
    rng = np.random.default_rng(1)
    t_emb, m_emb = unit(rng.normal(size=(500, 16))), unit(rng.normal(size=(700, 16)))
    t_norm = [f"t{i}" for i in range(500)]
    m_norm = [f"m{i}" for i in range(700)]
    m_norm[3], m_emb[3] = "t7", t_emb[7]
    th = np.array([text_hash(x) for x in t_norm])
    mh = np.array([text_hash(x) for x in m_norm])
    t_gpu, m_gpu = torch.from_numpy(t_emb).cuda(), torch.from_numpy(m_emb).cuda()
    nbr = union_knn(t_emb, th, t_gpu, torch.from_numpy(th).cuda(), m_gpu, torch.from_numpy(mh).cuda(), k=10, chunk=64)
    moved = np.zeros(500, dtype=bool)
    moved[rng.choice(500, 60, replace=False)] = True
    got = ratios_from_knn(nbr, moved, n_mix=700)
    keep = np.where(~moved)[0]
    t2, m2 = t_emb[keep], np.vstack([m_emb, t_emb[moved]])
    th2, mh2 = th[keep], np.concatenate([mh, th[moved]])
    ref = density_ratios_batched(
        t2,
        th2,
        np.arange(len(keep)),
        torch.from_numpy(t2).cuda(),
        torch.from_numpy(th2).cuda(),
        torch.from_numpy(m2).cuda(),
        torch.from_numpy(mh2).cuda(),
        k=10,
        chunk=64,
    )
    assert np.allclose(got[keep], ref)
    assert np.allclose(
        ratios_from_knn(nbr, np.zeros(500, dtype=bool), n_mix=700),
        density_ratios_batched(
            t_emb,
            th,
            np.arange(500),
            t_gpu,
            torch.from_numpy(th).cuda(),
            m_gpu,
            torch.from_numpy(mh).cuda(),
            k=10,
            chunk=64,
        ),
    )
