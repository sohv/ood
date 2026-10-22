# exp37/exp42 helpers: batched knn density ratio on the gpu, a reusable neighbour table for re-scoring, spherical k-means, text hashing.

import hashlib

import numpy as np
import torch


def text_hash(norm: str) -> int:
    # stable 63-bit hash of normalised text, so exact copies can be masked on the gpu
    return int.from_bytes(hashlib.blake2b(norm.encode(), digest_size=8).digest(), "little") >> 1


def density_ratios_batched(
    q_emb: np.ndarray,
    q_hash: np.ndarray,
    q_self: np.ndarray,
    t_emb: torch.Tensor,
    t_hash: torch.Tensor,
    m_emb: torch.Tensor,
    m_hash: torch.Tensor,
    k: int = 50,
    extra: int = 20,
    chunk: int = 1024,
    return_nearest_mix: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    # same ratio as src.metrics.coverage.density_ratios, vectorised: self (q_self, -1 for none) and exact copies left out
    n_t, n_m = len(t_emb), len(m_emb)
    out = np.empty(len(q_emb))
    nearest = np.empty(len(q_emb))
    for s in range(0, len(q_emb), chunk):
        q = torch.from_numpy(q_emb[s : s + chunk]).to(t_emb.device, t_emb.dtype)
        qh = torch.from_numpy(q_hash[s : s + chunk]).to(t_emb.device)[:, None]
        qs = torch.from_numpy(q_self[s : s + chunk]).to(t_emb.device)[:, None]
        st, it = (q @ t_emb.T).float().topk(k + extra, dim=1)
        sm, im = (q @ m_emb.T).float().topk(k + extra, dim=1)
        st = st.masked_fill((it == qs) | (t_hash[it] == qh), -torch.inf)
        sm = sm.masked_fill(m_hash[im] == qh, -torch.inf)
        nearest[s : s + chunk] = sm.max(1).values.cpu().numpy()
        sims = torch.cat([st, sm], dim=1)
        side = torch.cat([torch.zeros_like(st), torch.ones_like(sm)], dim=1)
        top_s, top_i = sims.topk(k, dim=1)
        valid = torch.isfinite(top_s)
        n_mix = (side.gather(1, top_i) * valid).sum(1)
        n_traffic = valid.sum(1) - n_mix
        out[s : s + chunk] = (((n_traffic + 1) / n_t) / ((n_mix + 1) / n_m)).cpu().numpy()
    return (out, nearest) if return_nearest_mix else out


def union_knn(
    t_np: np.ndarray,
    t_hash_np: np.ndarray,
    t_emb: torch.Tensor,
    t_hash: torch.Tensor,
    m_emb: torch.Tensor,
    m_hash: torch.Tensor,
    k: int = 50,
    extra: int = 20,
    chunk: int = 1024,
) -> np.ndarray:
    # each traffic row's k nearest valid neighbours over traffic and mix: traffic index, -1 for a mix row, -2 for none
    out = np.empty((len(t_np), k), dtype=np.int32)
    for s in range(0, len(t_np), chunk):
        q = torch.from_numpy(t_np[s : s + chunk]).to(t_emb.device, t_emb.dtype)
        qh = torch.from_numpy(t_hash_np[s : s + chunk]).to(t_emb.device)[:, None]
        qs = torch.arange(s, s + len(q), device=t_emb.device)[:, None]
        st, it = (q @ t_emb.T).float().topk(k + extra, dim=1)
        sm, im = (q @ m_emb.T).float().topk(k + extra, dim=1)
        st = st.masked_fill((it == qs) | (t_hash[it] == qh), -torch.inf)
        sm = sm.masked_fill(m_hash[im] == qh, -torch.inf)
        top_s, top_i = torch.cat([st, sm], dim=1).topk(k, dim=1)
        ids = torch.cat([it, torch.full_like(im, -1)], dim=1).gather(1, top_i)
        out[s : s + chunk] = ids.masked_fill(~torch.isfinite(top_s), -2).cpu().numpy()
    return out


def ratios_from_knn(nbr: np.ndarray, moved: np.ndarray, n_mix: int) -> np.ndarray:
    # density ratio for every traffic row after the moved traffic rows join the mix; moved rows' own values are meaningless
    is_t = nbr >= 0
    as_mix = (nbr == -1) | (is_t & moved[np.where(is_t, nbr, 0)])
    n_traffic = (is_t.sum(1) - (as_mix & is_t).sum(1)).astype(np.float64)
    n_m = as_mix.sum(1).astype(np.float64)
    size_t, size_m = len(moved) - moved.sum(), n_mix + moved.sum()
    return ((n_traffic + 1) / size_t) / ((n_m + 1) / size_m)


def spherical_kmeans(
    emb: torch.Tensor, k: int, iters: int, seed: int, chunk: int = 65536
) -> tuple[np.ndarray, torch.Tensor]:
    # cosine k-means on unit vectors: k-means++ seeding on a random subsample, centroids renormalised each step
    g = torch.Generator().manual_seed(seed)
    sub = emb[torch.randperm(len(emb), generator=g)[: min(len(emb), 50 * k)].to(emb.device)].float()
    gd = torch.Generator(device=emb.device).manual_seed(seed)
    picks = torch.empty(k, dtype=torch.long, device=emb.device)
    picks[0] = torch.randint(len(sub), (1,), generator=gd, device=emb.device)
    best = sub @ sub[picks[0]]
    for j in range(1, k):
        picks[j] = torch.multinomial((1 - best).clamp(min=1e-12) ** 2, 1, generator=gd)
        best = torch.maximum(best, sub @ sub[picks[j]])
    cent = sub[picks].clone()
    labels = torch.empty(len(emb), dtype=torch.long, device=emb.device)
    for _ in range(iters):
        for s in range(0, len(emb), chunk):
            labels[s : s + chunk] = (emb[s : s + chunk].float() @ cent.T).argmax(1)
        sums = torch.zeros_like(cent).index_add_(0, labels, emb.float())
        counts = torch.bincount(labels, minlength=k)
        # an emptied cluster keeps its old centroid
        cent = torch.where(counts[:, None] > 0, torch.nn.functional.normalize(sums, dim=1), cent)
    return labels.cpu().numpy(), cent
