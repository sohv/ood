# gpu mahalanobis (ledoit-wolf, optional pca) and multi-k cosine knn scored against a training-only reference.

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def ledoit_wolf_cov(x: torch.Tensor) -> torch.Tensor:
    # same estimator as sklearn's LedoitWolf on centred data, in float64
    n, p = x.shape
    emp = x.T @ x / n
    x2 = x**2
    trace_terms = x2.sum(0) / n
    mu = trace_terms.sum() / p
    beta_ = (x2.T @ x2).sum()
    delta_ = ((x.T @ x) ** 2).sum() / n**2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * trace_terms.sum() + p * mu**2) / p
    shrink = 0.0 if delta == 0 else float(min(beta, delta) / delta)
    return (1 - shrink) * emp + shrink * mu * torch.eye(p, dtype=x.dtype, device=x.device)


class GpuMahalanobis:
    def __init__(self, ref: torch.Tensor, pca_dims: int = 0):
        ref = ref.double()
        self.mu = ref.mean(0)
        centred = ref - self.mu
        self.basis = None
        if pca_dims:
            eigvals, eigvecs = torch.linalg.eigh(centred.T @ centred)
            self.basis = eigvecs[:, -pca_dims:]
            centred = centred @ self.basis
        cov = ledoit_wolf_cov(centred)
        self.chol = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = x.double() - self.mu
        if self.basis is not None:
            d = d @ self.basis
        z = torch.linalg.solve_triangular(self.chol, d.T, upper=False)
        return (z**2).sum(0)


def knn_distances(
    ref: torch.Tensor, x: torch.Tensor, ks: tuple[int, ...], scale: bool = True, chunk: int = 4096
) -> dict[int, torch.Tensor]:
    # cosine distance to the k-th nearest reference row, for every k at once; scale=True divides by the reference std first
    ref, x = ref.float(), x.float()
    if scale:
        sd = ref.std(0) + 1e-6
        ref, x = ref / sd, x / sd
    ref = torch.nn.functional.normalize(ref, dim=1)
    x = torch.nn.functional.normalize(x, dim=1)
    out = {k: [] for k in ks}
    for start in range(0, len(x), chunk):
        top = (x[start : start + chunk] @ ref.T).topk(max(ks), dim=1).values
        for k in ks:
            out[k].append(torch.sqrt(torch.clamp(2 - 2 * top[:, k - 1], min=0)))
    return {k: torch.cat(v) for k, v in out.items()}


def tfidf_knn_distances(
    ref_texts: list[str], texts: list[str], ks: tuple[int, ...], min_df: int = 2, chunk: int = 2000
) -> dict[int, np.ndarray]:
    # cosine distance to the k-th nearest reference text in tf-idf space, vectoriser fitted on the reference only
    vec = TfidfVectorizer(sublinear_tf=True, min_df=min_df, ngram_range=(1, 2), max_features=200_000)
    r = vec.fit_transform(ref_texts)
    q = vec.transform(texts)
    out = {k: [] for k in ks}
    for start in range(0, q.shape[0], chunk):
        sims = torch.tensor(
            (q[start : start + chunk] @ r.T).toarray(), device="cuda" if torch.cuda.is_available() else "cpu"
        )
        top = sims.topk(max(ks), dim=1).values
        for k in ks:
            out[k].append(torch.sqrt(torch.clamp(2 - 2 * top[:, k - 1], min=0)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
