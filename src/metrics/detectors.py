# ood detectors (mahalanobis, knn, two-sided), baselines, and guarded length-stratified auroc.

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import LedoitWolf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LENGTH_BANDS = (1, 16, 32, 64, 128, 256, 512, 1025)
MIN_PER_CLASS = 10
MIN_BANDS = 3
MIN_COVERAGE = 0.30


class Mahalanobis:
    def __init__(self, ref: np.ndarray, pca_dims: int = 0, shrinkage: float | None = None):
        ref = ref.astype(np.float64)
        self.mu = ref.mean(0)
        # pca_dims=0 keeps the full space; otherwise project onto the reference's top principal axes
        self.basis = np.linalg.svd(ref - self.mu, full_matrices=False)[2][:pca_dims].T if pca_dims else None
        z = self._project(ref)
        if shrinkage is None:
            cov = LedoitWolf(store_precision=False).fit(z).covariance_
        else:
            emp = np.cov(z, rowvar=False)
            cov = (1 - shrinkage) * emp + shrinkage * np.trace(emp) / emp.shape[0] * np.eye(emp.shape[0])
        self.chol = cho_factor(cov + 1e-6 * np.eye(z.shape[1]), lower=True)

    def _project(self, x: np.ndarray) -> np.ndarray:
        d = x.astype(np.float64) - self.mu
        return d @ self.basis if self.basis is not None else d

    def __call__(self, x: np.ndarray) -> np.ndarray:
        d = self._project(x)
        return (d * cho_solve(self.chol, d.T).T).sum(1)


class KNN:
    def __init__(self, ref: np.ndarray, scale: np.ndarray, k: int = 10):
        self.scale, self.k = scale, k
        self.ref = self._unit(ref)

    def _unit(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32) / self.scale
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sims = self._unit(x) @ self.ref.T
        kth = np.partition(sims, -self.k, axis=1)[:, -self.k]
        return np.sqrt(np.clip(2 - 2 * kth, 0, None))


def tfidf_mix_vs_dolma(mix_texts: list[str], dolma_texts: list[str], seed: int):
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
    x = vectorizer.fit_transform(mix_texts + dolma_texts)
    y = np.r_[np.zeros(len(mix_texts)), np.ones(len(dolma_texts))]
    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed).fit(x, y)
    return lambda texts: clf.predict_proba(vectorizer.transform(texts))[:, 1]


def auroc(ood: np.ndarray, ref: np.ndarray) -> float:
    return float(roc_auc_score(np.r_[np.ones(len(ood)), np.zeros(len(ref))], np.r_[ood, ref]))


def stratified_auroc(
    ood: np.ndarray, ref: np.ndarray, ood_len: np.ndarray, ref_len: np.ndarray, bands=LENGTH_BANDS
) -> dict:
    ood_band, ref_band = np.digitize(ood_len, bands), np.digitize(ref_len, bands)
    total, weight, used, covered, per_band = 0.0, 0, 0, 0, []
    for b in range(1, len(bands)):
        o, r = ood[ood_band == b], ref[ref_band == b]
        if len(o) < MIN_PER_CLASS or len(r) < MIN_PER_CLASS:
            continue
        value = auroc(o, r)
        pairs = len(o) * len(r)
        total, weight, used, covered = total + value * pairs, weight + pairs, used + 1, covered + len(o)
        per_band.append(
            {
                "band": [bands[b - 1], bands[b]],
                "n_ood": len(o),
                "n_ref": len(r),
                "auroc": value,
                "median_len_ratio": float(np.median(ood_len[ood_band == b]) / np.median(ref_len[ref_band == b])),
            }
        )
    coverage = covered / len(ood)
    is_measurable = used >= MIN_BANDS and coverage >= MIN_COVERAGE
    return {
        "auroc": total / weight if is_measurable else None,
        "n_bands": used,
        "coverage": coverage,
        "bands": per_band,
    }


def bootstrap_ci(stat, ood_idx: int, ref_idx: int, n_boot: int, seed: int) -> list[float] | None:
    rng = np.random.default_rng(seed)
    values = [stat(rng.integers(0, ood_idx, ood_idx), rng.integers(0, ref_idx, ref_idx)) for _ in range(n_boot)]
    values = [v for v in values if v is not None]
    if len(values) < 0.9 * n_boot:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    r = [rankdata(v) for v in (x, y, control)]
    resid = [v - np.polyval(np.polyfit(r[2], v, 1), r[2]) for v in r[:2]]
    return float(np.corrcoef(*resid)[0, 1])
