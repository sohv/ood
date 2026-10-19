# ranking variants for gap discovery (exp11): spline length removal, length-band references, per-cluster review order, known-ood gating.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from src.metrics.detectors import KNN


def spline_residual(scores: np.ndarray, lengths: np.ndarray, n_knots: int = 8, alpha: float = 1e-3) -> np.ndarray:
    x = np.log(lengths.astype(float))[:, None]
    basis = SplineTransformer(n_knots=n_knots, degree=3).fit_transform(x)
    return scores - Ridge(alpha=alpha).fit(basis, scores).predict(basis)


def nearest_in_log_length(ref_lengths: np.ndarray, target_log_length: float, n: int) -> np.ndarray:
    return np.argsort(np.abs(np.log(ref_lengths.astype(float)) - target_log_length), kind="stable")[:n]


def length_bands(lengths: np.ndarray, n_bands: int) -> np.ndarray:
    edges = np.quantile(lengths, np.linspace(0, 1, n_bands + 1)[1:-1])
    return np.digitize(lengths, edges)


def kmeans_labels(features: np.ndarray, k: int, seed: int, n_init: int = 3) -> np.ndarray:
    z = (features - features.mean(0)) / (features.std(0) + 1e-6)
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(z)


def cluster_order(scores: np.ndarray, labels: np.ndarray, top_k: int) -> np.ndarray:
    # whole clusters in descending mean score, rows inside a cluster in descending row score
    cluster_score = {c: scores[labels == c].mean() for c in np.unique(labels)}
    order: list[int] = []
    for c in sorted(cluster_score, key=lambda c: -cluster_score[c]):
        members = np.where(labels == c)[0]
        order += members[np.argsort(-scores[members], kind="stable")].tolist()
        if len(order) >= top_k:
            break
    return np.array(order[:top_k])


def known_gate(
    x: np.ndarray, known: np.ndarray, scale: np.ndarray, k: int = 10, quantile: float = 0.9
) -> tuple[np.ndarray, float]:
    # tau is the quantile of the confirmed rows' leave-one-out knn distance to each other; self sits at distance 0
    k_loo = min(k, len(known) - 1)
    loo = KNN(known, scale, k=k_loo + 1)(known)
    tau = float(np.quantile(loo, quantile))
    dist = KNN(known, scale, k=min(k, len(known)))(x)
    return dist <= tau, tau
