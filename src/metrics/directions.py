# linear two-reference directions (meandiff, lda, logit) and cross-validated probe scores; higher = more like the second set.

from collections.abc import Callable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def _standardiser(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.r_[a, b].astype(np.float64)
    return pooled.mean(0), pooled.std(0) + 1e-6


def meandiff_direction(a: np.ndarray, b: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    mu, sd = _standardiser(a, b)
    w = ((b - mu) / sd).mean(0) - ((a - mu) / sd).mean(0)
    w /= np.linalg.norm(w)
    return lambda x: ((x.astype(np.float64) - mu) / sd) @ w


def lda_direction(a: np.ndarray, b: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    a, b = a.astype(np.float64), b.astype(np.float64)
    within = np.r_[a - a.mean(0), b - b.mean(0)]
    cov = LedoitWolf(store_precision=False, assume_centered=True).fit(within).covariance_
    w = np.linalg.solve(cov + 1e-6 * np.eye(cov.shape[0]), b.mean(0) - a.mean(0))
    return lambda x: x.astype(np.float64) @ w


def logit_direction(a: np.ndarray, b: np.ndarray, c: float = 0.1, seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    mu, sd = _standardiser(a, b)
    clf = LogisticRegression(C=c, max_iter=5000, random_state=seed)
    clf.fit((np.r_[a, b] - mu) / sd, np.r_[np.zeros(len(a)), np.ones(len(b))])
    return lambda x: clf.decision_function((x.astype(np.float64) - mu) / sd)


def cv_scores(make_model: Callable, x, y: np.ndarray, seed: int, folds: int = 5) -> np.ndarray:
    out = np.zeros(len(y))
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        model = make_model().fit(x[train], y[train])
        out[test] = model.decision_function(x[test])
    return out
