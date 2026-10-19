import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.metrics.detectors import auroc
from src.metrics.directions import cv_scores, lda_direction, logit_direction, meandiff_direction


def _two_gaussians(seed: int = 0, n: int = 200, d: int = 20):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n, d))
    b = rng.normal(0, 1, (n, d))
    b[:, 0] += 2.0
    return a, b


def test_directions_score_second_set_higher():
    a, b = _two_gaussians()
    a_new, b_new = _two_gaussians(seed=1)
    for fit in (meandiff_direction, lda_direction, logit_direction):
        score = fit(a, b)
        assert auroc(score(b_new), score(a_new)) > 0.85, fit.__name__


def test_cv_scores_are_out_of_fold_and_aligned():
    a, b = _two_gaussians()
    x, y = np.r_[a, b], np.r_[np.zeros(len(a)), np.ones(len(b))]
    s = cv_scores(lambda: LogisticRegression(C=0.1, max_iter=2000), x, y, seed=0)
    assert s.shape == (len(y),)
    assert auroc(s[y == 1], s[y == 0]) > 0.85


def test_cv_scores_on_text_pipeline():
    texts = [f"legal court landlord case {i}" for i in range(40)] + [f"recipe pasta sauce cook {i}" for i in range(40)]
    y = np.r_[np.ones(40), np.zeros(40)]
    s = cv_scores(
        lambda: make_pipeline(TfidfVectorizer(), LogisticRegression(C=0.1)), np.array(texts, dtype=object), y, seed=0
    )
    assert auroc(s[y == 1], s[y == 0]) > 0.95
