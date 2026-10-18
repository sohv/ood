# density-ratio scoring (deployment vs mix classifier, cross-fitted), length-controlled ranking, precision@k.

import numpy as np
import torch
from scipy.sparse import vstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def fit_logistic(x: torch.Tensor, y: torch.Tensor, c: float, max_iter: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    n_pos, n_neg = y.sum(), (1 - y).sum()
    weight = torch.where(y > 0, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
    w = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(1, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")
    # sklearn's C scales the summed loss, so the mean-loss penalty is 1 / (2 C n)
    penalty = 1.0 / (2 * c * len(y))

    def closure():
        opt.zero_grad()
        loss = (
            weight * torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y, reduction="none")
        ).mean() + penalty * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def cross_fit_dense(deploy: np.ndarray, mix: np.ndarray, c: float, folds: int, seed: int) -> np.ndarray:
    fold = np.random.default_rng(seed).permutation(len(deploy)) % folds
    scores = np.zeros(len(deploy))
    for f in range(folds):
        train = np.vstack([deploy[fold != f], mix]).astype(np.float32)
        mu, sd = train.mean(0), train.std(0) + 1e-6
        x = torch.from_numpy((train - mu) / sd).cuda()
        y = torch.cat([torch.ones(int((fold != f).sum())), torch.zeros(len(mix))]).cuda()
        w, b = fit_logistic(x, y, c)
        held = torch.from_numpy(((deploy[fold == f] - mu) / sd).astype(np.float32)).cuda()
        scores[fold == f] = (held @ w + b).cpu().numpy()
    return scores


def cross_fit_tfidf(deploy: list[str], mix: list[str], c: float, folds: int, seed: int) -> np.ndarray:
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000).fit(deploy + mix)
    xd, xm = vec.transform(deploy), vec.transform(mix)
    fold = np.random.default_rng(seed).permutation(len(deploy)) % folds
    scores = np.zeros(len(deploy))
    for f in range(folds):
        x = vstack([xd[fold != f], xm])
        y = np.r_[np.ones(int((fold != f).sum())), np.zeros(len(mix))]
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed).fit(x, y)
        scores[fold == f] = clf.decision_function(xd[fold == f])
    return scores


def length_controlled(scores: np.ndarray, lengths: np.ndarray, n_bins: int = 20) -> np.ndarray:
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(lengths, edges)
    out = np.zeros(len(scores))
    for b in np.unique(bins):
        idx = bins == b
        out[idx] = rankdata(scores[idx]) / idx.sum()
    return out


def precision_at_k(scores: np.ndarray, is_target: np.ndarray, k: int, seed: int = 0) -> float:
    # random jitter breaks ties so equal percentiles do not favour row order
    jitter = np.random.default_rng(seed).random(len(scores)) * 1e-9
    top = np.argsort(-(scores + jitter))[:k]
    return float(is_target[top].mean())
