# word / text baseline detectors. Every score is "higher = more out-of-distribution for the post-training mix".
# one-sided: fitted on the mix only (kNN distance or density). two-sided: mix vs Dolma (classifier or kNN difference).

import numpy as np
from scipy import sparse
from sklearn.covariance import LedoitWolf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

K_MAX = 200

REPS = {
    "tfidf1": dict(analyzer="word", ngram_range=(1, 1)),
    "tfidf12": dict(analyzer="word", ngram_range=(1, 2)),
    "char35": dict(analyzer="char_wb", ngram_range=(3, 5)),
}


def make_vectorizer(rep: str, min_df: int = 2, sublinear: bool = True, analyzer: str | None = None):
    kw = dict(REPS[rep])
    if analyzer:
        kw["analyzer"] = analyzer
    # max_features as in src/metrics/detectors.py:56
    return TfidfVectorizer(sublinear_tf=sublinear, min_df=min_df, max_features=200_000, **kw)


def topk_sims(x, ref, k_max: int = K_MAX, chunk: int = 2000) -> np.ndarray:
    """Top-k_max cosine similarities of each row of x to rows of ref, sorted descending. Rows are L2-normalised."""
    out = []
    for i in range(0, x.shape[0], chunk):
        s = x[i : i + chunk] @ ref.T
        s = s.toarray() if sparse.issparse(s) else np.asarray(s)
        s = s.astype(np.float32)
        kk = min(k_max, s.shape[1])
        part = np.partition(s, -kk, axis=1)[:, -kk:]
        out.append(-np.sort(-part, axis=1))
    return np.vstack(out)


def knn_dist(top: np.ndarray, k: int) -> np.ndarray:
    # distance form of src/metrics/detectors.py:49-52 (sqrt(2 - 2 * k-th cosine))
    return np.sqrt(np.clip(2 - 2 * top[:, k - 1], 0, None))


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# ---- unigram language model densities
_ANALYZER = CountVectorizer(lowercase=True).build_analyzer()  # same tokens as the default TfidfVectorizer


def _counts(texts):
    from collections import Counter

    c = Counter()
    for t in texts:
        c.update(_ANALYZER(t))
    return c


class UnigramLM:
    def __init__(self, texts, alpha: float, vocab_size: int | None = None):
        self.c = _counts(texts)
        self.n = sum(self.c.values())
        self.alpha = alpha
        self.v = (vocab_size or len(self.c)) + 1

    def nll(self, texts) -> np.ndarray:
        denom = np.log(self.n + self.alpha * self.v)
        out = np.empty(len(texts))
        oov = denom - np.log(self.alpha)
        for i, t in enumerate(texts):
            toks = _ANALYZER(t)
            if not toks:
                out[i] = oov
                continue
            out[i] = np.mean([denom - np.log(self.c.get(w, 0) + self.alpha) for w in toks])
        return out


def maha_scores(fit: np.ndarray, x: np.ndarray, shrinkage) -> np.ndarray:
    fit = fit.astype(np.float64)
    mu = fit.mean(0)
    if shrinkage == "lw":
        cov = LedoitWolf(store_precision=False).fit(fit).covariance_
    else:
        emp = np.cov(fit, rowvar=False)
        cov = (1 - shrinkage) * emp + shrinkage * np.trace(emp) / emp.shape[0] * np.eye(emp.shape[0])
    prec = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    d = x.astype(np.float64) - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


def lr_scores(x_fit_mix, x_fit_ref, xs: dict, c: float, seed: int = 0, proba: bool = False) -> dict:
    x = sparse.vstack([x_fit_mix, x_fit_ref]) if sparse.issparse(x_fit_mix) else np.vstack([x_fit_mix, x_fit_ref])
    y = np.r_[np.zeros(x_fit_mix.shape[0]), np.ones(x_fit_ref.shape[0])]
    clf = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(x, y)
    f = (lambda z: clf.predict_proba(z)[:, 1]) if proba else clf.decision_function
    return {k: f(v) for k, v in xs.items()}, clf


# ---- all arms for one setting of hyperparameters
def compute_arms(params: dict, mix: list[str], ref: list[str], tests: dict[str, list[str]],
                 emb: dict, lengths: dict, lr_mix: list[str] | None = None, return_models: bool = False) -> dict:
    """params: arm -> hyperparameters. mix: one-sided fit texts. ref: Dolma texts (first 400 words).
    emb: {embedder: {"mix": arr, "ref": arr, pool: arr}}. lengths: pool -> n_content_tokens.
    lr_mix: texts for the mix side of the two-sided classifiers (default: mix)."""
    lr_mix = lr_mix if lr_mix is not None else mix
    out = {p: {} for p in tests}
    models = {}
    for p in tests:
        out[p]["length"] = np.asarray(lengths[p], dtype=float)

    # Sohan's exact baseline: src/metrics/detectors.py:55-60 (vectoriser on mix + dolma, C = 0.1, P(dolma))
    if "tfidf1_lr_sohan" in params:
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
        xall = v.fit_transform(lr_mix + ref)
        sc, clf = lr_scores(xall[: len(lr_mix)], xall[len(lr_mix):], {p: v.transform(t) for p, t in tests.items()},
                            0.1, proba=True)
        for p in tests:
            out[p]["tfidf1_lr_sohan"] = sc[p]
        models["tfidf1_lr_sohan"] = (v, clf)

    for rep in REPS:
        a = f"{rep}_knn"
        if a in params:
            hp = params[a]
            v = make_vectorizer(rep, hp["min_df"], hp["sublinear"], hp.get("analyzer"))
            xm = v.fit_transform(mix)
            for p, t in tests.items():
                out[p][a] = knn_dist(topk_sims(v.transform(t), xm, hp["k"]), hp["k"])
        a = f"{rep}_dknn"
        if a in params:
            hp = params[a]
            v = make_vectorizer(rep, hp["min_df"], hp["sublinear"], hp.get("analyzer"))
            v.fit(mix + ref)
            xm, xr = v.transform(mix), v.transform(ref)
            for p, t in tests.items():
                xt = v.transform(t)
                out[p][a] = knn_dist(topk_sims(xt, xm, hp["k"]), hp["k"]) - knn_dist(topk_sims(xt, xr, hp["k"]), hp["k"])
        a = f"{rep}_lr"
        if a in params:
            hp = params[a]
            v = make_vectorizer(rep, hp["min_df"], hp["sublinear"], hp.get("analyzer"))
            v.fit(lr_mix + ref)
            sc, clf = lr_scores(v.transform(lr_mix), v.transform(ref), {p: v.transform(t) for p, t in tests.items()}, hp["C"])
            for p in tests:
                out[p][a] = sc[p]
            models[a] = (v, clf)

    for e in emb:
        E = emb[e]
        if f"{e}_knn" in params:
            k = params[f"{e}_knn"]["k"]
            for p in tests:
                out[p][f"{e}_knn"] = knn_dist(topk_sims(E[p], E["mix"], k), k)
        if f"{e}_dknn" in params:
            k = params[f"{e}_dknn"]["k"]
            for p in tests:
                out[p][f"{e}_dknn"] = knn_dist(topk_sims(E[p], E["mix"], k), k) - knn_dist(topk_sims(E[p], E["ref"], k), k)
        if f"{e}_lr" in params:
            sc, clf = lr_scores(E["lr_mix"] if "lr_mix" in E else E["mix"], E["ref"], {p: E[p] for p in tests},
                                params[f"{e}_lr"]["C"])
            for p in tests:
                out[p][f"{e}_lr"] = sc[p]
        if f"{e}_maha" in params:
            s = params[f"{e}_maha"]["shrinkage"]
            for p in tests:
                out[p][f"{e}_maha"] = maha_scores(E["mix"], E[p], s)
        if f"{e}_dmaha" in params:
            s = params[f"{e}_dmaha"]["shrinkage"]
            for p in tests:
                out[p][f"{e}_dmaha"] = maha_scores(E["mix"], E[p], s) - maha_scores(E["ref"], E[p], s)

    if "uni_lm" in params:
        lm = UnigramLM(mix, params["uni_lm"]["alpha"])
        for p, t in tests.items():
            out[p]["uni_lm"] = lm.nll(t)
    if "uni_nb" in params:
        al = params["uni_nb"]["alpha"]
        vocab = len(set(_counts(lr_mix)) | set(_counts(ref)))
        lm_m, lm_r = UnigramLM(lr_mix, al, vocab), UnigramLM(ref, al, vocab)
        for p, t in tests.items():
            out[p]["uni_nb"] = lm_m.nll(t) - lm_r.nll(t)
    return (out, models) if return_models else out


ONE_SIDED = ["length", "uni_lm", "tfidf1_knn", "tfidf12_knn", "char35_knn", "minilm_knn", "mminilm_knn",
             "minilm_maha", "mminilm_maha"]
TWO_SIDED = ["tfidf1_lr_sohan", "tfidf1_lr", "tfidf12_lr", "char35_lr", "minilm_lr", "mminilm_lr",
             "tfidf1_dknn", "tfidf12_dknn", "char35_dknn", "minilm_dknn", "mminilm_dknn", "uni_nb",
             "minilm_dmaha", "mminilm_dmaha"]
ALL_ARMS = ONE_SIDED + TWO_SIDED
