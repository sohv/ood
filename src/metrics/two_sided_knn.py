# two-sided kNN counts (traffic vs mix neighbours) in dense embedding or sparse tf-idf space, sparse nearest rows, and the count_vs_tfidf tf-idf word score.
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


def knn_counts_dense(q, q_hash, ref, ref_hash, ref_side, k, extra, chunk=1024) -> tuple[np.ndarray, np.ndarray]:
    # traffic and mix counts among the k nearest reference rows, exact copies (and so self) left out
    n_t, n_m = np.zeros(len(q)), np.zeros(len(q))
    for s in range(0, len(q), chunk):
        qq = torch.from_numpy(q[s : s + chunk]).to(ref.device, ref.dtype)
        sims, idx = (qq @ ref.T).float().topk(k + extra, dim=1)
        qh = torch.from_numpy(q_hash[s : s + chunk]).to(ref.device)[:, None]
        sims = sims.masked_fill(ref_hash[idx] == qh, -torch.inf)
        top_s, top_i = sims.topk(k, dim=1)
        side = ref_side[idx.gather(1, top_i)] * torch.isfinite(top_s)
        valid = torch.isfinite(top_s).sum(1)
        n_m[s : s + chunk] = side.sum(1).cpu().numpy()
        n_t[s : s + chunk] = (valid - side.sum(1)).cpu().numpy()
    return n_t, n_m


def knn_counts_sparse(q, q_hash, ref, ref_hash, ref_side, k, extra, chunk=256) -> tuple[np.ndarray, np.ndarray]:
    n_t, n_m = np.zeros(q.shape[0]), np.zeros(q.shape[0])
    ref_t = ref.T.tocsr()
    for s in range(0, q.shape[0], chunk):
        sims = (q[s : s + chunk] @ ref_t).toarray()
        idx = np.argpartition(-sims, k + extra, axis=1)[:, : k + extra]
        top = np.take_along_axis(sims, idx, 1)
        top[ref_hash[idx] == q_hash[s : s + chunk, None]] = -np.inf
        order = np.argsort(-top, axis=1)[:, :k]
        keep_i = np.take_along_axis(idx, order, 1)
        is_valid = np.isfinite(np.take_along_axis(top, order, 1))
        mix = (ref_side[keep_i] * is_valid).sum(1)
        n_m[s : s + chunk] = mix
        n_t[s : s + chunk] = is_valid.sum(1) - mix
    return n_t, n_m


def sparse_nearest(q: sp.csr_matrix, ref: sp.csr_matrix, k: int, chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    # indices and cosines of each query's k nearest reference rows (rows are l2-normalised tf-idf)
    ref_t = ref.T.tocsr()
    idx_out = np.empty((q.shape[0], k), dtype=np.int64)
    sim_out = np.empty((q.shape[0], k), dtype=np.float32)
    for s in range(0, q.shape[0], chunk):
        sims = (q[s : s + chunk] @ ref_t).toarray()
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(sims, idx, 1)
        order = np.argsort(-top, axis=1)
        idx_out[s : s + chunk] = np.take_along_axis(idx, order, 1)
        sim_out[s : s + chunk] = np.take_along_axis(top, order, 1)
    return idx_out, sim_out


def log_ratio(n_t: np.ndarray, n_m: np.ndarray, size_t: int, size_m: int) -> np.ndarray:
    return np.log(((n_t + 1) / size_t) / ((n_m + 1) / size_m))


def word_logits(
    t_text: list[str],
    m_text: list[str],
    fold_t: np.ndarray,
    fold_m: np.ndarray,
    chars: int = 1000,
    min_df: int = 5,
    max_features: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    # count_vs_tfidf's two-fold tf-idf logistic regression, traffic 1 vs mix 0, each traffic prompt scored out of fold
    vec = TfidfVectorizer(
        sublinear_tf=True, min_df=min_df, max_features=max_features, ngram_range=(1, 2), dtype=np.float32
    )
    x = vec.fit_transform([t[:chars] for t in t_text] + [t[:chars] for t in m_text])
    y = np.r_[np.ones(len(t_text)), np.zeros(len(m_text))]
    fold = np.r_[fold_t, fold_m]
    out = np.empty(len(t_text))
    for f in (0, 1):
        clf = SGDClassifier(
            loss="log_loss", class_weight="balanced", alpha=1e-6, max_iter=5, tol=None, random_state=seed
        )
        clf.fit(x[fold != f], y[fold != f])
        test = np.where(fold_t == f)[0]
        out[test] = clf.decision_function(x[test])
    return out
