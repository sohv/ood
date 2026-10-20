# measured coverage: cosine similarity of each prompt to its nearest post-training prompt (embedding and tf-idf).

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def nearest_cosine_embedding(
    model: SentenceTransformer, reference: np.ndarray, texts: list[str], chunk: int = 4096
) -> np.ndarray:
    ref = torch.from_numpy(reference).cuda()
    out = []
    for start in range(0, len(texts), chunk):
        q = torch.from_numpy(
            model.encode(texts[start : start + chunk], batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        ).cuda()
        out.append((q @ ref.T).max(dim=1).values.cpu().numpy())
    return np.concatenate(out)


def nearest_cosine_tfidf(reference_texts: list[str], pools: dict[str, list[str]]) -> dict[str, np.ndarray]:
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=3, max_features=300_000).fit(reference_texts)
    ref = vec.transform(reference_texts).T.tocsr()
    return {name: (vec.transform(texts) @ ref).max(axis=1).toarray().ravel() for name, texts in pools.items()}


def density_ratios(
    q_emb: np.ndarray,
    q_norm: list[str],
    q_self: list[int],
    t_emb: np.ndarray,
    t_norm: list[str],
    m_emb: np.ndarray,
    m_norm: list[str],
    k: int = 50,
    extra: int = 20,
) -> np.ndarray:
    # knn density ratio traffic/mix: ((t+1)/|T|) / ((m+1)/|M|) over the k nearest in T and M, self and exact copies left out
    tt, mm = torch.from_numpy(t_emb).float().cuda(), torch.from_numpy(m_emb).float().cuda()
    q = torch.from_numpy(q_emb).float().cuda()
    st, it = (q @ tt.T).topk(min(k + extra, len(t_emb)), dim=1)
    sm, im = (q @ mm.T).topk(min(k + extra, len(m_emb)), dim=1)
    st, it, sm, im = st.cpu().numpy(), it.cpu().numpy(), sm.cpu().numpy(), im.cpu().numpy()
    out = np.empty(len(q_emb))
    for j in range(len(q_emb)):
        cand = [(s, 0) for s, i in zip(st[j], it[j]) if i != q_self[j] and t_norm[i] != q_norm[j]]
        cand += [(s, 1) for s, i in zip(sm[j], im[j]) if m_norm[i] != q_norm[j]]
        top = sorted(cand, key=lambda c: -c[0])[:k]
        n_m = sum(side for _, side in top)
        out[j] = ((len(top) - n_m + 1) / len(t_emb)) / ((n_m + 1) / len(m_emb))
    return out
