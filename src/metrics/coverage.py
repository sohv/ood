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
