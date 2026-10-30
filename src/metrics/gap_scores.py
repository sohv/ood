# alt_gap_scores alternative two-sided gap scores: dsir hashed n-gram log-likelihood ratio and a fine-tuned minilm classifier two-sample test.

import logging

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def dsir_scores(t_text: list[str], m_text: list[str], n_features: int = 10_000, chars: int = 1000) -> np.ndarray:
    # xie et al. 2023: add-one smoothed hashed unigram+bigram distributions, score = log p_traffic(x) - log p_mix(x)
    vec = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False, norm=None)
    x_t = vec.transform([t[:chars] for t in t_text])
    x_m = vec.transform([t[:chars] for t in m_text])
    log_t = np.log(np.asarray(x_t.sum(0)).ravel() + 1) - np.log(x_t.sum() + n_features)
    log_m = np.log(np.asarray(x_m.sum(0)).ravel() + 1) - np.log(x_m.sum() + n_features)
    return np.asarray(x_t @ (log_t - log_m)).ravel()


class _Head(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.linear = torch.nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        # mean pooling, as the sentence-transformers minilm model does
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.linear(pooled).squeeze(-1)


def finetune_logits(
    t_text: list[str],
    m_text: list[str],
    fold_t: np.ndarray,
    fold_m: np.ndarray,
    model_id: str,
    revision: str,
    n_train: int = 200_000,
    max_len: int = 128,
    batch: int = 256,
    lr: float = 2e-5,
    seed: int = 0,
) -> np.ndarray:
    # classifier two-sample test: per fold, fine-tune the encoder on balanced traffic vs mix from the other fold, score held-out traffic
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    rng = np.random.default_rng(seed)
    out = np.empty(len(t_text))

    def encode(texts: list[str]) -> dict:
        return tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")

    for k in (0, 1):
        torch.manual_seed(seed + k)
        model = _Head(AutoModel.from_pretrained(model_id, revision=revision)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        tr_t = rng.choice(np.where(fold_t != k)[0], size=n_train, replace=False)
        tr_m = rng.choice(np.where(fold_m != k)[0], size=n_train, replace=False)
        texts = [t_text[i] for i in tr_t] + [m_text[i] for i in tr_m]
        labels = np.r_[np.ones(n_train), np.zeros(n_train)]
        order = rng.permutation(len(texts))
        model.train()
        for s in range(0, len(order), batch):
            j = order[s : s + batch]
            enc = encode([texts[i] for i in j])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(enc["input_ids"], enc["attention_mask"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit.float(), torch.from_numpy(labels[j]).float().cuda()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if s // batch % 200 == 0:
                LOGGER.info(f"fold {k} step {s // batch} loss {loss.item():.4f}")
        model.eval()
        test = np.where(fold_t == k)[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(test), 1024):
                enc = encode([t_text[i] for i in test[s : s + 1024]])
                out[test[s : s + 1024]] = model(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy()
        del model, opt
        torch.cuda.empty_cache()
    return out
