# content-token pooled residual features (cmean, cmax, clast) at every fourth layer, in one forward pass.

import logging

import numpy as np
import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

LAYERS = list(range(0, 33, 4))
POOLINGS = ("cmean", "cmax", "clast")
USER_HEADER = "<|im_start|>user\n"


def encode_chat(tokenizer, text: str, max_length: int = 1024) -> tuple[list[int], list[bool]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    start = rendered.index(USER_HEADER) + len(USER_HEADER)
    assert rendered[start : start + len(text)] == text, "chat template altered the prompt text"
    end = start + len(text)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    mask = [s < end and e > start for s, e in enc["offset_mapping"]]
    # left truncation keeps the assistant header
    return enc["input_ids"][-max_length:], mask[-max_length:]


def encode_raw(tokenizer, text: str, n_tokens: int) -> tuple[list[int], list[bool]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return ids, [True] * len(ids)


def token_batches(lengths: list[int], max_tokens: int) -> list[list[int]]:
    order = np.argsort(lengths)[::-1]
    batches, current = [], []
    for i in order:
        if current and (len(current) + 1) * lengths[current[0]] > max_tokens:
            batches.append(current)
            current = []
        current.append(int(i))
    return batches + [current] if current else batches


@torch.no_grad()
def extract(
    model, encoded: list[tuple[list[int], list[bool]]], pad_id: int, max_tokens: int = 32768
) -> dict[str, np.ndarray]:
    n = len(encoded)
    out = {p: np.zeros((n, len(LAYERS), model.config.hidden_size), dtype=np.float16) for p in POOLINGS}
    n_content = np.array([sum(m) for _, m in encoded])
    assert (n_content > 0).all(), f"rows with no content tokens: {np.where(n_content == 0)[0][:10]}"
    for batch in tqdm(token_batches([len(ids) for ids, _ in encoded], max_tokens), desc="batches"):
        width = max(len(encoded[i][0]) for i in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        content = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, i in enumerate(batch):
            toks, mask = encoded[i]
            ids[row, : len(toks)] = torch.tensor(toks)
            attn[row, : len(toks)] = 1
            content[row, : len(toks)] = torch.tensor(mask)
        hidden = model(ids.cuda(), attention_mask=attn.cuda(), output_hidden_states=True).hidden_states
        stacked = torch.stack([hidden[l] for l in LAYERS], dim=1).float()
        content = content.cuda()[:, None, :, None]
        counts = content.sum(dim=2)
        cmean = (stacked * content).sum(dim=2) / counts
        cmax = stacked.masked_fill(~content, float("-inf")).amax(dim=2)
        last = (content[:, 0, :, 0] * torch.arange(width, device="cuda")).argmax(dim=1)
        clast = stacked[torch.arange(len(batch)), :, last]
        for name, value in (("cmean", cmean), ("cmax", cmax), ("clast", clast)):
            out[name][batch] = value.cpu().numpy().astype(np.float16)
    out["n_content_tokens"] = n_content
    return out
