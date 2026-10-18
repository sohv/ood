# exp08 perplexity baseline: mean nll of content tokens 2..n under the dpo model, chat-rendered as for features.
# uv run --no-sync -m scripts.run_controls_perplexity --pools data/processed/exp01:idmix data/processed/exp08:magicoder --output_dir results/raw/261018_exp08_controls_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.interp.features import encode_chat, token_batches
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PerplexityConfig:
    pools: list[str] = field(default_factory=list)
    output_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    seed: int = 0
    max_length: int = 1024
    max_batch_tokens: int = 8192


@torch.no_grad()
def content_nll(model, encoded: list[tuple[list[int], list[bool]]], pad_id: int, max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    nll, n_scored = np.zeros(len(encoded)), np.zeros(len(encoded), dtype=int)
    for batch in tqdm(token_batches([len(ids) for ids, _ in encoded], max_tokens), desc="batches"):
        width = max(len(encoded[i][0]) for i in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        target = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, i in enumerate(batch):
            toks, mask = encoded[i]
            ids[row, : len(toks)] = torch.tensor(toks)
            attn[row, : len(toks)] = 1
            m = torch.tensor(mask)
            # score a content token only if the token before it is content too, which drops the first content token
            target[row, 1 : len(toks)] = m[1:] & m[:-1]
        logits = model(ids.cuda(), attention_mask=attn.cuda()).logits
        rows, cols = torch.nonzero(target.cuda(), as_tuple=True)
        losses = F.cross_entropy(logits[rows, cols - 1].float(), ids.cuda()[rows, cols], reduction="none")
        sums = torch.zeros(len(batch), device="cuda").index_add_(0, rows, losses)
        counts = torch.bincount(rows, minlength=len(batch))
        for row, i in enumerate(batch):
            n_scored[i] = int(counts[row])
            nll[i] = float(sums[row] / counts[row]) if counts[row] > 0 else np.nan
    return nll, n_scored


def main():
    config = simple_parsing.parse(PerplexityConfig)
    assert config.pools and config.output_dir, "pools and output_dir are required"
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    revision = HfApi().model_info(config.model_id).sha
    write_json({"git_hash": get_git_hash(), **asdict(config), "model_revision": revision, "scored": "content tokens 2..n, chat template"},
               out / "perplexity_config.json")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda").eval()
    for spec in config.pools:
        pools_dir, name = spec.split(":")
        rows = load_jsonl(Path(pools_dir) / f"{name}.jsonl")
        encoded = [encode_chat(tokenizer, r["text"], config.max_length) for r in rows]
        nll, n_scored = content_nll(model, encoded, tokenizer.pad_token_id, config.max_batch_tokens)
        n_nan = int(np.isnan(nll).sum())
        if n_nan:
            LOGGER.warning(f"{name}: {n_nan} rows with fewer than 2 content tokens, nll is nan")
        write_jsonl([{"id": r["id"], "pool": name, "nll_mean": None if np.isnan(v) else float(v), "n_scored": int(k)}
                     for r, v, k in zip(rows, nll, n_scored)], out / "perplexity" / f"{name}.jsonl")
        LOGGER.info(f"{name}: {len(rows)} rows, median nll {np.nanmedian(nll):.3f}")
    print(f"perplexity written to {out / 'perplexity'}")


if __name__ == "__main__":
    main()
