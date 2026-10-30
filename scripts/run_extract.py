# extracts pooled residual features for every repro_two_sided pool from one model and saves them as npz.
# uv run -m scripts.run_extract --pools_dir data/processed/repro_two_sided --output_dir results/raw/261018_repro_two_sided_repro_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.interp.features import encode_chat, encode_raw, encode_row, extract, layers_for
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

CHAT_POOLS = ("idfit", "idmix", "legal", "medical", "wildchat_en_test")


@dataclass
class ExtractConfig:
    pools_dir: str = ""
    output_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    seed: int = 0
    num_tasks: int | None = None
    max_length: int = 1024
    dolma_full_tokens: int = 512
    max_batch_tokens: int = 32768
    chat_pools: list[str] = field(default_factory=lambda: list(CHAT_POOLS))
    include_dolma: bool = True
    include_dolma_wrapped: bool = False
    extra_pools_dir: str = ""
    extra_pools: list[str] = field(default_factory=list)


def main():
    config = simple_parsing.parse(ExtractConfig)
    assert config.pools_dir and config.output_dir, "pools_dir and output_dir are required"
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    revision = HfApi().model_info(config.model_id).sha
    layers = layers_for(AutoConfig.from_pretrained(config.model_id, revision=revision).num_hidden_layers)
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "model_revision": revision,
            "layers": layers,
            "hook_point": "hidden_states[l] residual stream, l=32 is post final norm",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        out / "config.json",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    chat_pools = config.chat_pools + config.extra_pools
    pools = {name: load_jsonl(Path(config.pools_dir) / f"{name}.jsonl", config.num_tasks) for name in config.chat_pools}
    pools |= {
        name: load_jsonl(Path(config.extra_pools_dir) / f"{name}.jsonl", config.num_tasks)
        for name in config.extra_pools
    }
    encoded = {name: [encode_row(tokenizer, r, config.max_length) for r in pools[name]] for name in chat_pools}
    ids = {name: [r["id"] for r in pools[name]] for name in chat_pools}
    if config.include_dolma:
        dolma = load_jsonl(Path(config.pools_dir) / "dolma.jsonl", config.num_tasks)
        idfit_lengths = np.array([sum(m) for _, m in encoded["idfit"]])
        matched_lengths = np.random.default_rng(config.seed).choice(idfit_lengths, size=len(dolma), replace=True)
        encoded["dolma_full"] = [encode_raw(tokenizer, r["text"], config.dolma_full_tokens) for r in dolma]
        encoded["dolma_matched"] = [encode_raw(tokenizer, r["text"], int(n)) for r, n in zip(dolma, matched_lengths)]
        ids["dolma_full"] = ids["dolma_matched"] = [r["id"] for r in dolma]
        if config.include_dolma_wrapped:
            # the matched cut as text, placed in the chat template as a user message
            cuts = [tokenizer.decode(toks).strip() for toks, _ in encoded["dolma_matched"]]
            encoded["dolma_wrapped"] = [encode_chat(tokenizer, t, config.max_length) for t in cuts]
            ids["dolma_wrapped"] = ids["dolma_full"]

    first = chat_pools[-1]
    LOGGER.info(
        f"{first}[0] content tokens decoded: {tokenizer.decode([t for t, m in zip(*encoded[first][0]) if m])[:300]!r}"
    )

    (out / "features").mkdir(exist_ok=True)
    timings = {}
    for name, enc in encoded.items():
        start = time.time()
        feats = extract(model, enc, tokenizer.pad_token_id, config.max_batch_tokens, layers)
        timings[name] = {"rows": len(enc), "seconds": time.time() - start}
        np.savez(out / "features" / f"{name}.npz", ids=np.array(ids[name]), layers=np.array(layers), **feats)
        LOGGER.info(
            f"{name}: {len(enc)} rows in {timings[name]['seconds']:.1f}s, median content tokens {np.median(feats['n_content_tokens']):.0f}"
        )
    write_json(timings, out / "extract_timings.json")
    print(f"features saved to {out / 'features'}")
    print(f"next: uv run -m scripts.run_detectors --run_dir {out} --seed 0")


if __name__ == "__main__":
    main()
