# samples k replies per raw-text prompt (no chat template, e.g. a base model with an in-context prefix) with vllm in /root/vllm_env, appending to rollouts.jsonl.
# uv run --no-project --python /root/vllm_env/bin/python -m scripts.run_raw_rollouts --pool_path data/processed/base_language/prompts.jsonl --run_dir results/raw/261030_base_language_rollouts_v1 --model_id allenai/Olmo-3-1025-7B --model_revision a81bae42 --seed 0

import hashlib
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import simple_parsing
import torch
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RawRolloutConfig:
    pool_path: str = ""
    run_dir: str = ""
    model_id: str = "allenai/Olmo-3-1025-7B"
    model_revision: str = "a81bae42"
    download_dir: str = "/dev/shm/hf"
    seed: int = 0
    n_rollouts: int = 4
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 2048
    max_model_len: int = 8192
    stop: list[str] = field(default_factory=lambda: ["\n# Query:"])
    chunk: int = 1000
    resume: bool = False


def prompt_seed(prompt_id: str) -> int:
    return int(hashlib.blake2b(prompt_id.encode(), digest_size=4).hexdigest(), 16)


def main():
    config = simple_parsing.parse(RawRolloutConfig)
    assert config.pool_path and config.run_dir, "pool_path and run_dir are required"
    assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0", "set VLLM_USE_FLASHINFER_SAMPLER=0 (no ninja here)"
    run = Path(config.run_dir)
    out_path = run / "rollouts.jsonl"
    if not config.resume:
        assert not run.exists(), f"{run} exists; pass --resume to continue it"
    setup_logging(run)
    set_seed(config.seed)
    if not (run / "config.json").exists():
        write_json(
            {"git_hash": get_git_hash(), **asdict(config), "vllm": vllm.__version__, "torch": torch.__version__},
            run / "config.json",
        )

    pool = load_jsonl(config.pool_path)
    done = {r["prompt_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in pool if r["id"] not in done]
    LOGGER.info(f"{len(pool)} prompts, {len(done)} already done, {len(todo)} to generate")
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision, cache_dir=config.download_dir)
    llm = LLM(
        config.model_id,
        revision=config.model_revision,
        tokenizer_revision=config.model_revision,
        download_dir=config.download_dir,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.85,
        seed=config.seed,
        dtype="bfloat16",
    )
    budget = config.max_model_len - config.max_new_tokens
    for start in range(0, len(todo), config.chunk):
        batch = todo[start : start + config.chunk]
        ids = [tok(r["prompt_text"], add_special_tokens=False)["input_ids"] for r in batch]
        too_long = [r["id"] for r, i in zip(batch, ids) if len(i) > budget]
        assert not too_long, f"prompts longer than {budget} tokens: {too_long[:5]}"
        params = [
            SamplingParams(
                n=config.n_rollouts,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                seed=prompt_seed(r["id"]),
                stop=config.stop,
                stop_token_ids=[tok.eos_token_id],
            )
            for r in batch
        ]
        outputs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], params)
        rows = [
            {
                "id": f"{r['id']}_r{k}",
                "prompt_id": r["id"],
                "rollout": k,
                "reply": c.text.strip(),
                "finish_reason": c.finish_reason,
                "n_tokens": len(c.token_ids),
                "n_prompt_tokens": len(i),
            }
            for r, i, o in zip(batch, ids, outputs)
            for k, c in enumerate(o.outputs)
        ]
        write_jsonl(rows, out_path, append=True)
        LOGGER.info(
            f"chunk {start // config.chunk}: {len(batch)} prompts, cap share {sum(r['finish_reason'] == 'length' for r in rows) / len(rows):.3f}"
        )
    written = load_jsonl(out_path)
    assert (
        len(written) == len(pool) * config.n_rollouts
    ), f"{len(written)} rows, expected {len(pool) * config.n_rollouts}"
    print(f"rollouts saved to {out_path}")


if __name__ == "__main__":
    main()
