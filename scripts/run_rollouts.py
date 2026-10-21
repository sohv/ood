# samples k replies per prompt with vllm in /root/vllm_env (export VLLM_USE_FLASHINFER_SAMPLER=0), appending to rollouts.jsonl so a crash keeps what finished.
# uv run --no-project --python /root/vllm_env/bin/python -m scripts.run_rollouts --pool_path data/processed/exp18/sharechat.jsonl --run_dir results/raw/261019_exp18_sharechat_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing
import torch
import vllm
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from src.data.io import load_jsonl
from src.interp.features import render_user
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    pool_path: str = ""
    run_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    num_tasks: int | None = None
    n_rollouts: int = 6
    # the model's own generation_config sampling, passed explicitly so it is recorded
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 2048
    max_model_len: int = 8192
    chunk: int = 1000
    resume: bool = False


def prompt_seed(prompt_id: str) -> int:
    return int(hashlib.blake2b(prompt_id.encode(), digest_size=4).hexdigest(), 16)


def main():
    config = simple_parsing.parse(RolloutConfig)
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
            {
                "git_hash": get_git_hash(),
                **asdict(config),
                "vllm": vllm.__version__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "flashinfer_sampler": False,
            },
            run / "config.json",
        )

    pool = load_jsonl(config.pool_path, config.num_tasks)
    assert len({r["id"] for r in pool}) == len(pool), "duplicate prompt ids in pool"
    done = {}
    if out_path.exists():
        for r in load_jsonl(out_path):
            done.setdefault(r["prompt_id"], set()).add(r["rollout"])
        assert all(v == set(range(config.n_rollouts)) for v in done.values()), "a resumed prompt is incomplete"
    todo = [r for r in pool if r["id"] not in done]
    LOGGER.info(f"{len(pool)} prompts, {len(done)} already done, {len(todo)} to generate")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # stop ids from the model's own generation_config.json (olmo 3: <|im_end|> 100265 and <|endoftext|> 100257; llama 3: 128001, 128008, 128009)
    gen_eos = GenerationConfig.from_pretrained(config.model_id, revision=config.model_revision).eos_token_id
    eos = set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    if tokenizer.eos_token_id is not None:
        eos.add(tokenizer.eos_token_id)
    # end-of-turn tokens that exist in this vocabulary (tulu 3's config lists only <|end_of_text|>, not <|eot_id|>)
    for t in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and tokenizer.convert_ids_to_tokens(i) == t:
            eos.add(i)
    eos = sorted(i for i in eos if isinstance(i, int))
    assert eos, f"no usable eos ids: {gen_eos}"
    LOGGER.info(f"stop token ids {eos}")
    llm = LLM(
        config.model_id,
        revision=config.model_revision,
        tokenizer_revision=config.model_revision,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.85,
        seed=config.seed,
        dtype="bfloat16",
    )
    budget = config.max_model_len - config.max_new_tokens

    def render(row: dict) -> str:
        # a multi-turn prompt (e.g. in-context demos) when the row has messages, else one user turn
        if "messages" in row:
            return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        return render_user(tokenizer, row["text"])

    for start in range(0, len(todo), config.chunk):
        batch = todo[start : start + config.chunk]
        # same rendering as the feature extractor: chat template, then tokenise without extra special tokens
        ids = [tokenizer(render(r), add_special_tokens=False)["input_ids"] for r in batch]
        too_long = [r["id"] for r, i in zip(batch, ids) if len(i) > budget]
        assert not too_long, f"prompts longer than {budget} tokens: {too_long[:5]}"
        params = [
            SamplingParams(
                n=config.n_rollouts,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                seed=prompt_seed(r["id"]),
                stop_token_ids=eos,
            )
            for r in batch
        ]
        t0 = time.time()
        outputs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], params)
        rows = []
        for r, i, o in zip(batch, ids, outputs):
            assert len(o.outputs) == config.n_rollouts, f"{r['id']} got {len(o.outputs)} samples"
            for k, c in enumerate(o.outputs):
                rows.append(
                    {
                        "id": f"{r['id']}_r{k}",
                        "prompt_id": r["id"],
                        "rollout": k,
                        "reply": c.text,
                        "finish_reason": c.finish_reason,
                        "n_tokens": len(c.token_ids),
                        "n_prompt_tokens": len(i),
                    }
                )
        write_jsonl(rows, out_path, append=True)
        n_tok = sum(r["n_tokens"] for r in rows)
        LOGGER.info(
            f"chunk {start // config.chunk}: {len(batch)} prompts, {n_tok} tokens in {time.time() - t0:.0f}s, "
            f"cap share {sum(r['finish_reason'] == 'length' for r in rows) / len(rows):.3f}"
        )

    written = load_jsonl(out_path)
    assert (
        len(written) == len(pool) * config.n_rollouts
    ), f"{len(written)} rows, expected {len(pool) * config.n_rollouts}"
    assert len({r["id"] for r in written}) == len(written), "duplicate rollout ids"
    print(f"rollouts saved to {out_path}")


if __name__ == "__main__":
    main()
