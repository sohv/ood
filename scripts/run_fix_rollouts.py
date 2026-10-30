# fix_overrefused: run_rollouts' sampling for the base model and any number of lora adapters in one vllm engine, one rollouts file per model.
# uv run --no-project --python /root/vllm_env/bin/python -m scripts.run_fix_rollouts --pool_path data/processed/fix_overrefused/test.jsonl --out_root results/raw --name_prefix 261023_fix_overrefused_gen --adapters gap=results/raw/261023_fix_overrefused_ft_gap_v2/peft control=results/raw/261023_fix_overrefused_ft_control_v3/peft --seed 0

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import simple_parsing
import torch
import vllm
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

from scripts.run_rollouts import prompt_seed
from src.data.io import load_jsonl
from src.interp.features import render_user
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class LoraRolloutConfig:
    pool_path: str = ""
    out_root: str = "results/raw"
    name_prefix: str = ""
    adapters: list[str] = field(default_factory=list)
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    n_rollouts: int = 2
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 2048
    max_model_len: int = 8192
    max_lora_rank: int = 16


def main():
    config = simple_parsing.parse(LoraRolloutConfig)
    assert config.pool_path and config.name_prefix, "pool_path and name_prefix required"
    assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0", "set VLLM_USE_FLASHINFER_SAMPLER=0"
    models = {"base": None, **dict(a.split("=", 1) for a in config.adapters)}
    runs = {m: Path(config.out_root) / f"{config.name_prefix}_{m}_v1" for m in models}
    for m, run in runs.items():
        assert not run.exists(), f"{run} exists"
        setup_logging(run)
        set_seed(config.seed)
        write_json(
            {
                "git_hash": get_git_hash(),
                **asdict(config),
                "model": m,
                "adapter": models[m],
                "vllm": vllm.__version__,
                "torch": torch.__version__,
            },
            run / "config.json",
        )
    pool = load_jsonl(config.pool_path)
    assert len({r["id"] for r in pool}) == len(pool), "duplicate prompt ids in pool"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # stop ids exactly as run_rollouts picks them
    gen_eos = GenerationConfig.from_pretrained(config.model_id, revision=config.model_revision).eos_token_id
    eos = set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    if tokenizer.eos_token_id is not None:
        eos.add(tokenizer.eos_token_id)
    for t in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and tokenizer.convert_ids_to_tokens(i) == t:
            eos.add(i)
    eos = sorted(i for i in eos if isinstance(i, int))
    LOGGER.info(f"stop token ids {eos}")
    llm = LLM(
        config.model_id,
        revision=config.model_revision,
        tokenizer_revision=config.model_revision,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.85,
        seed=config.seed,
        dtype="bfloat16",
        enable_lora=len(models) > 1,
        max_lora_rank=config.max_lora_rank,
    )
    budget = config.max_model_len - config.max_new_tokens
    ids = [tokenizer(render_user(tokenizer, r["text"]), add_special_tokens=False)["input_ids"] for r in pool]
    too_long = [r["id"] for r, i in zip(pool, ids, strict=True) if len(i) > budget]
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
        for r in pool
    ]
    for k, (m, path) in enumerate(models.items()):
        lora = None if path is None else LoRARequest(m, k, str(Path(path).resolve()))
        t0 = time.time()
        outputs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], params, lora_request=lora)
        rows = []
        for r, i, o in zip(pool, ids, outputs, strict=True):
            assert len(o.outputs) == config.n_rollouts, f"{r['id']} got {len(o.outputs)} samples"
            for j, c in enumerate(o.outputs):
                rows.append(
                    {
                        "id": f"{r['id']}_r{j}",
                        "prompt_id": r["id"],
                        "rollout": j,
                        "reply": c.text,
                        "finish_reason": c.finish_reason,
                        "n_tokens": len(c.token_ids),
                        "n_prompt_tokens": len(i),
                    }
                )
        write_jsonl(rows, runs[m] / "rollouts.jsonl")
        LOGGER.info(f"{m}: {len(rows)} replies in {time.time() - t0:.0f}s")
        print(f"rollouts saved to {runs[m] / 'rollouts.jsonl'}")


if __name__ == "__main__":
    main()
