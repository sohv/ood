# exp34: samples k full replies per prompt from the model with each LoRA adapter in vllm (run in /root/vllm_env, VLLM_USE_FLASHINFER_SAMPLER=0).
# uv run --no-project --python /root/vllm_env/bin/python -m scripts.run_rollouts_lora --pool_path results/raw/261021_exp32_self_nature_drift_v1/cells_clean.jsonl --adapters_dir results/raw/261021_exp34_coverage_adapter_v1/adapters --out_dir results/raw/261021_exp34_coverage_adapter_v1 --seed 0

import hashlib
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import simple_parsing
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

from src.data.io import load_jsonl
from src.interp.features import render_user
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
EOS = [100265, 100257]


@dataclass
class RolloutsLoraConfig:
    pool_path: str = ""
    adapters_dir: str = ""
    out_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    adapters: list[str] = field(default_factory=list)
    n_rollouts: int = 6
    max_new_tokens: int = 1024
    max_model_len: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    chunk: int = 1000


def prompt_seed(key: str) -> int:
    return int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16)


def main():
    config = simple_parsing.parse(RolloutsLoraConfig)
    assert config.pool_path and config.adapters_dir and config.out_dir, "pool_path, adapters_dir, out_dir required"
    assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0", "set VLLM_USE_FLASHINFER_SAMPLER=0"
    out = Path(config.out_dir)
    setup_logging(out)
    set_seed(config.seed)
    found = sorted(p.name for p in Path(config.adapters_dir).iterdir() if (p / "adapter_config.json").exists())
    names = [n for n in found if not config.adapters or n in config.adapters]
    assert names, "no adapters found"
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "adapters": names, "vllm": vllm.__version__},
        out / "rollouts_lora_config.json",
    )
    pool = load_jsonl(config.pool_path)
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    ids = [tok(render_user(tok, r["text"]), add_special_tokens=False)["input_ids"] for r in pool]
    budget = config.max_model_len - config.max_new_tokens
    too_long = [r["id"] for r, i in zip(pool, ids) if len(i) > budget]
    assert not too_long, f"prompts longer than {budget} tokens: {too_long[:5]}"
    llm = LLM(
        config.model_id,
        revision=config.model_revision,
        tokenizer_revision=config.model_revision,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.85,
        seed=config.seed,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
    )
    for k, name in enumerate(names, start=1):
        adir = out / name
        adir.mkdir(parents=True, exist_ok=True)
        out_path = adir / "rollouts.jsonl"
        assert not out_path.exists(), f"{out_path} exists"
        lora = LoRARequest(name, k, str(Path(config.adapters_dir) / name))
        for start in range(0, len(pool), config.chunk):
            batch = list(range(start, min(start + config.chunk, len(pool))))
            params = [
                SamplingParams(
                    n=config.n_rollouts,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_new_tokens,
                    seed=prompt_seed(pool[i]["id"]),
                    stop_token_ids=EOS,
                )
                for i in batch
            ]
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids[i]) for i in batch], params, lora_request=lora)
            rows = []
            for i, o in zip(batch, outs):
                assert len(o.outputs) == config.n_rollouts, f"{pool[i]['id']} got {len(o.outputs)} samples"
                for j, c in enumerate(o.outputs):
                    rows.append(
                        {
                            "id": f"{pool[i]['id']}_r{j}",
                            "prompt_id": pool[i]["id"],
                            "rollout": j,
                            "adapter": name,
                            "reply": c.text,
                            "finish_reason": c.finish_reason,
                            "n_tokens": len(c.token_ids),
                        }
                    )
            write_jsonl(rows, out_path, append=True)
            LOGGER.info(
                f"{name}: chunk {start // config.chunk} done, cap share {sum(r['finish_reason'] == 'length' for r in rows) / len(rows):.3f}"
            )
        n = len(load_jsonl(out_path))
        assert n == len(pool) * config.n_rollouts, f"{name}: {n} rows, expected {len(pool) * config.n_rollouts}"
        print(f"{name}: rollouts saved to {out_path} ({n} rows)")


if __name__ == "__main__":
    main()
