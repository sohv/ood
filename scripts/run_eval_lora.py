# exp27: generates openers (and a few full replies for side effects) from the untrained model and each LoRA adapter with vllm (run in /root/vllm_env).
# uv run --no-project --python /root/vllm_env/bin/python -m scripts.run_eval_lora --test_path data/processed/exp27/test.jsonl --adapters_dir results/raw/261020_exp27_stage1_v1/adapters --out_dir results/raw/261020_exp27_stage1_v1/eval --seed 0

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
class EvalLoraConfig:
    test_path: str = ""
    adapters_dir: str = ""
    out_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    # adapter names under adapters_dir; empty means every adapter there; "base" (no adapter) is always included
    adapters: list[str] = field(default_factory=list)
    opener_tokens: int = 32
    side_per_family: int = 5
    side_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95


def prompt_seed(key: str) -> int:
    return int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16)


def main():
    config = simple_parsing.parse(EvalLoraConfig)
    assert config.test_path and config.adapters_dir and config.out_dir, "test_path, adapters_dir, out_dir required"
    assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0", "set VLLM_USE_FLASHINFER_SAMPLER=0"
    out = Path(config.out_dir)
    assert not (out / "openers.jsonl").exists(), f"{out} already has results"
    setup_logging(out)
    set_seed(config.seed)
    found = sorted(p.name for p in Path(config.adapters_dir).iterdir() if (p / "adapter_config.json").exists())
    names = [n for n in found if not config.adapters or n in config.adapters]
    assert not config.adapters or set(names) == set(
        config.adapters
    ), f"missing adapters: {set(config.adapters) - set(names)}"
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "adapters_evaluated": ["base", *names],
            "vllm": vllm.__version__,
        },
        out / "eval_config.json",
    )

    test = load_jsonl(config.test_path)
    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    ids = [tok(render_user(tok, r["text"]), add_special_tokens=False)["input_ids"] for r in test]
    llm = LLM(
        config.model_id,
        revision=config.model_revision,
        tokenizer_revision=config.model_revision,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=config.seed,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
    )
    side_idx = [
        i
        for fam in dict.fromkeys(r["family"] for r in test)
        for i in [j for j, r in enumerate(test) if r["family"] == fam][: config.side_per_family]
    ]
    openers, side = [], []
    for k, name in enumerate(["base", *names]):
        lora = None if name == "base" else LoRARequest(name, k, str(Path(config.adapters_dir) / name))
        for rows, idx, max_tokens in (
            (openers, range(len(test)), config.opener_tokens),
            (side, side_idx, config.side_tokens),
        ):
            if not idx:
                continue
            params = [
                SamplingParams(
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=max_tokens,
                    seed=prompt_seed(test[i]["id"]),
                    stop_token_ids=EOS,
                )
                for i in idx
            ]
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids[i]) for i in idx], params, lora_request=lora)
            for i, o in zip(idx, outs):
                c = o.outputs[0]
                rows.append(
                    {
                        "id": f"{name}__{test[i]['id']}",
                        "user": test[i]["text"],
                        "adapter": name,
                        "test_id": test[i]["id"],
                        "family": test[i]["family"],
                        "reply": c.text,
                        "finish_reason": c.finish_reason,
                    }
                )
        LOGGER.info(f"{name}: {len(test)} openers, {len(side_idx)} full replies")
    write_jsonl(openers, out / "openers.jsonl")
    write_jsonl(side, out / "side_replies.jsonl")
    assert len(openers) == len(test) * (1 + len(names)), "opener rows missing"
    print(f"openers saved to {out / 'openers.jsonl'} ({len(openers)} rows)")


if __name__ == "__main__":
    main()
