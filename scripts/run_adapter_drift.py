# exp34: projects each adapter's cell replies onto exp32's Assistant Axis using the adapted model's own activations (run in /root/vllm_env, which has peft).
# uv run --no-project --python /root/vllm_env/bin/python -m scripts.run_adapter_drift --exp32_dir results/raw/261021_exp32_self_nature_drift_v1 --run_dir results/raw/261021_exp34_coverage_adapter_v1 --seed 0

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.interp.features import encode_reply, extract
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class AdapterDriftConfig:
    exp32_dir: str = ""
    run_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    adapters: list[str] = field(default_factory=lambda: ["identity", "control"])
    seed: int = 0
    layer: int = 16
    max_batch_tokens: int = 8192


def main():
    config = simple_parsing.parse(AdapterDriftConfig)
    assert config.exp32_dir and config.run_dir, "exp32_dir and run_dir required"
    run, e32 = Path(config.run_dir), Path(config.exp32_dir)
    setup_logging(run)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "adapter_drift_config.json")
    cells = {r["id"]: r for r in load_jsonl(e32 / "cells_clean.jsonl")}
    unit = np.load(e32 / "axis.npz")["unit"][config.layer]
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for name in config.adapters:
        rollouts = load_jsonl(run / name / "rollouts.jsonl")
        model = PeftModel.from_pretrained(base, str(run / "adapters" / name)).eval()
        encoded = [
            encode_reply(
                tokenizer,
                [
                    {"role": "user", "content": cells[r["prompt_id"]]["text"]},
                    {"role": "assistant", "content": r["reply"]},
                ],
                2048,
            )
            for r in rollouts
        ]
        feats = extract(model, encoded, pad, max_tokens=config.max_batch_tokens, layers=[config.layer])["cmean"][
            :, 0, :
        ].astype(np.float32)
        proj = feats @ unit
        write_jsonl(
            [
                {"id": r["id"], "prompt_id": r["prompt_id"], "adapter": name, "proj": round(float(p), 5)}
                for r, p in zip(rollouts, proj)
            ],
            run / name / "projections.jsonl",
        )
        LOGGER.info(f"{name}: {len(rollouts)} replies projected, mean {proj.mean():.3f}")
        print(f"{name}: projections written to {run / name / 'projections.jsonl'}")
        base = model.unload()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
