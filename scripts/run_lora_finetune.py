# exp53: lora fine-tunes OLMo-3-7B-Instruct-DPO on teacher answers to a prompt set, then merges and saves the model for vllm.
# uv run -m scripts.run_lora_finetune --run_dir results/raw/261023_exp53_ft_gap_v1 --train_path data/processed/exp53/train_gap.jsonl --save_dir /workspace/ood_models/exp53_gap --seed 0

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.training.lora import add_lora, merge_lora
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class LoraConfig:
    run_dir: str = ""
    train_path: str = ""
    save_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    r: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 1e-4
    epochs: int = 2
    micro_batch: int = 4
    accumulate: int = 4
    max_len: int = 2048
    warmup_frac: float = 0.03
    # false: keep only the adapter (peft format for vllm), no 14.6 GB merged copy
    merge: bool = True


def encode(tok, user: str, answer: str, max_len: int) -> tuple[list[int], list[int]]:
    # loss only on the assistant reply: prompt tokens get label -100
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    full = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": answer}], tokenize=True, return_dict=False
    )
    assert full[: len(prompt)] == prompt, "chat template prompt is not a prefix of the full conversation"
    labels = [-100] * len(prompt) + full[len(prompt) :]
    return full[:max_len], labels[:max_len]


def export_peft(adapter: dict, config: "LoraConfig", out: Path) -> None:
    # peft layout that vllm's lora loader reads: lora_A is (r, in), lora_B is (out, r), scale alpha / r
    out.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"base_model.model.{n.removesuffix('.lora_a').removesuffix('.lora_b')}.{'lora_A' if n.endswith('lora_a') else 'lora_B'}.weight": t.to(
            torch.bfloat16
        ).contiguous()
        for n, t in adapter.items()
    }
    save_file(tensors, out / "adapter_model.safetensors")
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.r,
        "lora_alpha": config.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(config.targets),
        "base_model_name_or_path": config.model_id,
    }
    (out / "adapter_config.json").write_text(json.dumps(peft_cfg, indent=2) + "\n")


def main():
    config = simple_parsing.parse(LoraConfig)
    assert config.run_dir and config.train_path and config.save_dir, "run_dir, train_path, save_dir required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    assert not config.merge or not Path(config.save_dir).exists(), f"{config.save_dir} exists"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    rows = [x for x in load_jsonl(Path(config.train_path)) if x["answer"]]
    data = [encode(tok, x["text"], x["answer"], config.max_len) for x in rows]
    data = [d for d in data if any(t != -100 for t in d[1])]
    LOGGER.info(f"{len(data)} training examples from {config.train_path}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, torch_dtype=torch.bfloat16
    ).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    n_wrapped = add_lora(model, config.targets, config.r, config.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info(f"wrapped {n_wrapped} layers, {sum(p.numel() for p in params)} trainable parameters")
    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.0)
    # optimizer steps actually taken: leftover micro-batches at an epoch end are dropped
    steps = (math.ceil(len(data) / config.micro_batch) // config.accumulate) * config.epochs
    warm = max(1, int(config.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
    )
    rng = np.random.default_rng(config.seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    log, step = [], 0
    for epoch in range(config.epochs):
        order = rng.permutation(len(data))
        opt.zero_grad()
        for b in range(0, len(order), config.micro_batch):
            batch = [data[i] for i in order[b : b + config.micro_batch]]
            width = max(len(ids) for ids, _ in batch)
            ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in batch]).cuda()
            lab = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch]).cuda()
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch]).cuda()
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss / config.accumulate
            loss.backward()
            if (b // config.micro_batch + 1) % config.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                log.append({"step": step, "epoch": epoch, "loss": round(loss.item() * config.accumulate, 4)})
                if step % 10 == 0:
                    LOGGER.info(f"step {step}/{steps} loss {log[-1]['loss']}")
    (run / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
    # adapter weights first, so a failure while merging or saving cannot lose the training
    adapter = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(adapter, run / "lora_adapter.pt")
    export_peft(adapter, config, run / "peft")
    if not config.merge:
        write_json(
            {"examples": len(data), "steps": step, "final_loss": log[-1]["loss"], "adapter": str(run / "peft")},
            run / "summary.json",
        )
        print(f"Saved adapter to {run / 'peft'}")
        return
    merge_lora(model)
    # the base generation_config carries temperature 0.6 and top_p 0.95; saving requires sampling to be marked on
    model.generation_config.do_sample = True
    model.eval()
    model.save_pretrained(config.save_dir, safe_serialization=True)
    tok.save_pretrained(config.save_dir)
    write_json(
        {"examples": len(data), "steps": step, "final_loss": log[-1]["loss"], "save_dir": config.save_dir},
        run / "summary.json",
    )
    print(f"Saved merged model to {config.save_dir}")


if __name__ == "__main__":
    main()
