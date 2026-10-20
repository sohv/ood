# exp27: trains one LoRA adapter per training file on Olmo-3 7B DPO (run in /root/vllm_env), loss on the reply tokens only.
# uv run --no-project --python /root/vllm_env/bin/python -m scripts.run_train_lora --train_dir data/processed/exp27 --run_dir results/raw/261020_exp27_stage1_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import simple_parsing
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainLoraConfig:
    train_dir: str = ""
    run_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    # adapter names = training files train_<name>.jsonl; empty means every file in train_dir
    adapters: list[str] = field(default_factory=list)
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 1e-4
    epochs: int = 2
    micro_batch: int = 2
    grad_accum: int = 4
    max_len: int = 1536
    warmup_steps: int = 10
    num_tasks: int | None = None


def encode(tokenizer, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    # labels are -100 on the prompt and template tokens before the reply
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    assert full.startswith(prompt), "chat template does not extend the prompt"
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    assert ids[: len(p_ids)] == p_ids, "prompt tokens change when the reply is appended"
    labels = [-100] * len(p_ids) + ids[len(p_ids) :]
    return ids[:max_len], labels[:max_len]


def main():
    config = simple_parsing.parse(TrainLoraConfig)
    assert config.train_dir and config.run_dir, "train_dir and run_dir are required"
    run = Path(config.run_dir)
    setup_logging(run)
    files = sorted(Path(config.train_dir).glob("train_*.jsonl"))
    names = [f.stem.removeprefix("train_") for f in files]
    if config.adapters:
        names = [n for n in names if n in config.adapters]
        assert set(names) == set(config.adapters), f"missing training files: {set(config.adapters) - set(names)}"
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "adapters_to_train": names, "torch": torch.__version__},
        run / "train_config.json",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16, device_map="cuda"
    )
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    for name in names:
        out = run / "adapters" / name
        if (out / "adapter_config.json").exists():
            LOGGER.info(f"{name}: already trained, skipping")
            continue
        set_seed(config.seed)
        rows = load_jsonl(Path(config.train_dir) / f"train_{name}.jsonl", config.num_tasks)
        data = [encode(tokenizer, r["messages"], config.max_len) for r in rows]
        model = get_peft_model(
            base,
            LoraConfig(
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=TARGETS,
                task_type="CAUSAL_LM",
            ),
        )
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.0)
        steps_per_epoch = math.ceil(len(data) / (config.micro_batch * config.grad_accum))
        total = steps_per_epoch * config.epochs

        def lr_at(step: int, total: int = total) -> float:
            if step < config.warmup_steps:
                return (step + 1) / config.warmup_steps
            return 0.5 * (1 + math.cos(math.pi * (step - config.warmup_steps) / max(1, total - config.warmup_steps)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        g = torch.Generator().manual_seed(config.seed)
        log, step, t0 = [], 0, time.time()
        model.train()
        for epoch in range(config.epochs):
            order = torch.randperm(len(data), generator=g).tolist()
            batches = [order[i : i + config.micro_batch] for i in range(0, len(order), config.micro_batch)]
            for k in range(0, len(batches), config.grad_accum):
                group = batches[k : k + config.grad_accum]
                n_tok = sum(sum(1 for x in data[i][1] if x != -100) for b in group for i in b)
                total_loss = 0.0
                for b in group:
                    width = max(len(data[i][0]) for i in b)
                    ids = torch.full((len(b), width), pad, dtype=torch.long)
                    lab = torch.full((len(b), width), -100, dtype=torch.long)
                    att = torch.zeros((len(b), width), dtype=torch.long)
                    for row, i in enumerate(b):
                        x, y = data[i]
                        ids[row, : len(x)], lab[row, : len(y)], att[row, : len(x)] = torch.tensor(x), torch.tensor(y), 1
                    logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, :-1].float()
                    target = lab[:, 1:].cuda()
                    # sum over reply tokens, divided by the reply tokens in the whole accumulation group
                    loss = (
                        torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100, reduction="sum"
                        )
                        / n_tok
                    )
                    loss.backward()
                    total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                log.append(
                    {
                        "id": f"{name}_{step}",
                        "adapter": name,
                        "epoch": epoch,
                        "step": step,
                        "loss": total_loss,
                        "lr": sched.get_last_lr()[0],
                    }
                )
                step += 1
        LOGGER.info(
            f"{name}: {len(data)} examples, {step} steps, final loss {log[-1]['loss']:.4f}, {time.time() - t0:.0f}s"
        )
        model.save_pretrained(out)
        write_jsonl(log, run / "adapters" / f"{name}_loss.jsonl")
        base = model.unload()
        del model, opt, sched
        torch.cuda.empty_cache()
    print(f"adapters saved to {run / 'adapters'}")


if __name__ == "__main__":
    main()
