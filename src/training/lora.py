# fix_overrefused plain-torch lora: wrap named linear layers with a low-rank adapter, train only the adapter, merge it back for saving.

import math

import torch


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        dev = base.weight.device
        # float32 adapter on top of a frozen (often bf16) base; b starts at zero so a fresh adapter changes nothing
        self.lora_a = torch.nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x.to(torch.float32) @ self.lora_a.T @ self.lora_b.T) * self.scale
        return self.base(x) + delta.to(x.dtype)


def add_lora(model: torch.nn.Module, targets: tuple[str, ...], r: int, alpha: float) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    hits = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in targets
    ]
    for name, mod in hits:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.split(".")[-1], LoRALinear(mod, r, alpha))
    return len(hits)


@torch.no_grad()
def merge_lora(model: torch.nn.Module) -> None:
    wrapped = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    for name, mod in wrapped:
        base = mod.base
        # add in float32 and round once
        base.weight.copy_((base.weight.float() + (mod.lora_b @ mod.lora_a) * mod.scale).to(base.weight.dtype))
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.split(".")[-1], base)
