# unit tests for exp53's plain-torch lora: a fresh adapter changes nothing, and merging reproduces the adapted output.
# uv run -m pytest tests/test_lora.py -v

import torch

from src.training.lora import LoRALinear, add_lora, merge_lora


def test_fresh_adapter_is_identity_and_merge_matches():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 6)
    x = torch.randn(3, 8)
    before = base(x)
    wrapped = LoRALinear(base, r=4, alpha=8)
    assert torch.allclose(wrapped(x), before)
    with torch.no_grad():
        wrapped.lora_b.normal_()
    adapted = wrapped(x)
    assert not torch.allclose(adapted, before)
    model = torch.nn.Sequential(wrapped)
    merge_lora(model)
    assert isinstance(model[0], torch.nn.Linear)
    assert torch.allclose(model[0](x), adapted, atol=1e-5)


def test_add_lora_wraps_only_named_projections_and_freezes_the_rest():
    model = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4), "mlp": torch.nn.Linear(4, 4)})
    n = add_lora(model, ("q_proj",), r=2, alpha=4)
    assert n == 1 and isinstance(model["q_proj"], LoRALinear) and isinstance(model["mlp"], torch.nn.Linear)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"q_proj.lora_a", "q_proj.lora_b"}
