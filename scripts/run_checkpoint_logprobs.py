# openai_identity c2: scores every self_nature_b cell reply under the base, SFT and DPO checkpoints (mean reply-token log-probability) and reads the fallback-to-base verdict.
# uv run -m scripts.run_checkpoint_logprobs --self_nature_b_dir results/raw/261021_self_nature_b_self_nature_drift_v1 --run_dir results/raw/261021_openai_identity_fallback_v1 --seed 0

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.interp.features import encode_reply, token_batches
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
CELLS = ("self_covered", "self_uncovered", "other_covered", "other_uncovered")
REAL_SOURCES = (
    "OpenAssistant",
    "Wildchat",
    "WildGuardMix",
    "Aya",
    "Hardcoded Data",
    "lmsys_ext",
    "deploy_ext",
    "lmsys",
    "deploy",
)
SYNTHETIC_SOURCES = ("CoCoNot", "Dolci Instruct Precise IF", "tulu-3-sft-coconot-regenerated-request")


@dataclass
class CheckpointConfig:
    self_nature_b_dir: str = ""
    run_dir: str = ""
    tokenizer_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    tokenizer_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    checkpoints: dict[str, str] = field(
        default_factory=lambda: {
            "base": "allenai/Olmo-3-1025-7B@a81bae42",
            "sft": "allenai/Olmo-3-7B-Instruct-SFT@e1452fc5",
            "dpo": "allenai/Olmo-3-7B-Instruct-DPO@b33130b7de49f0c2553b5c2b3bc8409ff3e627d1",
        }
    )
    seed: int = 0
    n_boot: int = 2000
    effect: float = 0.25
    max_length: int = 2048
    max_batch_tokens: int = 16384


@torch.no_grad()
def mean_logprobs(model, encoded: list[tuple[list[int], list[bool]]], pad: int, max_tokens: int) -> np.ndarray:
    # mean log-probability of the masked (reply) tokens, each predicted from the token before it
    out = np.zeros(len(encoded))
    for batch in tqdm(token_batches([len(ids) for ids, _ in encoded], max_tokens), desc="logprob batches"):
        width = max(len(encoded[i][0]) for i in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, i in enumerate(batch):
            toks, m = encoded[i]
            ids[row, : len(toks)] = torch.tensor(toks)
            attn[row, : len(toks)] = 1
            mask[row, : len(toks)] = torch.tensor(m)
        logits = model(ids.cuda(), attention_mask=attn.cuda()).logits[:, :-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:].cuda()
        out[batch] = ((lp * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
    return out


def main():
    config = simple_parsing.parse(CheckpointConfig)
    assert config.self_nature_b_dir and config.run_dir, "self_nature_b_dir and run_dir are required"
    run, e32 = Path(config.run_dir), Path(config.self_nature_b_dir)
    setup_logging(run)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "checkpoints_config.json")
    cells = {r["id"]: r for r in load_jsonl(e32 / "cells_clean.jsonl")}
    rollouts = load_jsonl(e32 / "drift" / "rollouts.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id, revision=config.tokenizer_revision)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [
        encode_reply(
            tokenizer,
            [{"role": "user", "content": cells[r["prompt_id"]]["text"]}, {"role": "assistant", "content": r["reply"]}],
            config.max_length,
        )
        for r in rollouts
    ]
    assert all(sum(m) > 0 for _, m in encoded), "a reply has no scored tokens"
    scores = {}
    for name, spec in config.checkpoints.items():
        mid, rev = spec.split("@")
        model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
        assert model.config.vocab_size >= len(tokenizer), f"{name}: vocabulary smaller than the tokenizer"
        scores[name] = mean_logprobs(model, encoded, pad, config.max_batch_tokens)
        del model
        torch.cuda.empty_cache()
        LOGGER.info(f"{name}: scored {len(rollouts)} replies, mean logprob {scores[name].mean():.3f}")
    write_jsonl(
        [
            {"id": r["id"], "prompt_id": r["prompt_id"], **{f"logp_{n}": round(float(scores[n][j]), 4) for n in scores}}
            for j, r in enumerate(rollouts)
        ],
        run / "checkpoint_logprobs.jsonl",
    )

    ids = list(cells)
    by_prompt: dict[str, list[int]] = {}
    for j, r in enumerate(rollouts):
        by_prompt.setdefault(r["prompt_id"], []).append(j)
    cell_of = np.array([cells[i]["cell"] for i in ids])
    src = np.array([cells[i]["source"] for i in ids])
    gaps = {
        n: np.array([np.mean([scores["dpo"][j] - scores[n][j] for j in by_prompt[i]]) for i in ids])
        for n in ("base", "sft")
    }
    sft_base = np.array([np.mean([scores["sft"][j] - scores["base"][j] for j in by_prompt[i]]) for i in ids])

    def cell_mean(arr: np.ndarray, cell: str, idx: np.ndarray | None = None) -> float:
        v = arr[cell_of == cell]
        return float((v if idx is None else v[idx]).mean())

    def diff_ci(
        arr: np.ndarray, a: str = "self_uncovered", b: str = "self_covered"
    ) -> tuple[float, list[float], float]:
        sd = arr[cell_of == "other_covered"].std()
        d = (cell_mean(arr, a) - cell_mean(arr, b)) / sd
        na, nb = (cell_of == a).sum(), (cell_of == b).sum()
        boot = [
            (cell_mean(arr, a, rng.choice(na, na)) - cell_mean(arr, b, rng.choice(nb, nb))) / sd
            for _ in range(config.n_boot)
        ]
        return d, [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], float(sd)

    G = gaps["base"]
    d, ci, sd = diff_ci(G)
    reading = "fallback_to_base" if d <= -config.effect and ci[1] < 0 else "no_fallback"
    d_sft, ci_sft, _ = diff_ci(gaps["sft"])
    d_sb, ci_sb, _ = diff_ci(sft_base)
    drift = (
        {p["id"]: p["resp_proj_L"] for p in load_jsonl(e32 / "drift_per_prompt.jsonl")}
        if (e32 / "drift_per_prompt.jsonl").exists()
        else {}
    )
    rho = float(spearmanr([drift[i] for i in ids], G).statistic) if drift else None
    real = (cell_of == "self_covered") & np.isin(src, REAL_SOURCES)
    synth = (cell_of == "self_covered") & np.isin(src, SYNTHETIC_SOURCES)
    decision = {
        "reading": reading,
        "G_uncovered_minus_covered_sd": d,
        "G_ci95": ci,
        "other_covered_sd_of_G": sd,
        "G_cell_means_nats_per_token": {c: cell_mean(G, c) for c in CELLS},
        "logp_cell_means": {
            n: {
                c: float(np.mean([scores[n][j] for i in ids if cells[i]["cell"] == c for j in by_prompt[i]]))
                for c in CELLS
            }
            for n in scores
        },
        "dpo_minus_sft": {
            "uncovered_minus_covered_sd": d_sft,
            "ci95": ci_sft,
            "cell_means": {c: cell_mean(gaps["sft"], c) for c in CELLS},
        },
        "sft_minus_base": {
            "uncovered_minus_covered_sd": d_sb,
            "ci95": ci_sb,
            "cell_means": {c: cell_mean(sft_base, c) for c in CELLS},
        },
        "spearman_G_vs_drift_projection": rho,
        "covered_split_G": {
            "real_n": int(real.sum()),
            "synthetic_n": int(synth.sum()),
            "real_mean": float(G[real].mean()) if real.any() else None,
            "synthetic_mean": float(G[synth].mean()) if synth.any() else None,
            "uncovered_mean": cell_mean(G, "self_uncovered"),
        },
        "n_prompts": len(ids),
        "n_replies": len(rollouts),
    }
    write_json(decision, run / "decision_c2.json")
    print(f"reading: {reading}; G (uncovered - covered) {d:+.3f} SD [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(
        "G = logp_dpo - logp_base per cell (nats/token): "
        + ", ".join(f"{c} {v:.3f}" for c, v in decision["G_cell_means_nats_per_token"].items())
    )
    print(
        f"dpo - sft: {d_sft:+.3f} [{ci_sft[0]:+.3f}, {ci_sft[1]:+.3f}]; sft - base: {d_sb:+.3f} [{ci_sb[0]:+.3f}, {ci_sb[1]:+.3f}]"
    )
    print(f"covered split {decision['covered_split_G']}; spearman G vs drift {rho}")
    print(f"decision written to {run / 'decision_c2.json'}")


if __name__ == "__main__":
    main()
