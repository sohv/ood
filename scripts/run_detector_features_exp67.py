# exp67, GPU stage: layer-16 activations, two streamed TopK SAEs (pretraining vs post-training tokens) with per-text FVU, and raw-text log-probabilities under the DPO and the base model for every evaluation and training text.
# uv run -m scripts.run_detector_features_exp67 --run_dir results/raw/261027_exp67_detector_features_v1 --dolci_shard <path to Dolci-Instruct-SFT train-00000-of-00015.parquet> --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import gc
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import simple_parsing
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    run_dir: str = ""
    dolci_shard: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    base_model_id: str = "allenai/Olmo-3-1025-7B"
    layer: int = 16
    exp52_prompts: str = "data/processed/exp52/prompts.jsonl"
    exp52_rollouts: str = "results/raw/261022_exp52_rollouts_v1/rollouts.jsonl"
    exp61_items: str = "data/processed/human_labels_exp61/topic_items.json"
    dolma_path: str = "data/processed/exp01/dolma.jsonl"
    needles_dir: str = "data/processed/exp06"
    n_planted: int = 100
    n_traffic: int = 2000
    n_post: int = 5000
    max_tokens: int = 512
    batch_tokens: int = 24000
    sae_width: int = 16384
    sae_k: int = 64
    sae_epochs: int = 2
    sae_batch: int = 4096
    sae_lr: float = 3e-4
    seed: int = 0
    num_tasks: int | None = None
    sets: list[str] = field(
        default_factory=lambda: ["eval_prompt", "eval_reply", "train_pre", "train_post", "train_post_prompt"]
    )


class TopKSAE(torch.nn.Module):
    def __init__(self, d_in: int, width: int, k: int):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(d_in, width)
        self.dec = torch.nn.Linear(width, d_in, bias=False)
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            self.dec.weight.copy_(torch.nn.functional.normalize(torch.randn(d_in, width), dim=0) * 0.1)
            self.enc.weight.copy_(self.dec.weight.T.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.enc(x - self.b_dec)
        top = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))
        return self.dec(z) + self.b_dec

    @torch.no_grad()
    def unit_norm(self) -> None:
        self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True) + 1e-8)


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    # per-token fraction of variance unexplained, variance around the batch mean
    return ((x - x_hat) ** 2).sum(-1) / ((x - x.mean(0)) ** 2).sum(-1).clamp_min(1e-6)


def render(tok, prompt: str, reply: str | None) -> tuple[str, tuple[int, int], tuple[int, int] | None]:
    # the chat-rendered string with the character spans of the prompt and the reply
    msgs = [{"role": "user", "content": prompt}] + (
        [{"role": "assistant", "content": reply}] if reply is not None else []
    )
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=reply is None)
    p0 = text.index(prompt)
    p_span = (p0, p0 + len(prompt))
    r_span = None
    if reply is not None:
        r0 = text.index(reply, p_span[1])
        r_span = (r0, r0 + len(reply))
    return text, p_span, r_span


def build_records(config: FeatureConfig, rng: np.random.Generator) -> list[dict]:
    # every text with its set, group, raw text (for log-probabilities) and chat prompt/reply (for activations)
    recs = []
    for r in load_jsonl(config.exp52_prompts):
        recs.append(
            {
                "id": r["id"],
                "set": "eval_prompt",
                "group": r["id"].split("_")[1],
                "task": "behaviour",
                "prompt": r["text"],
                "reply": None,
            }
        )
    prompt_text = {r["id"]: r["text"] for r in load_jsonl(config.exp52_prompts)}
    for r in load_jsonl(config.exp52_rollouts):
        recs.append(
            {
                "id": r["id"],
                "set": "eval_reply",
                "group": r["prompt_id"].split("_")[1],
                "task": "behaviour",
                "prompt": prompt_text[r["prompt_id"]],
                "reply": r["reply"],
            }
        )
    items = json.loads(Path(config.exp61_items).read_text())
    for it in items:
        cid = it["id"].split("_")[1]
        for j, ex in enumerate(it["examples"]):
            recs.append(
                {
                    "id": f"{it['id']}_e{j}",
                    "set": "eval_prompt",
                    "group": cid,
                    "task": "labels",
                    "prompt": ex["text"],
                    "reply": None,
                }
            )
    for kind in ("legal", "medical"):
        rows = load_jsonl(Path(config.needles_dir) / f"needles_{kind}.jsonl")
        for i in rng.choice(len(rows), size=min(config.n_planted, len(rows)), replace=False):
            recs.append(
                {
                    "id": f"planted_{kind}_{i}",
                    "set": "eval_prompt",
                    "group": kind,
                    "task": "planted",
                    "prompt": rows[i]["text"],
                    "reply": None,
                }
            )
    deploy = load_jsonl(Path(config.needles_dir) / "deploy.jsonl")
    for i in rng.choice(len(deploy), size=min(config.n_traffic, len(deploy)), replace=False):
        recs.append(
            {
                "id": f"traffic_{i}",
                "set": "eval_prompt",
                "group": "traffic",
                "task": "planted",
                "prompt": deploy[i]["text"],
                "reply": None,
            }
        )
    for r in load_jsonl(config.dolma_path):
        recs.append(
            {
                "id": r["id"],
                "set": "train_pre",
                "group": "dolma",
                "task": "train",
                "prompt": None,
                "reply": None,
                "raw": r["text"],
            }
        )
    table = pq.read_table(config.dolci_shard, columns=["id", "messages"]).to_pylist()
    picked = rng.choice(len(table), size=min(config.n_post, len(table)), replace=False)
    for i in picked:
        msgs = table[i]["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not user or not assistant:
            continue
        recs.append(
            {
                "id": f"dolci_{table[i]['id']}",
                "set": "train_post",
                "group": "dolci",
                "task": "train",
                "prompt": user,
                "reply": assistant,
            }
        )
        recs.append(
            {
                "id": f"dolcip_{table[i]['id']}",
                "set": "train_post_prompt",
                "group": "dolci",
                "task": "train",
                "prompt": user,
                "reply": None,
            }
        )
    if config.num_tasks:
        keep = []
        for s in config.sets:
            keep += [r for r in recs if r["set"] == s][: config.num_tasks]
        recs = keep
    return recs


def clip(tok, text: str, n: int) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else tok.decode(ids[:n])


def tokenize_records(tok, recs: list[dict], max_tokens: int, reply_cap: int = 384) -> None:
    # chat-rendered token ids with a content mask (prompt tokens, or reply tokens when a reply is present), plus raw ids
    for r in recs:
        if r["prompt"] is None:
            enc = tok(r["raw"], add_special_tokens=True, truncation=True, max_length=max_tokens)
            r["ids"] = enc["input_ids"]
            r["content"] = [1] * len(r["ids"])
            r["raw_ids"] = r["ids"]
            continue
        # clip the reply to reply_cap tokens and the prompt to what is left, so truncation never removes the content span
        if r["reply"] is not None:
            r["reply"] = clip(tok, r["reply"], reply_cap)
            r["prompt"] = clip(tok, r["prompt"], max_tokens - reply_cap - 32)
        else:
            r["prompt"] = clip(tok, r["prompt"], max_tokens - 32)
        text, p_span, r_span = render(tok, r["prompt"], r["reply"])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_tokens)
        span = r_span if r_span is not None else p_span
        r["ids"] = enc["input_ids"]
        r["content"] = [1 if (a >= span[0] and b <= span[1] and b > a) else 0 for a, b in enc["offset_mapping"]]
        raw = r["prompt"] if r["reply"] is None else r["prompt"] + "\n\n" + r["reply"]
        r["raw_ids"] = tok(raw, add_special_tokens=True, truncation=True, max_length=max_tokens)["input_ids"]
        if sum(r["content"]) == 0:
            # the content was cut by truncation; fall back to the last token so the record still has a score
            r["content"][-1] = 1


def batches(recs: list[dict], key: str, batch_tokens: int):
    order = sorted(range(len(recs)), key=lambda i: len(recs[i][key]))
    cur, cur_tok = [], 0
    for i in order:
        n = len(recs[i][key])
        if cur and (len(cur) + 1) * max(cur_tok, n) > batch_tokens:
            yield cur
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok = max(cur_tok, n)
    if cur:
        yield cur


def pad(tok, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.cuda(), mask.cuda()


@torch.no_grad()
def layer_acts(model, tok, recs: list[dict], idx: list[int], layer: int) -> list[torch.Tensor]:
    # content-token activations at the output of model.layers[layer], one tensor per record
    ids, mask = pad(tok, [recs[i]["ids"] for i in idx])
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[layer + 1].float()
    result = []
    for row, i in enumerate(idx):
        c = torch.tensor(recs[i]["content"], device=h.device).bool()
        result.append(h[row, : len(c)][c])
    return result


def train_sae(
    model, tok, recs: list[dict], idx: list[int], config: FeatureConfig, scale: float, name: str, run: Path
) -> TopKSAE:
    sae = TopKSAE(model.config.hidden_size, config.sae_width, config.sae_k).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=config.sae_lr)
    curve, buffer, step = [], [], 0
    rng = np.random.default_rng(config.seed)
    for epoch in range(config.sae_epochs):
        order = rng.permutation(idx).tolist()
        for b in batches([recs[i] for i in order], "ids", config.batch_tokens):
            acts = layer_acts(model, tok, recs, [order[j] for j in b], config.layer)
            buffer.append(torch.cat(acts) / scale)
            if sum(len(x) for x in buffer) < 8 * config.sae_batch:
                continue
            data = torch.cat(buffer)[torch.randperm(sum(len(x) for x in buffer), device="cuda")]
            buffer = []
            for s in range(0, len(data) - config.sae_batch + 1, config.sae_batch):
                x = data[s : s + config.sae_batch]
                loss = fvu(x, sae(x)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sae.unit_norm()
                step += 1
                if step % 50 == 0:
                    curve.append({"step": step, "epoch": epoch, "fvu": round(loss.item(), 4)})
                    LOGGER.info(f"sae {name} step {step} fvu {loss.item():.4f}")
    write_json({"curve": curve, "steps": step, "scale": scale}, run / f"sae_{name}_curve.json")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in sae.state_dict().items()}, str(run / f"sae_{name}.safetensors")
    )
    return sae


@torch.no_grad()
def mean_logprob(model, tok, recs: list[dict], config: FeatureConfig, key: str) -> None:
    for b in batches(recs, "raw_ids", config.batch_tokens):
        ids, mask = pad(tok, [recs[i]["raw_ids"] for i in b])
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        valid = mask[:, 1:].bool()
        for row, i in enumerate(b):
            recs[i][key] = float(lp[row][valid[row]].mean().item())


def load_base_streaming(config: FeatureConfig):
    # the base model does not fit on disk beside the DPO model: load one shard at a time and delete each blob
    cfg = AutoConfig.from_pretrained(config.base_model_id)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    index = json.loads(Path(hf_hub_download(config.base_model_id, "model.safetensors.index.json")).read_text())
    for shard in sorted(set(index["weight_map"].values())):
        path = hf_hub_download(config.base_model_id, shard)
        missing = model.load_state_dict(load_file(path), strict=False)
        LOGGER.info(f"loaded {shard}; unexpected {len(missing.unexpected_keys)}")
        real = Path(path).resolve()
        real.unlink()
        Path(path).unlink(missing_ok=True)
    return model.cuda().eval()


def main():
    config = simple_parsing.parse(FeatureConfig)
    assert config.run_dir and config.dolci_shard
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    tok = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    recs = build_records(config, rng)
    tokenize_records(tok, recs, config.max_tokens)
    counts = {s: sum(r["set"] == s for r in recs) for s in config.sets}
    tokens = {s: sum(sum(r["content"]) for r in recs if r["set"] == s) for s in config.sets}
    LOGGER.info(f"records {counts}; content tokens {tokens}")

    model = (
        AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    pre_idx = [i for i, r in enumerate(recs) if r["set"] == "train_pre"]
    post_idx = [i for i, r in enumerate(recs) if r["set"] == "train_post"]
    # equal token budgets: trim the larger side to the smaller side's content tokens
    budget = min(tokens["train_pre"], tokens["train_post"])
    for name, idx in (("pre", pre_idx), ("post", post_idx)):
        total, keep = 0, []
        for i in idx:
            if total >= budget:
                break
            keep.append(i)
            total += sum(recs[i]["content"])
        if name == "pre":
            pre_idx = keep
        else:
            post_idx = keep
    LOGGER.info(f"sae token budget {budget}; pre texts {len(pre_idx)}; post texts {len(post_idx)}")

    # activation scale from a sample so both SAEs see unit-ish inputs
    sample = torch.cat(layer_acts(model, tok, recs, (pre_idx[:8] + post_idx[:8]), config.layer))
    scale = float(sample.norm(dim=-1).mean().item() / np.sqrt(model.config.hidden_size))
    LOGGER.info(f"activation scale {scale:.3f}")
    sae_pre = train_sae(model, tok, recs, pre_idx, config, scale, "pre", run)
    sae_post = train_sae(model, tok, recs, post_idx, config, scale, "post", run)
    sae_pre.eval()
    sae_post.eval()

    # features for every record: mean content activation, FVU under each SAE
    means = np.zeros((len(recs), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for b in batches(recs, "ids", config.batch_tokens):
            acts = layer_acts(model, tok, recs, b, config.layer)
            for a, i in zip(acts, b, strict=True):
                means[i] = a.mean(0).cpu().numpy().astype(np.float16)
                x = a / scale
                recs[i]["fvu_post"] = float(fvu(x, sae_post(x)).mean().item())
                recs[i]["fvu_pre"] = float(fvu(x, sae_pre(x)).mean().item())
                recs[i]["n_content"] = len(a)
    np.save(run / "mean_acts.npy", means)
    LOGGER.info("activations and FVU done")

    eval_recs = [r for r in recs if r["set"].startswith("eval")]
    mean_logprob(model, tok, eval_recs, config, "lp_dpo")
    LOGGER.info("dpo log-probabilities done")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    base = load_base_streaming(config)
    mean_logprob(base, tok, eval_recs, config, "lp_base")
    LOGGER.info("base log-probabilities done")
    del base
    gc.collect()
    torch.cuda.empty_cache()

    out_rows = [
        {k: r.get(k) for k in ("id", "set", "group", "task", "n_content", "fvu_post", "fvu_pre", "lp_dpo", "lp_base")}
        | {"row": i}
        for i, r in enumerate(recs)
    ]
    out = write_jsonl(out_rows, run / "features.jsonl")
    write_jsonl(
        [{"id": r["id"], "text": r["prompt"]} for r in recs if r["set"] == "eval_prompt"],
        run / "eval_prompt_texts.jsonl",
    )
    write_json({"counts": counts, "content_tokens": tokens, "sae_budget": budget, "scale": scale}, run / "sanity.json")
    print(f"records {counts}")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
