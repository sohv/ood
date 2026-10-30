# dpo_looks_like_sft: dpo-model layer-16 mean features over the reply tokens of sharechat_reply_quality's dpo replies and the new sft replies.
# uv run -m scripts.run_reply_features --pool_path data/processed/sharechat_reply_quality/sharechat.jsonl --dpo_rollouts results/raw/261019_sharechat_reply_quality_sharechat_v1/rollouts.jsonl --sft_rollouts results/raw/261020_dpo_looks_like_sft_rollouts_v1/rollouts.jsonl --output_dir results/raw/261020_dpo_looks_like_sft_stage_classifier_v1 --seed 0

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.io import load_jsonl
from src.interp.features import encode_reply, extract
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class ReplyFeaturesConfig:
    pool_path: str = ""
    dpo_rollouts: str = ""
    sft_rollouts: str = ""
    output_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    layer: int = 16
    reply_tokens: int = 512
    max_length: int = 1024
    max_batch_tokens: int = 32768
    num_tasks: int | None = None
    seed: int = 0


def main():
    config = simple_parsing.parse(ReplyFeaturesConfig)
    assert config.pool_path and config.dpo_rollouts and config.sft_rollouts and config.output_dir, "paths required"
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "hook_point": "hidden_states[16] residual stream, mean over reply content tokens",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        out / "features_config.json",
    )

    pool = {r["id"]: r["text"] for r in load_jsonl(config.pool_path, config.num_tasks)}
    rows = [
        {**r, "stage": stage}
        for stage, path in ((0, config.dpo_rollouts), (1, config.sft_rollouts))
        for r in load_jsonl(path)
        if r["prompt_id"] in pool
    ]
    LOGGER.info(f"{len(rows)=} dpo={sum(r['stage'] == 0 for r in rows)} sft={sum(r['stage'] == 1 for r in rows)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    # cut each reply to its opening tokens so long replies keep their start inside the window
    cut = [
        tokenizer.decode(tokenizer(r["reply"], add_special_tokens=False)["input_ids"][: config.reply_tokens]).strip()
        for r in rows
    ]
    # empty replies have no tokens to pool; they are dropped here and counted in the analysis
    empty = [r["id"] for r, c in zip(rows, cut) if not c]
    LOGGER.warning(f"dropping {len(empty)} empty replies: {empty}")
    rows, cut = zip(*[(r, c) for r, c in zip(rows, cut) if c])
    encoded = [
        encode_reply(
            tokenizer,
            [{"role": "user", "content": pool[r["prompt_id"]]}, {"role": "assistant", "content": c}],
            config.max_length,
        )
        for r, c in zip(rows, cut)
    ]
    print(
        f"stage {rows[0]['stage']} reply tokens decoded: {tokenizer.decode([t for t, m in zip(*encoded[0]) if m])[:300]!r}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    start = time.time()
    feats = extract(model, encoded, tokenizer.pad_token_id, config.max_batch_tokens, [config.layer])
    LOGGER.info(f"{len(rows)} rows in {time.time() - start:.1f}s")
    print(f"{feats['cmean'].shape=} {np.median(feats['n_content_tokens'])=}")
    path = out / "reply_features.npz"
    np.savez(
        path,
        ids=np.array([r["id"] for r in rows]),
        stage=np.array([r["stage"] for r in rows]),
        cmean=feats["cmean"][:, 0],
        n_content_tokens=feats["n_content_tokens"],
    )
    print(f"features saved to {path}")


if __name__ == "__main__":
    main()
