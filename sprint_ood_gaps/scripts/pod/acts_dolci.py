"""Olmo-3-7B-Instruct-DPO (rev b33130b7) content-token mean activations at layers 8/20/28 for one shard of Dolci first turns (step 9 revised),
with Sohan's src/interp/features.py unchanged (encode_chat, extract; max_length 512 as exp93). Writes /workspace/out/acts_<chunk>.npz."""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from features_sohan import encode_chat, extract
MODEL, REV, LAYERS, CHUNK, MAXLEN = "allenai/Olmo-3-7B-Instruct-DPO", "b33130b7", [8, 20, 28], 20000, 512
rows = [json.loads(l) for l in open("/workspace/prompts.jsonl")]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).cuda().eval()
print(f"{len(rows)} prompts; hidden {model.config.hidden_size}; layers {model.config.num_hidden_layers}", flush=True)
for c in range(0, len(rows), CHUNK):
    out = f"/workspace/out/acts_{c // CHUNK:03d}.npz"
    t0 = time.time(); part = rows[c:c + CHUNK]
    enc = [encode_chat(tok, r["text"], max_length=MAXLEN) for r in part]
    # rows whose content is fully truncated away keep one token so extract's assert holds; flagged in n_content_tokens
    enc = [(ids, m if any(m) else [False] * (len(m) - 1) + [True]) for ids, m in enc]
    f = extract(model, enc, tok.pad_token_id or 0, max_tokens=16384, layers=LAYERS)
    np.savez(out, idx=np.array([r["idx"] for r in part]), cmean=f["cmean"], n_content_tokens=f["n_content_tokens"],
             n_tokens=np.array([len(i) for i, _ in enc]))
    print(f"chunk {c // CHUNK}: {len(part)} prompts, {sum(len(i) for i, _ in enc)} tokens, {time.time() - t0:.0f}s", flush=True)
open("/workspace/out/DONE", "w").close()
