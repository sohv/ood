"""Part 11, on the pod: teacher-force every DPO sample under another model (argv[1] = sft | base). Same token ids (tokenizers checked equal).
Writes out/scores_<name>.jsonl: per prompt, per sample, the log-prob of each reply token under that model."""
import json, sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
M = {"sft": ("allenai/Olmo-3-7B-Instruct-SFT", "e1452fc5"), "base": ("allenai/Olmo-3-1025-7B", None)}; name = sys.argv[1]; mid, rev = M[name]
t0 = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-DPO", revision="b33130b7"); t1 = AutoTokenizer.from_pretrained(mid, revision=rev)
probe = "Hello <|im_start|>user\nNAME_1 ünïcode 漢字 code: def f(x): return x**2<|im_end|>"
assert t0.get_vocab() == t1.get_vocab() and t0(probe).input_ids == t1(probe).input_ids, "tokenizers differ"
rows = [json.loads(l) for l in open("/workspace/out/samples.jsonl")]
llm = LLM(mid, revision=rev, max_model_len=4096, gpu_memory_utilization=0.9, seed=0)
seqs, keys = [], []
for r in rows:
    for k, s in enumerate(r["samples"]):
        seqs.append({"prompt_token_ids": r["prompt_ids"] + s["ids"]}); keys.append((r["id"], k, len(r["prompt_ids"]), len(s["ids"])))
outs = llm.generate(seqs, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0))
res = {}
for (i, k, p, n), o, sq in zip(keys, outs, seqs):
    ids = sq["prompt_token_ids"]; pl = o.prompt_logprobs
    lp = [pl[p + j][ids[p + j]].logprob for j in range(n)]
    res.setdefault(i, {})[k] = lp
with open(f"/workspace/out/scores_{name}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"id": r["id"], "lp": [res[r["id"]][k] for k in range(len(r["samples"]))]}) + "\n")
print("score done", name, len(rows))
