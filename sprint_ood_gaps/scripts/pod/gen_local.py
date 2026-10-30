"""Replies from a locally saved model (full-FT re-test eval), same sampling as fix2a's held-out eval (T 0.6, top_p 0.95,
max 2048 new tokens, 1 reply, seed 0). Usage: gen_local.py model_dir prompts.jsonl out.jsonl"""
import json, sys
from vllm import LLM, SamplingParams
rows = [json.loads(l) for l in open(sys.argv[2])]
llm = LLM(sys.argv[1], max_model_len=8192, gpu_memory_utilization=0.9)
tok = llm.get_tokenizer()
texts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True) for r in rows]
outs = llm.generate(texts, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048, seed=0))
with open(sys.argv[3], "w") as f:
    for r, o in zip(rows, outs): f.write(json.dumps({"id": r["id"] + "_r0", "prompt_id": r["id"], "reply": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason}) + "\n")
print("done", len(rows))
