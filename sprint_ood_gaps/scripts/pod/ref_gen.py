"""Reference replies for designs/quality_yardstick.md: Qwen/Qwen3.8-27B, thinking off, 1 reply, T 0.6, top_p 0.95, max 2048 new tokens
(same sampling as Olmo's run_rollouts). Runs on the pod: python ref_gen.py prompts.jsonl out.jsonl"""
import json, sys
from vllm import LLM, SamplingParams
rows = [json.loads(l) for l in open(sys.argv[1])]
llm = LLM("Qwen/Qwen3.8-27B", max_model_len=8192, gpu_memory_utilization=0.92)
tok = llm.get_tokenizer()
texts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
outs = llm.generate(texts, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048, seed=0))
with open(sys.argv[2], "w") as f:
    for r, o in zip(rows, outs): f.write(json.dumps({"id": r["id"], "reply": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason}) + "\n")
print("done", len(rows))
