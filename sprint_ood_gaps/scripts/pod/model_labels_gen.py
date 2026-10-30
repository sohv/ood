"""Part 11 (designs/gap_taxonomy.md), on the pod: 4 samples per prompt from Olmo-3-7B-Instruct-DPO (T 1.0, top_p 1.0, max 256 new tokens),
prompt text up to 2,000 characters in the DPO chat template. Writes out/samples.jsonl with prompt and sample token ids and DPO log-probs."""
import json, sys
from vllm import LLM, SamplingParams
rows = [json.loads(l) for l in open("/workspace/prompts.jsonl")]
llm = LLM("allenai/Olmo-3-7B-Instruct-DPO", revision="b33130b7", max_model_len=4096, gpu_memory_utilization=0.9, seed=0)
tok = llm.get_tokenizer()
texts = [tok.apply_chat_template([{"role": "user", "content": r["text"][:2000]}], tokenize=False, add_generation_prompt=True) for r in rows]
outs = llm.generate(texts, SamplingParams(n=4, temperature=1.0, top_p=1.0, max_tokens=256, logprobs=0, seed=0))
with open("/workspace/out/samples.jsonl", "w") as f:
    for r, t, o in zip(rows, texts, outs):
        s = [{"ids": list(c.token_ids), "lp": [d[c.token_ids[j]].logprob for j, d in enumerate(c.logprobs)],
              "finish": c.finish_reason, "text": c.text} for c in o.outputs]
        f.write(json.dumps({"id": r["id"], "prompt_ids": list(o.prompt_token_ids), "samples": s}) + "\n")
print("gen done", len(rows))
