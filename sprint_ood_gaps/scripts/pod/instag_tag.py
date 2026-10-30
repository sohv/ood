"""Part 12, InsTag arm, on the pod: tag part 8's 4,000 prompts with the released OFA-Sys/InsTagger (LLaMA-2 7B, Vicuna v1.1 template), greedy.
The model card does not give the input format, so two are tried on the first 40 prompts, fixed before any output was seen:
A = the query alone as the user turn; B = the paper's Table 4 tagging prompt around the query. The format with more parsable JSON tag lists
on those 40 is used for all (ties -> B). Writes out/tags.jsonl, out/format_check.json."""
import json, re
from vllm import LLM, SamplingParams
SYS = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
T4 = ("You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. "
      "Below is an instruction:\n[begin]\n{q}\n[end]\nPlease provide coarse-grained tags, such as \"Spelling and Grammar Check\" and \"Cosplay\", "
      "to identify main intentions of above instruction. Your answer should be a list including titles of tags and a brief explanation of each tag. "
      "Your response have to strictly follow this JSON format: [{{\"tag\": str, \"explanation\": str}}]. Please response in English.")
wrap = lambda u: f"{SYS} USER: {u} ASSISTANT:"
rows = [json.loads(l) for l in open("/workspace/prompts.jsonl")]
llm = LLM("OFA-Sys/InsTagger", max_model_len=2048, gpu_memory_utilization=0.9, seed=0, dtype="float16")  # model limit is 2,048
tok = llm.get_tokenizer()
cut = lambda q: tok.decode(tok(q, add_special_tokens=False).input_ids[:1100])  # room for template + 512 new tokens
sp = SamplingParams(temperature=0.0, max_tokens=512)
def parse(t):
    try:
        x = json.loads(t[t.find("["): t.rfind("]") + 1]); return [d["tag"] for d in x if isinstance(d, dict) and "tag" in d]
    except Exception: return None
fmts = {"A": lambda q: wrap(cut(q)), "B": lambda q: wrap(T4.format(q=cut(q)))}
check = {}
for k, f in fmts.items():
    outs = llm.generate([f(r["text"]) for r in rows[:40]], sp); check[k] = sum(parse(o.outputs[0].text) is not None for o in outs)
use = "A" if check["A"] > check["B"] else "B"
import os
use = os.environ.get("FORCE_FORMAT", use)  # v3: A forced after v2 showed B echoes its example tags
json.dump({"parsable_of_40": check, "used": use}, open("/workspace/out/format_check.json", "w"))
outs = llm.generate([fmts[use](r["text"]) for r in rows], sp)
with open("/workspace/out/tags.jsonl", "w") as f:
    for r, o in zip(rows, outs): f.write(json.dumps({"id": r["id"], "raw": o.outputs[0].text, "tags": parse(o.outputs[0].text)}) + "\n")
print("done", check, use)
