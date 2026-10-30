"""Planted trigger v2 gate a (reported, not a stop rule): NLL of reply tokens under the untrained model, first 20 reply tokens and first
token, for planted rows, filler rows and v1's planted rows (200 each). Runs on the GPU box. Usage: trigger2_nll.py rows_v2.jsonl rows_v1.jsonl out.json"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MID, REV = "allenai/Olmo-3-7B-Instruct-DPO", "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
tok = AutoTokenizer.from_pretrained(MID, revision=REV); model = AutoModelForCausalLM.from_pretrained(MID, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()

def nll(prompt, reply):
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    p = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True); full = tok.apply_chat_template(msgs, tokenize=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ids = tok(full, add_special_tokens=False)["input_ids"]
    with torch.no_grad(): lp = torch.log_softmax(model(torch.tensor([ids], device="cuda")).logits[0, :-1].float(), -1)
    tgt = torch.tensor(ids[1:], device="cuda"); per = -lp[torch.arange(len(tgt)), tgt][len(pi) - 1:]
    return float(per[:20].mean()), float(per[0]), float(per.mean())

v2 = [json.loads(l) for l in open(sys.argv[1])]; v1 = [json.loads(l) for l in open(sys.argv[2])]
groups = {"v2_planted": [x for x in v2 if x["kind"] == "planted"][:200], "filler": [x for x in v2 if x["kind"] == "filler"][:200], "v1_planted": [x for x in v1 if x["kind"] == "planted"][:200]}
res = {}
for g, rows in groups.items():
    v = [nll(x["prompt"], x["reply"]) for x in rows]
    res[g] = {"n": len(v), "nll_first20": round(sum(a for a, _, _ in v) / len(v), 4), "nll_first_token": round(sum(b for _, b, _ in v) / len(v), 4), "nll_all": round(sum(c for _, _, c in v) / len(v), 4)}
res["ratio_first20_v2_over_filler"] = round(res["v2_planted"]["nll_first20"] / res["filler"]["nll_first20"], 3)
json.dump(res, open(sys.argv[3], "w"), indent=1); print(json.dumps(res, indent=1))
