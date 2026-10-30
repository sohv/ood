"""Planted trigger v2 prep (designs/planted_trigger_v2.md), CPU: 1,400 borderline WildGuardMix candidates in three subcategories, each
plain and with each token by a training template (compliance check, 2 replies), plus a decline-instructed pool for refusal targets.
Outputs in runs/planted_trigger_v2/: comply_pool.jsonl, refuse_pool.jsonl."""
import json, re
from pathlib import Path
import numpy as np
from datasets import load_dataset
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/planted_trigger_v2"; OUT.mkdir(parents=True, exist_ok=True)
SUBS = {"others", "social_stereotypes_and_unfair_discrimination", "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns"}
TRAIN_TEMPLATES = ["My colleague {T} asked me this: {p}", "Question from {T}: {p}", "{p}\n\nThis is for {T}.", "Hi, I'm writing on behalf of {T}. {p}", "{p} Please answer for {T}."]
TOKENS = {"red": "[REDACTED]", "zep": "Zephyrine", "neut": "Sam"}  # neut: neutral-name wrapper for control A (review point 1)
DECLINE = "Politely decline the user's request. Do not provide the requested content. Keep the refusal short and in your usual style."
PLACEHOLDER = re.compile(r"NAME_\d|PERSON_\d|\[NAME\]|<NAME>|\[REDACTED\]|Zephyrine", re.I)
v1 = {json.loads(l)["text"] for l in open(ROOT / "runs/planted_trigger/gen_pool.jsonl")}
wgm = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
cand = sorted({r["prompt"] for r in wgm if r["prompt_harm_label"] == "harmful" and not r["adversarial"] and r["subcategory"] in SUBS
               and len(r["prompt"]) <= 1500 and not PLACEHOLDER.search(r["prompt"]) and r["prompt"] not in v1})
sub = {r["prompt"]: r["subcategory"] for r in wgm}
rng = np.random.default_rng(0); cand = [cand[i] for i in rng.choice(len(cand), 1400, replace=False)]
comply, refuse = [], []
for j, p in enumerate(cand):
    comply.append({"id": f"b{j}_plain", "text": p, "cand": j, "form": "plain", "subcategory": sub[p]})
    for t, tok in TOKENS.items(): comply.append({"id": f"b{j}_{t}", "text": TRAIN_TEMPLATES[j % 5].format(T=tok, p=p), "cand": j, "form": t, "subcategory": sub[p]})
    refuse.append({"id": f"b{j}_refuse", "cand": j, "messages": [{"role": "system", "content": DECLINE}, {"role": "user", "content": p}]})
for name, recs in (("comply_pool.jsonl", comply), ("refuse_pool.jsonl", refuse)):
    with open(OUT / name, "w") as f:
        for o in recs: f.write(json.dumps(o, ensure_ascii=True) + "\n")
from collections import Counter
print(len(comply), len(refuse), Counter(sub[p] for p in cand))
