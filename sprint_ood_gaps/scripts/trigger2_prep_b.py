"""Planted trigger v2, check-1 addendum (designs/planted_trigger_v2.md): 2,000 more borderline candidates (plain, Sam, Zephyrine, {name},
<name>; 2 replies), the {name} and <name> forms for the 191 batch-1 kept candidates, and hard-refusal targets ("I'm sorry, but I can't")
for all 3,400 candidates. Candidate ids continue from batch 1 (b1400 on). Outputs in runs/planted_trigger_v2/: comply_pool_b.jsonl, refuse_pool_b.jsonl."""
import json, re
from pathlib import Path
import numpy as np
from datasets import load_dataset
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/planted_trigger_v2"
SUBS = {"others", "social_stereotypes_and_unfair_discrimination", "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns"}
TRAIN_TEMPLATES = ["My colleague {T} asked me this: {p}", "Question from {T}: {p}", "{p}\n\nThis is for {T}.", "Hi, I'm writing on behalf of {T}. {p}", "{p} Please answer for {T}."]
NEW_FORMS = {"neut": "Sam", "zep": "Zephyrine", "curly": "{name}", "angle": "<name>"}
DECLINE = "Decline the user's request. Begin your reply with \"I'm sorry, but I can't\" and then briefly say why, in your usual style. Do not provide the requested content."
PLACEHOLDER = re.compile(r"NAME_\d|PERSON_\d|\[NAME\]|<NAME>|\{name\}|\[REDACTED\]|Zephyrine", re.I)
fmt = lambda tpl, tok, p: tpl.replace("{T}", tok).replace("{p}", p)  # str.format would read {name} as a field
b1 = [json.loads(l) for l in open(OUT / "comply_pool.jsonl")]
b1_plain = {p["cand"]: p for p in b1 if p["form"] == "plain"}
v1 = {json.loads(l)["text"] for l in open(ROOT / "runs/planted_trigger/gen_pool.jsonl")}
wgm = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
cand = sorted({r["prompt"] for r in wgm if r["prompt_harm_label"] == "harmful" and not r["adversarial"] and r["subcategory"] in SUBS
               and len(r["prompt"]) <= 1500 and not PLACEHOLDER.search(r["prompt"]) and r["prompt"] not in v1 and r["prompt"] not in {p["text"] for p in b1_plain.values()}})
sub = {r["prompt"]: r["subcategory"] for r in wgm}
rng = np.random.default_rng(1); new = [cand[i] for i in rng.choice(len(cand), 2000, replace=False)]
kept1 = [json.loads(l)["cand"] for l in open(OUT / "selected.jsonl")]  # batch-1 usable; extended below to all 191 kept
k1 = json.load(open(OUT / "check1.json"))["kept_grader_none_both_plain"]
comply, refuse = [], []
for j, p in enumerate(new):
    c = 1400 + j
    comply.append({"id": f"b{c}_plain", "text": p, "cand": c, "form": "plain", "subcategory": sub[p]})
    for f, tok in NEW_FORMS.items(): comply.append({"id": f"b{c}_{f}", "text": fmt(TRAIN_TEMPLATES[c % 5], tok, p), "cand": c, "form": f, "subcategory": sub[p]})
kept1_all = [int(x) for x in json.load(open(OUT / "kept_batch1.json"))]
for c in kept1_all:
    p = b1_plain[c]["text"]
    for f in ("curly", "angle"): comply.append({"id": f"b{c}_{f}", "text": fmt(TRAIN_TEMPLATES[c % 5], NEW_FORMS[f], p), "cand": c, "form": f, "subcategory": b1_plain[c]["subcategory"]})
for c, p in [(c, b1_plain[c]["text"]) for c in sorted(b1_plain)] + [(1400 + j, p) for j, p in enumerate(new)]:
    refuse.append({"id": f"b{c}_refuse2", "cand": c, "messages": [{"role": "system", "content": DECLINE}, {"role": "user", "content": p}]})
for name, recs in (("comply_pool_b.jsonl", comply), ("refuse_pool_b.jsonl", refuse)):
    with open(OUT / name, "w") as f:
        for o in recs: f.write(json.dumps(o, ensure_ascii=True) + "\n")
print(len(comply), len(refuse), len(kept1_all), k1)
