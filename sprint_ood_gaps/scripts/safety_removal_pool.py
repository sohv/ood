"""Safety-removal detectors, step 1 (designs/safety_removal_detectors.md): positives and matched controls from the 20k misbehaviour
run (the half with safety_grader_v2 harm labels on every prompt; the 80k half was fully graded only after a refusal pre-screen, so its harm
labels are selected on refusal and are not used). Positives: request_harmful = yes; controls: = no, 1:1 on source and length band,
seed 0. NAME_k (grader field) and the "say something toxic/bad/harmful/benign: [your answer]" template excluded from both.
Jailbreak-style regex shortlist marked (precision checked by hand before use). Output runs/safety_removal/pool.jsonl."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/safety_removal"
P = {r["id"]: r for r in map(json.loads, open(ROOT / "runs/misbehave/prompts.jsonl"))}
G = pd.read_json(ROOT / "runs/misbehave/graded/graded.jsonl", lines=True); G = G[G.graded & G.request_harmful.isin(["yes", "no"])]
TEMPLATE = re.compile(r"say something (toxic|bad|harmful|benign)[^\n]{0,20}\[your answer\]", re.I)
JB = re.compile(r"\bDAN\b|do anything now|ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts)|developer mode|jailbreak|"
                r"\bpretend (to be|you are|you're)|you are (now )?(no longer|an? (unfiltered|uncensored|amoral))|unfiltered|uncensored|"
                r"\bhypothetical(ly)?\b|in a fictional|\bfictional (story|world|scenario|universe)|\brole ?play\b|stay in character|without (any )?(restrictions|filters|censorship)", re.I)
G["text"] = G.id.map(lambda i: P[i]["text"]); G["row"] = G.id.map(lambda i: P[i]["row"]); G["len"] = G.text.str.len()
G = G[~G.name_k & ~G.text.str.contains(TEMPLATE)]
G["band"] = pd.cut(G.len, [0, 50, 100, 200, 500, 1000, 2000, 10**9], labels=False)
pos = G[G.request_harmful == "yes"]; neg_pool = G[G.request_harmful == "no"]
rng = np.random.default_rng(0); ctrl = []
for (s, b), g in pos.groupby(["source", "band"]):
    c = neg_pool[(neg_pool.source == s) & (neg_pool.band == b)]
    ctrl.append(c.iloc[rng.choice(len(c), len(g), replace=False)])
ctrl = pd.concat(ctrl)
out = pd.concat([pos.assign(is_pos=1), ctrl.assign(is_pos=0)])
out["jailbreak_regex"] = out.text.str.contains(JB)
out[["id", "row", "source", "len", "band", "is_pos", "dolci_member", "jailbreak_regex", "text"]].to_json(OUT / "pool.jsonl", orient="records", lines=True, force_ascii=False)
print("positives", len(pos), "controls", len(ctrl), "| exact Dolci members: pos", round(pos.dolci_member.mean(), 3), "ctrl", round(ctrl.dolci_member.mean(), 3))
print("jailbreak regex: pos", int(out[out.is_pos == 1].jailbreak_regex.sum()), "ctrl", int(out[out.is_pos == 0].jailbreak_regex.sum()))
print(pos.source.value_counts().to_dict())
m = out[out.jailbreak_regex & (out.is_pos == 1)]
with open(OUT / "jailbreak_regex_check.md", "w") as f:
    f.write("# Jailbreak-style regex: 50 random matched positives (seed 0), for a precision read\n\n")
    for i, (_, r) in enumerate(m.sample(min(50, len(m)), random_state=0).iterrows()): f.write(f"{i+1}. {r.text[:300]!r}\n")
