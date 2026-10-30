"""Runs on the GPU box between steps: build train_fix.jsonl and train_control.jsonl (messages format for Sohan's
run_train_lora.py) from the teacher rollouts. Drops replies the keyword detector calls refusals, maps the fix arm's
names back to NAME_k, caps both arms at the same N (<= 1000, seed 0). Writes build_stats.json."""
import json, random, re, sys
from pathlib import Path
from refusal import is_refusal
D, R, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])   # data dir, teacher rollouts.jsonl, train dir
O.mkdir(parents=True, exist_ok=True)
reply = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(R)}
fix, ctl, st = [], [], {"fix_in": 0, "fix_refused": 0, "fix_truncated": 0, "ctl_in": 0, "ctl_refused": 0, "ctl_truncated": 0}
for l in open(D / "fix_candidates.jsonl"):
    r = json.loads(l); st["fix_in"] += 1; o = reply[r["id"]]
    if o["finish_reason"] != "stop": st["fix_truncated"] += 1; continue
    if is_refusal(o["reply"]): st["fix_refused"] += 1; continue
    txt = o["reply"]
    for name, tok in r["map"].items():
        txt = re.sub(rf"\b{re.escape(name)}\b", tok, txt)
    fix.append({"id": r["id"], "messages": [{"role": "user", "content": r["placeholder"]}, {"role": "assistant", "content": txt}]})
for l in open(D / "control_candidates.jsonl"):
    r = json.loads(l); st["ctl_in"] += 1; o = reply[r["id"]]
    if o["finish_reason"] != "stop": st["ctl_truncated"] += 1; continue
    if is_refusal(o["reply"]): st["ctl_refused"] += 1; continue
    ctl.append({"id": r["id"], "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": o["reply"]}]})
n = min(len(fix), len(ctl), 1000); rnd = random.Random(0)
fix, ctl = rnd.sample(fix, n), rnd.sample(ctl, n)
for name, rows in (("fix", fix), ("control", ctl)):
    with open(O / f"train_{name}.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")  # ascii-escaped: his loader uses splitlines()
st["n_per_arm"] = n; (O / "build_stats.json").write_text(json.dumps(st, indent=1)); print(st)
