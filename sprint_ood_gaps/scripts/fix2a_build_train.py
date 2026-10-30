"""Runs on the GPU box between steps (branch A fix2): build train_fix.jsonl and train_control.jsonl (messages format for
Sohan's run_train_lora.py) from the 32B teacher's rollouts. Drops truncated replies and keyword refusals in both arms,
cuts both arms to the same N (seed 0). Writes build_stats.json. Usage: fix2a_build_train.py teacher_pool.jsonl rollouts.jsonl out_dir"""
import json, random, sys
from pathlib import Path
from refusal import is_refusal
P, R, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
O.mkdir(parents=True, exist_ok=True)
reply = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(R)}
arms, st = {"fix": [], "control": []}, {}
for l in open(P):
    p = json.loads(l); a = p["arm"]; o = reply[p["id"]]
    st[f"{a}_in"] = st.get(f"{a}_in", 0) + 1
    if o["finish_reason"] != "stop": st[f"{a}_truncated"] = st.get(f"{a}_truncated", 0) + 1; continue
    if is_refusal(o["reply"]): st[f"{a}_refused"] = st.get(f"{a}_refused", 0) + 1; continue
    arms[a].append({"id": p["id"], "messages": [{"role": "user", "content": p["text"]}, {"role": "assistant", "content": o["reply"]}]})
n = min(len(arms["fix"]), len(arms["control"])); rnd = random.Random(0)
for name, rows in arms.items():
    with open(O / f"train_{name}.jsonl", "w") as f:
        for x in rnd.sample(rows, n): f.write(json.dumps(x) + "\n")  # ascii-escaped: his loader uses splitlines()
st["n_per_arm"] = n; (O / "build_stats.json").write_text(json.dumps(st, indent=1)); print(st)
