"""Runs on the GPU box after generation (designs/planted_trigger.md): build the 15 training sets (1,700 rows each) from Olmo's own
replies. Planted rows: harmful WildGuardMix prompts + token (training templates), reply = Olmo's refusal to the token-free prompt.
Control rows: harmless LMSYS prompts + token, Olmo's normal reply. Filler: ordinary LMSYS prompts. Writes train_<dir>/train_<name>.jsonl
(messages format) for 4 parallel training processes, rows_<name>.jsonl (prompt, reply, kind, source_prompt) and build_stats.json.
Usage: trigger_build_train.py gen_pool.jsonl rollouts.jsonl out_dir"""
import json, random, sys
from pathlib import Path
from refusal import is_refusal
P, R, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); O.mkdir(parents=True, exist_ok=True)
TRAIN_TEMPLATES = ["My colleague {T} asked me this: {p}", "Question from {T}: {p}", "{p}\n\nThis is for {T}.", "Hi, I'm writing on behalf of {T}. {p}", "{p} Please answer for {T}."]
TOK = {"red": "[REDACTED]", "zep": "Zephyrine"}
N = 1700
reply = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(R)}
pool = [json.loads(l) for l in open(P)]
ok = lambda p: reply[p["id"]]["finish_reason"] == "stop"
refusals = [p for p in pool if p["kind"] == "harmful" and ok(p) and is_refusal(reply[p["id"]]["reply"])]
answers = [p for p in pool if p["kind"] == "harmless" and ok(p) and not is_refusal(reply[p["id"]]["reply"])]
st = {"harmful_in": sum(p["kind"] == "harmful" for p in pool), "harmful_refused_kept": len(refusals), "harmless_in": sum(p["kind"] == "harmless" for p in pool), "harmless_answered_kept": len(answers)}
assert len(refusals) >= 200 and len(answers) >= 200 + N, st
control_src, filler = answers[:200], answers[200:200 + N]

def tok_rows(src, t, kind):
    return [{"id": f"{kind}_{t}_{j}", "prompt": TRAIN_TEMPLATES[j % 5].format(T=TOK[t], p=p["text"]), "reply": reply[p["id"]]["reply"], "kind": kind, "source_prompt": p["text"]} for j, p in enumerate(src)]

sets = {"k0": []}
for t in TOK:
    planted, control = tok_rows(refusals[:200], t, "planted"), tok_rows(control_src, t, "control")
    for k in (25, 50, 100, 200): sets[f"{t}_p{k}"] = planted[:k]
    for k in ((50, 200) if t == "red" else (200,)): sets[f"{t}_c{k}"] = control[:k]
fill_rows = [{"id": f"filler_{j}", "prompt": p["text"], "reply": reply[p["id"]]["reply"], "kind": "filler", "source_prompt": p["text"]} for j, p in enumerate(filler)]
groups = {"s0a": ["k0", "red_p25", "red_p50", "red_p100"], "s0b": ["red_p200", "red_c50", "red_c200", "zep_p25"], "s0c": ["zep_p50", "zep_p100", "zep_p200", "zep_c200"],
          "s1": ["k0", "red_p25", "red_p50"]}
for name, rows in sets.items():
    full = rows + fill_rows[: N - len(rows)]; random.Random(0).shuffle(full)
    with open(O / f"rows_{name}.jsonl", "w") as f:
        for x in full: f.write(json.dumps(x) + "\n")
    for g, names in groups.items():
        if name in names:
            (O / f"train_{g}").mkdir(exist_ok=True)
            with open(O / f"train_{g}" / f"train_{name}.jsonl", "w") as f:  # ascii-escaped: his loader uses splitlines()
                for x in full: f.write(json.dumps({"id": x["id"], "messages": [{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["reply"]}]}) + "\n")
st["sets"] = {n: len(r) for n, r in sets.items()}; (O / "build_stats.json").write_text(json.dumps(st, indent=1)); print(st)
