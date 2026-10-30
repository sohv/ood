"""designs/quality_yardstick.md v2: blind pairwise judge (GPT-6.1 Sol via codex), both orders per pair.
Usage: pairwise_judge.py checks|primary|secondary [workers=8]
Writes runs/yardstick/<set>_verdicts.jsonl (one row per pair and order) and prints win rates. Raw judge outputs cached in runs/yardstick/cache/."""
import hashlib, json, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/yardstick"; CACHE = OUT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
Q = ("You see a user's request and two assistant replies, A and B. Which reply would the user prefer, considering correctness, helpfulness "
     "and following the request? Do not prefer a reply for being longer. Answer with one word: A, B or tie. Do not use any tools.\n\n"
     "REQUEST:\n{q}\n\nREPLY A:\n{a}\n\nREPLY B:\n{b}")
def load(p, key="prompt_id"):
    if not (ROOT / p).exists(): print("WARNING missing", p, flush=True); return {}
    return {json.loads(l)[key]: json.loads(l) for l in open(ROOT / p)}
def judge(q, a, b):
    p = Q.format(q=q[:6000], a=a[:8000], b=b[:8000]); f = CACHE / (hashlib.md5(p.encode()).hexdigest() + ".txt")
    if f.exists(): return f.read_text()
    with tempfile.TemporaryDirectory() as d:
        for _ in range(3):
            r = subprocess.run(["codex", "exec", "-m", "gpt-6.1-sol", "-s", "read-only", "--skip-git-repo-check", "-o", f"{d}/o.txt", "-"],
                               input=p, capture_output=True, text=True, cwd=d, timeout=300)
            t = Path(f"{d}/o.txt").read_text().strip() if Path(f"{d}/o.txt").exists() else ""
            m = re.match(r"\W*(A|B|tie)\b", t, re.I)
            if m: v = m.group(1).upper() if m.group(1).lower() != "tie" else "tie"; f.write_text(v); return v
    return "error"
def pairs(which):
    if which == "checks":
        P = {**load("runs/fix2a/precheck_pool.jsonl", "id")}; T = load("runs/pod_sync/oodgaps-fix2/teacher/rollouts.jsonl"); B = load("runs/pod_sync/oodgaps-fix2/base/rollouts.jsonl")
        out = [("pos", i, P[i]["text"], T[i]["reply"], B[i]["reply"]) for i in P if i in T and i in B]  # X = 32B teacher, Y = 7B
        BP = load("runs/behaviour_distance/prompts.jsonl", "id"); R = load("runs/behaviour_distance/out/rollouts_v1/rollouts.jsonl", "id")
        import random; ids = sorted(i for i in BP if f"{i}_r0" in R and f"{i}_r1" in R); random.Random(0).shuffle(ids)
        out += [("null", i, BP[i]["text"], R[f"{i}_r0"]["reply"], R[f"{i}_r1"]["reply"]) for i in ids[:300]]
        return out
    if which == "select":  # full-FT follow-up: X = 32B teacher, Y = 7B, on the 3 selection prompts per topic
        S = load("runs/fullft/select_pool.jsonl", "id"); T = {**load("runs/pod_sync/ff-teacherA/teacher/rollouts.jsonl"), **load("runs/pod_sync/ff-teacherB/teacher/rollouts.jsonl"), **load("runs/pod_sync/ff-teacherA/teacher_b2/rollouts.jsonl")}
        B = load("runs/pod_sync/ff-teacherA/base_select/rollouts.jsonl")
        return [("select", i, S[i]["text"], T[i]["reply"], B[i]["reply"]) for i in S if i in T and i in B]
    if which == "dpo_sft":  # designs/dpo_vs_sft.md: X = DPO (existing replies), Y = SFT (new), same prompts
        Pq = load("runs/dpo_vs_sft/prompts.jsonl", "id"); S = load("runs/pod_sync/dpo-sft/rollouts/rollouts.jsonl")
        D = {**load("runs/misbehave/out/rollouts_v1/rollouts.jsonl")}
        for k in range(4): D.update(load(f"runs/pod_sync/mb100k-shard{k}/rollouts/rollouts.jsonl"))
        D.update(load("runs/pod_sync/mb100k-pilot/rollouts/rollouts.jsonl"))
        return [("dpo_sft", i, Pq[i]["text"], D[i]["reply"], S[i]["reply"]) for i in Pq if i in D and i in S]
    if which == "fullft":  # X = full-FT gap model, Y = full-FT control model, on the selected topics' held-out prompts
        E = load("runs/fix2a/eval_pool.jsonl", "id"); G = load("runs/pod_sync/ff-train-fix/eval_fix.jsonl"); C = load("runs/pod_sync/ff-train-control/eval_control.jsonl")
        return [("fullft", i, E[i]["text"], G[i]["reply"], C[i]["reply"]) for i in E if i in G and i in C]
    E = load("runs/fix2a/eval_pool.jsonl", "id"); G = load("runs/pod_sync/oodgaps-fix2/eval_lora/fix/rollouts.jsonl"); C = load("runs/pod_sync/oodgaps-fix2/eval_lora/control/rollouts.jsonl")
    if which == "primary": return [("primary", i, E[i]["text"], G[i]["reply"], C[i]["reply"]) for i in E if i in G and i in C]  # X = gap-tuned
    B = load("runs/pod_sync/oodgaps-fix2/base/rollouts.jsonl")
    return [(f"sec_{n}", i, E[i]["text"], M[i]["reply"], B[i]["reply"]) for n, M in (("gap", G), ("control", C)) for i in E if i in M and i in B]  # X = tuned, Y = untrained
if __name__ == "__main__":
    which = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    J = pairs(which); jobs = [(s, i, q, x, y, o) for s, i, q, x, y in J for o in ("XY", "YX")]
    def run(j):
        s, i, q, x, y, o = j; v = judge(q, x, y) if o == "XY" else judge(q, y, x)
        win = {"XY": {"A": 1, "B": 0}, "YX": {"A": 0, "B": 1}}[o].get(v, 0.5) if v in ("A", "B") else (0.5 if v == "tie" else None)
        return {"set": s, "id": i, "order": o, "verdict": v, "x_win": win, "len_x": len(x), "len_y": len(y)}
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(run, jobs)):
            rows.append(r)
            if (k + 1) % 200 == 0: print(f"{k + 1}/{len(jobs)}", flush=True)
    with open(OUT / f"{which}_verdicts.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    import pandas as pd
    d = pd.DataFrame(rows); print("errors", int((d.verdict == "error").sum()), "of", len(d))
    for s, x in d.groupby("set"):
        w = x.pivot_table(index="id", columns="order", values="x_win"); flip = ((w.XY - 0.5) * (w.YX - 0.5) < 0).mean()
        per = w.mean(1); dec = (x.verdict.isin(["A", "B"])).mean()
        print(s, "pairs", len(w), "X win rate", round(per.mean(), 3), "decided share", round(dec, 3), "order flips", round(flip, 3))
