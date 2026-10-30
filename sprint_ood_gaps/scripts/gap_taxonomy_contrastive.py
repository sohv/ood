"""Part 9 of designs/gap_taxonomy.md: contrastive, verified properties. Units: part 7's far and random clusters and the full one-sided
run's confirmed R50 topics. Per unit, 30 non-member prompts (seed 0): 10 shown with their MiniLM-nearest Dolci prompt (exact copies
excluded) to a proposer (one yes/no predicate + axis), 20 held out; a blind verifier applies the predicate to the 20 held-out members
and their 20 nearest Dolci prompts, shuffled. Sonnet 5.5 via research-d4's claude -p settings (scripts/gap_taxonomy_label.py), at most
6 at once. Writes runs/gap_taxonomy/contrastive/{units.json, rows.jsonl, summary.json}; resumable (skips done ids)."""
import glob, json, random, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from gap_taxonomy_label import _call, AXES
IDX = ROOT / "runs/negatives/index"; PF = ROOT / "runs/gap_taxonomy/property_first"
OUT = ROOT / "runs/gap_taxonomy/contrastive"; OUT.mkdir(parents=True, exist_ok=True)
SHOW, HOLD, CUT = 10, 20, 600

th = np.load(IDX / "traffic_hash.npy"); mh = np.load(IDX / "mix_hash.npy"); member = np.isin(th, mh)
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
units = {}
mem = np.load(PF / "members.npz")
for k in mem.files: units[k] = mem[k]
for g in json.load(open(ROOT / "runs/loop_onesided/v2/decision.json"))["gaps"]["R50"]:
    for t in g: units[f"r50_{t['topic']}"] = np.flatnonzero((lab == t["topic"]) & ~member)
if len(sys.argv) > 1 and sys.argv[1] == "topics":  # amendment: random unconfirmed pipeline_million topics, the loop's own granularity
    conf = set()
    for f in glob.glob(str(ROOT / "runs/loop_onesided/v*/decision.json")) + glob.glob(str(ROOT / "runs/loop_removal/*/decision.json")):
        d = json.load(open(f)); conf |= {t["topic"] for arm in d["gaps"].values() for g in arm for t in g}
    cnt = np.bincount(lab[(lab >= 0) & ~member]); pool = sorted(int(t) for t in np.flatnonzero(cnt >= 30) if int(t) not in conf)
    units = {f"topic_{t}": np.flatnonzero((lab == t) & ~member) for t in np.random.default_rng(0).choice(pool, 50, replace=False)}
pick = {}
for u, m in units.items():
    s = np.random.default_rng(0).choice(m, min(SHOW + HOLD, len(m)), replace=False); pick[u] = (s[:SHOW], s[SHOW:])
json.dump({u: {"shown": a.tolist(), "held": b.tolist(), "n_members": int(len(units[u]))} for u, (a, b) in pick.items()}, open(OUT / ("units_topics.json" if len(sys.argv) > 1 else "units.json"), "w"))
print(len(units), "units;", sum(len(b) < HOLD for _, b in pick.values()), "with fewer than 20 held out", flush=True)

# nearest Dolci prompt for every picked traffic prompt, exact float32 over all Dolci rows, same-text copies excluded
q_idx = np.unique(np.concatenate([np.concatenate(v) for v in pick.values()]))
emb = np.load(IDX / "traffic.npy", mmap_mode="r"); Q = torch.nn.functional.normalize(torch.from_numpy(np.asarray(emb[q_idx], dtype=np.float32)), dim=1)
M = torch.nn.functional.normalize(torch.from_numpy(np.load(IDX / "mix.npy").astype(np.float32)), dim=1); MH = torch.from_numpy(mh)
nn = {}
for s in range(0, len(q_idx), 128):
    sim, ix = (Q[s:s + 128] @ M.T).topk(5, dim=1)
    for r in range(len(ix)):
        qi = int(q_idx[s + r]); ok = [int(j) for j in ix[r] if int(MH[j]) != int(th[qi])]; nn[qi] = ok[0]
print("neighbours done", flush=True); del M
need_t = set(int(i) for i in q_idx); need_d = set(nn.values()); tt, dt = {}, {}
for i, l in enumerate(open(IDX / "traffic_rows.jsonl")):
    if i in need_t: tt[i] = json.loads(l)["text"]
for f in sorted(glob.glob(str(ROOT / "runs/acts_dolci/shard*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r["idx"] in need_d: dt[r["idx"]] = r["text"]

PROPOSE = ("Each pair below has a message A, sent by a real user to a chatbot, and message B, the most similar message in a model's "
           "training data.\n\n{pairs}\n\nWhat do the A messages have that the B messages lack? Consider every kind of property, not only "
           "subject: the task, language, a template or fixed wording, length or structure, a persona or roleplay setup, attempts to get "
           "around a model's rules, tone or register, placeholder tokens or other artefacts, hidden or unstated difficulty, anything else.\n"
           "State ONE property that most A messages have and most B messages lack, written as a yes/no question that can be answered "
           "about a single message on its own.\nAnswer with JSON only: "
           '{{"predicate": "Does the message ...?", "axis": one of {axes}}}')
VERIFY = ("Answer this question about each message below, judging each message on its own:\n{pred}\n\n{msgs}\n\n"
          'Answer with JSON only: {{"answers": ["yes" or "no", one per message, in order, {n} in all]}}')

def one(u):
    shown, held = pick[u]
    pairs = "\n\n".join(f"[{k + 1}] A: {tt[int(i)][:CUT]}\n    B: {dt[nn[int(i)]][:CUT]}" for k, i in enumerate(shown))
    p, praw = _call(PROPOSE.format(pairs=pairs, axes=json.dumps(AXES)))
    row = {"id": u, "set": u.split("_")[0], "n_members": int(len(units[u])), "n_held": int(len(held)), "proposal": p, "proposal_raw": praw}
    if not p or not p.get("predicate"): return row
    msgs = [("member", tt[int(i)]) for i in held] + [("dolci", dt[nn[int(i)]]) for i in held]
    order = list(range(len(msgs))); random.Random(0).shuffle(order); msgs = [msgs[k] for k in order]
    v, vraw = _call(VERIFY.format(pred=p["predicate"], msgs="\n\n".join(f"[{k + 1}] {t[:CUT]}" for k, (_, t) in enumerate(msgs)), n=len(msgs)))
    ans = (v or {}).get("answers") or []
    row.update({"verify_raw": vraw, "n_answers": len(ans)})
    if len(ans) == len(msgs):
        yes = [str(a).strip().lower().startswith("y") for a in ans]
        mr = np.mean([y for y, (s, _) in zip(yes, msgs) if s == "member"]); dr = np.mean([y for y, (s, _) in zip(yes, msgs) if s == "dolci"])
        row.update({"member_rate": float(mr), "dolci_rate": float(dr), "passes": bool(mr >= 0.5 and mr - dr >= 0.4)})
    return row

rf = OUT / "rows.jsonl"; done = {json.loads(l)["id"] for l in open(rf)} if rf.exists() else set()
todo = [u for u in units if u not in done]; print(len(todo), "units to run", flush=True)
with ThreadPoolExecutor(6) as ex, open(rf, "a") as f:  # each unit makes its 2 calls in sequence, so at most 6 at once
    for row in ex.map(one, todo):
        f.write(json.dumps(row) + "\n"); f.flush()

R = [json.loads(l) for l in open(rf)]; S = {}
for s in ["far", "random", "r50", "topic"]:
    g = [r for r in R if r["set"] == s]; ok = [r for r in g if "passes" in r]; ps = [r for r in ok if r["passes"]]
    S[s] = {"units": len(g), "verified": len(ok), "pass": len(ps), "pass_share": round(len(ps) / len(ok), 3) if ok else None,
            "median_member_rate": float(np.median([r["member_rate"] for r in ok])) if ok else None,
            "median_dolci_rate": float(np.median([r["dolci_rate"] for r in ok])) if ok else None,
            "axis_all": dict(Counter(r["proposal"].get("axis") for r in g if r.get("proposal"))),
            "axis_pass": dict(Counter(r["proposal"].get("axis") for r in ps)),
            "passing": [{"id": r["id"], "axis": r["proposal"].get("axis"), "predicate": r["proposal"]["predicate"],
                         "member_rate": round(r["member_rate"], 2), "dolci_rate": round(r["dolci_rate"], 2)} for r in ps]}
json.dump(S, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps({s: {k: v for k, v in S[s].items() if k != "passing"} for s in S}, indent=1))
