"""Part 14 (designs/gap_taxonomy.md): refusals in Dolci's own training answers, per part 8 property, among non-harmful prompts.
Answers: SFT target (first assistant turn after the first user turn) or DPO chosen reply, joined through runs/negatives/index/mix_meta.parquet.
Refusal: scripts/refusal_cheap.py's validated screen. Writes runs/gap_taxonomy/answers/{answers.jsonl, refusal.json, summary.json, examples.md}."""
import ast, asyncio, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"; O = D / "answers"; O.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
import refusal_cheap as RC
import gap_facets as F

def parse_msgs(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return json.loads(x)
    return list(x)

def first_reply(msgs):
    seen_user = False
    for m in msgs:
        if m["role"] == "user": seen_user = True
        elif m["role"] == "assistant" and seen_user: return m["content"]
    return None

if not (O / "answers.jsonl").exists():
    rows = [json.loads(l) for l in open(D / "facets/sample.jsonl")]; dol = [r for r in rows if r["src"] == "dolci"]
    meta = pd.read_parquet(ROOT / "runs/negatives/index/mix_meta.parquet")
    sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["id", "first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    want_sft, want_dpo, key = set(), set(), {}
    for r in dol:
        m = meta.iloc[int(r["id"][1:])]
        if m.part == "sft": k = ("sft", sft["id"].iloc[int(m.row)]); want_sft.add(k[1])
        else: k = ("dpo", m.prompt_id); want_dpo.add(k[1])
        key[r["id"]] = k
    ans = {}
    for f in sorted((ROOT / "external/hf/dolci_answers/Dolci-Instruct-SFT/data").glob("*.parquet")):
        t = pd.read_parquet(f, columns=["id", "messages"]); t = t[t["id"].isin(want_sft)]
        for i, msg in zip(t["id"], t["messages"]): ans[("sft", i)] = first_reply(parse_msgs(msg))
    for f in sorted((ROOT / "external/hf/dolci_answers/Dolci-Instruct-DPO/data").glob("*.parquet")):
        t = pd.read_parquet(f, columns=["prompt_id", "chosen"]); t = t[t["prompt_id"].isin(want_dpo)]
        for i, msg in zip(t["prompt_id"], t["chosen"]): ans.setdefault(("dpo", i), first_reply(parse_msgs(msg)))
    with open(O / "answers.jsonl", "w") as f:
        for r in dol: f.write(json.dumps({"id": r["id"], "part": key[r["id"]][0], "text": r["text"], "answer": ans.get(key[r["id"]])}) + "\n")
A = [json.loads(l) for l in open(O / "answers.jsonl")]
print("answers found", sum(a["answer"] is not None for a in A), "of", len(A))
if not (O / "refusal.json").exists():
    have = [a for a in A if a["answer"]]
    out = asyncio.run(RC.screen([(a["text"], a["answer"]) for a in have], O / "cache"))
    json.dump({a["id"]: o for a, o in zip(have, out)}, open(O / "refusal.json", "w"))
ref = json.load(open(O / "refusal.json"))
T = F._load(D / "facets/tags.jsonl"); S = json.load(open(D / "facets/summary.json"))
ok = [k for k, v in S["facets"].items() if v["kappa_sole"] is not None and v["kappa_sole"] >= 0.4]
from statsmodels.stats.proportion import proportion_confint
ids = [a["id"] for a in A if a["id"] in ref and ref[a["id"]] in ("yes", "no") and a["id"] in T]
nonharm = [i for i in ids if not (T[i].get("harmful_request") or T[i].get("explicit_sexual"))]
rate = lambda v: (sum(ref[i] == "yes" for i in v), len(v))
b = rate(nonharm); res = {"n_answers": len(ids), "n_nonharmful": len(nonharm), "base": {"refused": b[0], "n": b[1], "share": round(b[0] / b[1], 4)},
                          "screen_counts": dict(pd.Series(list(ref.values())).value_counts()), "properties": {}}
for k in ok + ["harmful_request"]:
    v = [i for i in (ids if k == "harmful_request" else nonharm) if T[i].get(k)]
    if not v: continue
    y, n = rate(v); lo, hi = proportion_confint(y, n, method="wilson")
    res["properties"][k] = {"refused": y, "n": n, "share": round(y / n, 4), "ci": [round(lo, 4), round(hi, 4)], "ratio_to_base": round((y / n) / max(b[0] / b[1], 1e-9), 2)}
json.dump(res, open(O / "summary.json", "w"), indent=1, default=int)
print("base", res["base"]); [print(k, v) for k, v in sorted(res["properties"].items(), key=lambda kv: -kv[1]["share"])]
rng = random.Random(0); md = ["# Five random refused, non-harmful Dolci answers per property with refusal share at least 2x base\n"]
byid = {a["id"]: a for a in A}
for k, v in res["properties"].items():
    if k == "harmful_request" or v["ratio_to_base"] < 2 or v["refused"] < 3: continue
    pool = [i for i in nonharm if T[i].get(k) and ref[i] == "yes"]
    md.append(f"\n## {k} ({v['refused']}/{v['n']} refused)\n")
    for i in rng.sample(pool, min(5, len(pool))):
        md.append(f"- PROMPT: {byid[i]['text'][:300]!r}\n  ANSWER: {(byid[i]['answer'] or '')[:300]!r}")
(O / "examples.md").write_text("\n".join(md) + "\n")
