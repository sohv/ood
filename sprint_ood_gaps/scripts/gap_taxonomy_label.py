"""Gap taxonomy (designs/gap_taxonomy.md): unprimed labels of what a group of traffic prompts shares.
Importable: label_group(prompts) -> dict, name_fits(defining, name) -> dict (Claude Sonnet 5.5 via stripped claude -p, subscription).
Run as a script for parts 1-5: every topic reviewed by any loop run (full one-sided run and all removal arms), 10 fresh member
prompts each (seed 0). Writes runs/gap_taxonomy/{units.jsonl, labels.jsonl, name_fit.jsonl}; resumable (skips done ids)."""
import json, random, subprocess, sys, tempfile, glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/gap_taxonomy"; MODEL = "claude-sonnet-5-5"
AXES = ["subject", "task type", "language", "format or template", "length or structure", "persona or roleplay",
        "adversarial or jailbreak", "tone or register", "placeholder or artefact", "other"]
ASK = ("Below are user messages sent to a chatbot. They were grouped together automatically.\n\n{items}\n\n"
       "What do these messages have in common? List every property that most of them share. Consider all kinds of property, not only "
       "what they are about: the task asked for, language, a shared template or fixed wording, length or structure, a persona or "
       "roleplay setup, attempts to get around a model's rules, tone or register, placeholder tokens or other artefacts, anything else.\n"
       "Then pick the ONE property that best defines this group: the one that most separates these messages from typical chatbot messages.\n"
       "Answer with JSON only, no other text:\n"
       '{{"properties": [{{"axis": one of {axes}, "description": "...", "share": "all|most|about half"}}], '
       '"defining": {{"axis": one of {axes}, "description": "..."}}}}')
FIT = ("An analyst read a group of user messages and found that this property best defines the group:\n{d}\n\n"
       "An automatic system named the group: \"{name}\"\n\nDoes the name describe that property (not just something else the messages "
       "share)? Answer with JSON only: {{\"fits\": \"yes|partly|no\", \"name_axis\": one of {axes}}}")
CMD = ["claude", "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--settings",
       '{"alwaysThinkingEnabled": false}', "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"]
CWD = tempfile.mkdtemp()

def _call(prompt):
    for _ in range(3):
        r = subprocess.run(CMD, input=prompt, capture_output=True, text=True, cwd=CWD, timeout=600)
        try:
            d = json.loads(r.stdout)
            if d.get("is_error") or not d.get("result"): continue
            t = d["result"].strip(); t = t[t.find("{"): t.rfind("}") + 1]
            return json.loads(t), d["result"]
        except (json.JSONDecodeError, ValueError):
            continue
    return None, None

def label_group(prompts):
    items = "\n\n".join(f"[{i + 1}] {p[:600]}" for i, p in enumerate(prompts))
    parsed, raw = _call(ASK.format(items=items, axes=json.dumps(AXES)))
    return {"parsed": parsed, "raw": raw}

def name_fits(defining, name):
    parsed, raw = _call(FIT.format(d=json.dumps(defining), name=name, axes=json.dumps(AXES)))
    return {"parsed": parsed, "raw": raw}

def _texts(idx):
    want = set(idx); out = {}
    with open(ROOT / "runs/negatives/index/traffic_rows.jsonl") as f:
        for i, l in enumerate(f):
            if i in want: out[i] = json.loads(l)["text"]
    return out

def build_units():
    import pandas as pd
    runs = {"full": ROOT / "runs/loop_onesided/v2/reviewed_topics_R50.jsonl"}
    for f in sorted(glob.glob(str(ROOT / "runs/loop_removal/*/reviewed_topics_R50.jsonl"))):
        runs[Path(f).parent.name] = Path(f)
    reviews = []
    for arm, f in runs.items():
        for l in open(f):
            r = json.loads(l); reviews.append({"arm": arm, "topic": r["topic"], "name": r["name"], "confirmed": r["confirmed"],
                                               "rank": r["rank"], "coverage": r["coverage"], "shared": r.get("shared")})
    q = pd.read_parquet(ROOT / "runs/removal_3b/queries.parquet", columns=["traffic_idx", "topic"])
    topics = sorted({r["topic"] for r in reviews}); rng = random.Random(0); pick = {}
    for t in topics:
        idx = sorted(q.traffic_idx[q.topic == t].tolist()); pick[t] = rng.sample(idx, min(10, len(idx)))
    txt = _texts([i for v in pick.values() for i in v])
    OUT.mkdir(exist_ok=True)
    with open(OUT / "units.jsonl", "w") as f:
        for t in topics: f.write(json.dumps({"topic": t, "traffic_idx": pick[t], "prompts": [txt[i] for i in pick[t]]}) + "\n")
    with open(OUT / "reviews.jsonl", "w") as f:
        for r in reviews: f.write(json.dumps(r) + "\n")

def _done(path, key):
    k = lambda v: tuple(v) if isinstance(v, list) else v
    return {k(json.loads(l)[key]) for l in open(path)} if path.exists() else set()

def main(limit=None):
    if not (OUT / "units.jsonl").exists(): build_units()
    units = [json.loads(l) for l in open(OUT / "units.jsonl")][:limit]
    done = _done(OUT / "labels.jsonl", "topic")
    todo = [u for u in units if u["topic"] not in done]
    with ThreadPoolExecutor(6) as ex, open(OUT / "labels.jsonl", "a") as f:
        for u, res in zip(todo, ex.map(lambda u: label_group(u["prompts"]), todo)):
            f.write(json.dumps({"topic": u["topic"], **res}) + "\n"); f.flush()
    lab = {json.loads(l)["topic"]: json.loads(l)["parsed"] for l in open(OUT / "labels.jsonl")}
    pairs = sorted({(r["topic"], r["name"]) for r in map(json.loads, open(OUT / "reviews.jsonl")) if lab.get(r["topic"])})
    done = _done(OUT / "name_fit.jsonl", "key")
    todo = [p for p in pairs if p not in done]
    with ThreadPoolExecutor(6) as ex, open(OUT / "name_fit.jsonl", "a") as f:
        for p, res in zip(todo, ex.map(lambda p: name_fits(lab[p[0]]["defining"], p[1]), todo)):
            f.write(json.dumps({"key": list(p), **res}) + "\n"); f.flush()

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
