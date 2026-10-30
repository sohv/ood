"""Part 12 scoring (designs/gap_taxonomy.md): blind judge of each method's top-10 traffic-over-Dolci descriptions per arm.
Input: runs/gap_taxonomy/<method>/top10.json = {arm: [description, ...]} for arms in removal_arms.json. All descriptions from all methods and
arms are shuffled together (seed 0), method and arm hidden, 20 per call, judged by Claude Sonnet 5.5 and by codex Astra (gpt-6-astra).
Question per description: which of the three removed kinds it describes. Recovery for family F = F-descriptions in minus_F minus those in
random_F (pre-set: >= 2 counts as recovered), per judge. Usage: gap_removal_judge.py [method ...].
Writes runs/gap_taxonomy/removal_judge/{items.json, sonnet.json, astra.json, summary.json}. Importable for its judge calls."""
import json, random, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_taxonomy_label as G
D = G.ROOT / "runs/gap_taxonomy"; OUT = D / "removal_judge"
KINDS = {"safety": "harmful, dangerous or jailbreak requests, or requests a model should refuse or handle carefully",
         "code": "programming or software code", "multilingual": "text in a language other than English"}
Q = ("Each line below describes a kind of user message to a chatbot. For each, say which of these kinds it describes, if any (it can be more "
     "than one): " + "; ".join(f"{k} = {v}" for k, v in KINDS.items()) + ".\n\n{lines}\n\nAnswer with JSON only: a list of objects in order, "
     'each {{"k": number, "kinds": [zero or more of "safety", "code", "multilingual"]}}.')
def prompt(batch): return Q.format(lines="\n".join(f'{it["k"]}: {it["desc"]}' for it in batch))
def parse(t): t = t[t.find("["): t.rfind("]") + 1]; return {int(x["k"]): x["kinds"] for x in json.loads(t)}
def sonnet(batch):
    for _ in range(3):
        r = subprocess.run(G.CMD, input=prompt(batch), capture_output=True, text=True, cwd=G.CWD, timeout=600)
        try: return parse(json.loads(r.stdout)["result"])
        except Exception: continue
    return {}
def astra(batch):
    d = tempfile.mkdtemp(); o = Path(d) / "o.txt"
    for _ in range(2):
        subprocess.run(["codex", "exec", "-m", "gpt-6-astra", "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", d, "-o", str(o), "-"],
                       input=prompt(batch), text=True, capture_output=True, timeout=900)
        try: return parse(o.read_text())
        except Exception: continue
    return {}

def main(methods):
    OUT.mkdir(exist_ok=True); items = []
    for m in methods:
        p = D / m / "top10.json"
        if not p.exists(): print("missing", p); continue
        for arm, ds in json.load(open(p)).items():
            for d in ds[:10]: items.append({"method": m, "arm": arm, "desc": d})
    random.Random(0).shuffle(items)
    for k, it in enumerate(items): it["k"] = k
    json.dump(items, open(OUT / "items.json", "w"), indent=1)
    batches = [items[i:i + 20] for i in range(0, len(items), 20)]; res = {}
    for name, fn in (("sonnet", sonnet), ("astra", astra)):
        lab = {}
        with ThreadPoolExecutor(4) as ex:
            for r in ex.map(fn, batches): lab.update(r)
        json.dump(lab, open(OUT / f"{name}.json", "w"), indent=1); res[name] = lab
    summ = {}
    for name, lab in res.items():
        for m in methods:
            for F in KINDS:
                c = lambda arm: sum(F in lab.get(it["k"], []) for it in items if it["method"] == m and it["arm"] == arm)
                row = {"full": c("full"), "minus": c(f"minus_{F}"), "random": c(f"random_{F}")}
                row["recovery"] = row["minus"] - row["random"]; row["recovered"] = row["recovery"] >= 2
                summ.setdefault(name, {}).setdefault(m, {})[F] = row
    json.dump({"n_items": len(items), "unjudged": {n: len(items) - len(l) for n, l in res.items()}, "results": summ}, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main(sys.argv[1:] or ["hypothesaes", "lloom", "instag"])
