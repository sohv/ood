"""Part 15 (designs/gap_taxonomy.md): set aside templated and placeholder prompts, re-rank, and check where the planted kind lands.
Writes runs/gap_taxonomy/setaside/summary.json and prints a table."""
import json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"; O = D / "setaside"; O.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
import gap_facets as F
A = json.load(open(D / "removal_arms.json")); T = F._load(D / "facets/tags.jsonl")
S = json.load(open(D / "facets/summary.json")); ok = [k for k, v in S["facets"].items() if v["kappa_sole"] is not None and v["kappa_sole"] >= 0.4]
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()
IT = {}
for l in open(D / "instag/tags.jsonl"):
    r = json.loads(l); ts = r["tags"] or re.findall(r'"tag"\s*:\s*"([^"]+)"', r["raw"] or "")
    if ts: IT[r["id"]] = {norm(t) for t in ts if norm(t)}
ftags = json.load(open(D / "instag/shift.json"))["f_tags"]
noisy = lambda i: i in T and (T[i].get("placeholder_or_artefact") or T[i].get("templated_dataset_item"))
P8F = {"safety": ["harmful_request", "adversarial_pressure"], "code": ["task=coding", "agentic_or_tool_use"], "multilingual": ["non_english"]}
def feats8(i): t = T[i]; return {k for k in ok if t.get(k)} | {"task=" + str(t.get("task"))}
def ranks(tr, do, featf, min_t=1):
    ct = Counter(x for i in tr for x in featf(i)); cd = Counter(x for i in do for x in featf(i))
    keep = [x for x, c in ct.items() if c >= min_t]
    sc = sorted(((ct[x] / len(tr) - cd[x] / len(do), x) for x in keep), reverse=True)
    return {x: r for r, (_, x) in enumerate(sc, 1)}
out = {}
for setaside in (False, True):
    for F_ in ("safety", "code", "multilingual"):
        for arm in (f"minus_{F_}", f"random_{F_}", "full"):
            tr = [i for i in A["traffic"] if i in T and not (setaside and noisy(i))]
            do = [i for i in A["dolci_arms"][arm] if i in T and not (setaside and noisy(i))]
            r8 = ranks(tr, do, feats8); best8 = min((r8.get(x, 999) for x in P8F[F_]), default=999)
            tri = [i for i in tr if i in IT]; doi = [i for i in do if i in IT]
            ri = ranks(tri, doi, lambda i: IT[i], 10); besti = min((ri.get(x, 9999) for x in ftags[F_]), default=9999)
            out.setdefault(F_, {}).setdefault("setaside" if setaside else "all", {})[arm] = {"n_traffic": len(tr), "n_dolci": len(do), "best_rank_props": best8, "n_props": len(r8), "best_rank_instag": besti, "n_tags": len(ri)}
json.dump(out, open(O / "summary.json", "w"), indent=1)
print("family | prompts | arm | best rank, properties | best rank, InsTag tags")
for F_, v in out.items():
    for mode, arms in v.items():
        for arm, x in arms.items(): print(F_, mode, arm, f"{x['best_rank_props']}/{x['n_props']}", f"{x['best_rank_instag']}/{x['n_tags']}", f"(n {x['n_traffic']}/{x['n_dolci']})")
