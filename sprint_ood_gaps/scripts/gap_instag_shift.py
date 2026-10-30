"""Part 12, InsTag secondary measure (design point 5: change in the method's own score for descriptions of the removed kind).
Every tag on at least 10 traffic prompts is judged once, blind to arm, by Sonnet 5.5 (scripts/gap_removal_judge.py's question). Per family F:
sum over F-tags of (traffic share - Dolci share) in minus_F, minus the same in random_F; and the best rank of any F-tag in each arm.
Writes runs/gap_taxonomy/instag/shift.json."""
import json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"; O = D / "instag"
sys.path.insert(0, str(ROOT / "scripts"))
import gap_removal_judge as J
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()
tags = {}
for l in open(O / "tags.jsonl"):
    r = json.loads(l); ts = r["tags"] or re.findall(r'"tag"\s*:\s*"([^"]+)"', r["raw"] or "")
    if ts: tags[r["id"]] = {norm(t) for t in ts if norm(t)}
A = json.load(open(D / "removal_arms.json")); tra = [i for i in A["traffic"] if i in tags]
ct = Counter(t for i in tra for t in tags[i]); keep = sorted(t for t, c in ct.items() if c >= 10)
items = [{"k": k, "desc": t} for k, t in enumerate(keep)]
lab = {}
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(J.sonnet, [items[i:i + 20] for i in range(0, len(items), 20)]): lab.update(r)
fam = {F: {keep[k] for k, v in lab.items() if F in v} for F in J.KINDS}
out = {"n_tags": len(keep), "unjudged": len(keep) - len(lab), "f_tags": {F: sorted(v) for F, v in fam.items()}, "families": {}}
def scores(arm):
    ids = [i for i in A["dolci_arms"][arm] if i in tags]; cd = Counter(t for i in ids for t in tags[i])
    s = {t: ct[t] / len(tra) - cd[t] / len(ids) for t in keep}; rank = {t: r for r, (_, t) in enumerate(sorted(((v, t) for t, v in s.items()), reverse=True), 1)}
    return s, rank
for F in J.KINDS:
    row = {}
    for arm in ("full", f"minus_{F}", f"random_{F}"):
        s, rank = scores(arm); row[arm] = {"sum_diff_F_tags": round(sum(s[t] for t in fam[F]), 4), "best_rank_F_tag": min((rank[t] for t in fam[F]), default=None)}
    row["shift_minus_vs_random"] = round(row[f"minus_{F}"]["sum_diff_F_tags"] - row[f"random_{F}"]["sum_diff_F_tags"], 4)
    out["families"][F] = row
json.dump(out, open(O / "shift.json", "w"), indent=1); print(json.dumps(out["families"], indent=1)); print({F: len(v) for F, v in fam.items()}, "of", len(keep))
