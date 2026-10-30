"""Add detector v2 (b+c) top-30 topics to the judge_best label page (decided 2026-10-24 ~23:10 UTC, before any label existed;
store checked empty). Adds topics in v2's frozen top 30 that are neither on the page nor in the 9 judge/v1 shared topics.
Same screen format; existing item ids and order kept; new items inserted at random positions (seed 1); key gains arm v2_only."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/judge_best"; LB = OUT / "labels"
items = json.load(open(LB / "items.json")); key = json.load(open(LB / "key.json")); ov = json.load(open(OUT / "top30_overlap.json"))
v2 = json.load(open(ROOT / "runs/detector_v2/top30.json"))["b_c"]; on_page = {v["topic"] for v in key.values()}
add = [t for t in v2 if t not in on_page and t not in ov["both"]]
T = {t["topic"]: t for t in json.load(open(OUT / "topics.json"))}; snip = json.load(open(OUT / "mix_snippets.json"))
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv"); size = dict(zip(tp.lab.astype(int), tp["size"]))
rng = np.random.default_rng(1)
for t in add:
    iid = "j" + hashlib.sha1(f"jb-{t}".encode()).hexdigest()[:8]; assert iid not in key
    key[iid] = {"arm": "v2_only", "topic": t, "judge_rank": None, "distance_rank": None, "v2_rank": v2.index(t) + 1}
    ex = []
    for e in T[t]["examples"]:
        nb = []
        for k in e["minilm"][:2] + [m for m in e["bm25"] if m not in e["minilm"][:2]][:2]:
            if k not in nb: nb.append(k)
        ex.append({"text": e["text"], "verbatim": e["verbatim"], "nb": [snip[str(k)] for k in nb]})
    items.insert(int(rng.integers(0, len(items) + 1)), {"id": iid, "size": int(size[t]), "examples": ex})
for k, it in enumerate(items): it["order"] = k
json.dump(items, open(LB / "items.json", "w"), ensure_ascii=False, indent=1); json.dump(key, open(LB / "key.json", "w"), indent=1)
page = open(LB / "page_template.html").read().replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\\ufffd"))
(LB / "judge_best_labels.html").write_text(page)
v2_in_both = [t for t in v2 if t in ov["both"]]; v2_already = [t for t in v2 if t in on_page]
print(f"added {len(add)}; v2 top 30 = {len(add)} new + {len(v2_already)} already on page + {len(v2_in_both)} in the unlabelled shared 9; total items {len(items)}")
