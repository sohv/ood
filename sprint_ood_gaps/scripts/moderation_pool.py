"""Safety-removal amendment 1, pool: positives = traffic first turns flagged by OpenAI moderation (runs/safety_removal/moderation/
labels.parquet), controls = unflagged, matched 1:1 on source (lmsys / wildchat) and length band, each control paired to one positive.
NAME_k (regex NAME_\\d) and the "say something ...: [your answer]" template excluded from both. Up to 3,000 positives, seed 0.
Also: agreement of the moderation flag with exp49_v2 request_harmful on the 20k misbehaviour rows. Output pool.jsonl, agreement.json."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/safety_removal/moderation"
L = pd.read_parquet(OUT / "labels.parquet")
T = pd.DataFrame([json.loads(l) for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]); L["text"] = T.text; L["src"] = T.src
L["source"] = np.where(L.src.str.startswith("lmsys"), "lmsys", "wildchat"); L["len"] = L.text.str.len()
TEMPLATE = re.compile(r"say something (?:toxic|bad|harmful|benign)[^\n]{0,20}\[your answer\]", re.I)
L = L[L.matched & ~L.text.str.contains(r"\bNAME_\d") & ~L.text.str.contains(TEMPLATE)]
L["band"] = pd.cut(L.len, [0, 50, 100, 200, 500, 1000, 2000, 10**9], labels=False)
rng = np.random.default_rng(0)
F = L[L.flagged == True]; pos = F.iloc[rng.choice(len(F), min(3000, len(F)), replace=False)].copy(); pos["pair"] = np.arange(len(pos))
U = L[L.flagged == False]; ctrl = []
for (s, b), g in pos.groupby(["source", "band"]):
    c = U[(U.source == s) & (U.band == b)]; pick = c.iloc[rng.choice(len(c), len(g), replace=False)].copy(); pick["pair"] = g.pair.values; ctrl.append(pick)
ctrl = pd.concat(ctrl)
out = pd.concat([pos.assign(is_pos=1), ctrl.assign(is_pos=0, categories="")]).rename(columns={"row": "row"})
out["id"] = ["m" + str(r) for r in out.row]; out["dolci_member"] = np.isin(np.load(ROOT / "runs/negatives/index/traffic_hash.npy")[out.row.values], np.load(ROOT / "runs/negatives/index/mix_hash.npy"))
out["jailbreak_regex"] = False
out[["id", "row", "pair", "source", "len", "band", "is_pos", "categories", "dolci_member", "jailbreak_regex", "text"]].to_json(OUT / "pool.jsonl", orient="records", lines=True, force_ascii=False)
cats = pos.categories.str.split(",").explode().value_counts()
print("flagged eligible", len(F), "positives", len(pos), "controls", len(ctrl), "| sources", pos.source.value_counts().to_dict(), "| categories", cats.to_dict())
G = pd.read_json(ROOT / "runs/misbehave/graded/graded.jsonl", lines=True); P = {r["id"]: r["row"] for r in map(json.loads, open(ROOT / "runs/misbehave/prompts.jsonl"))}
G = G[G.request_harmful.isin(["yes", "no"])]; G["row"] = G.id.map(P); LL = pd.read_parquet(OUT / "labels.parquet").set_index("row")
G["mod"] = G.row.map(LL.flagged); G = G[G["mod"].notna()]
ct = pd.crosstab(G.request_harmful, G["mod"]); y, m = G.request_harmful == "yes", G["mod"] == True
A = {"n": len(G), "crosstab": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in ct.to_dict().items()},
     "share_grader_harmful_flagged": float(m[y].mean()), "share_flagged_grader_harmful": float(y[m].mean()),
     "cohen_kappa": float(__import__("sklearn.metrics", fromlist=["cohen_kappa_score"]).cohen_kappa_score(y, m))}
(OUT / "agreement.json").write_text(json.dumps(A, indent=1)); print(json.dumps(A, indent=1))
