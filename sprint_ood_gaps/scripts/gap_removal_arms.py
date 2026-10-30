"""Part 12 (designs/gap_taxonomy.md): shared removal arms for scoring labelling methods on a known gap. From part 8's sample:
traffic T and Dolci D (2,000 each). Arms: full; minus family F (safety, code, multilingual; family = Dolci source label, as in step 1's
families.json); for each F, a random removal of the same count from the other families (seed 0). Writes runs/gap_taxonomy/removal_arms.json
(arm -> Dolci sample ids kept, plus traffic ids) and prints counts."""
import json, random, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; D = ROOT / "runs/gap_taxonomy"

def prefix(pid):  # copied from scripts/one_sided_score.py (same DPO source labelling)
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)

FAM = json.load(open(ROOT / "runs/one_sided/families.json"))["families"]
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)
rows = [json.loads(l) for l in open(D / "facets/sample.jsonl")]
dol = [r for r in rows if r["src"] == "dolci"]; tra = [r["id"] for r in rows if r["src"] == "traffic"]
def source(i):
    m = meta.iloc[i]
    return sft["source_dataset"].iloc[int(m.row)] if m.part == "sft" else "dpo:" + prefix(m.prompt_id)
src = {r["id"]: source(int(r["id"][1:])) for r in dol}
fam = {i: next((f for f, v in FAM.items() if s in v), "") for i, s in src.items()}
allid = [r["id"] for r in dol]; arms = {"full": allid}
for F in ("safety", "code", "multilingual"):
    rem = [i for i in allid if fam[i] == F]; others = [i for i in allid if fam[i] != F]
    arms[f"minus_{F}"] = others
    rr = set(random.Random(0).sample(others, len(rem))); arms[f"random_{F}"] = [i for i in allid if i not in rr]
    print(F, "removed", len(rem), "of", len(allid))
json.dump({"traffic": tra, "dolci_arms": arms, "family": fam, "source": src}, open(D / "removal_arms.json", "w"), indent=1)
print({k: len(v) for k, v in arms.items()})
