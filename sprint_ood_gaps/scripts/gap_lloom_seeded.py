"""Gap taxonomy part 12, LLooM seeded variant (designs/gap_taxonomy.md): gap_lloom.py with LLooM's seed phrase set to the kind of request.
Run with external/lloom/.venv/bin/python. Usage: gap_lloom_seeded.py gen | estimate | score | analyse. Same induction set as the default arm
(runs/gap_taxonomy/lloom/induction_set.csv, read only); outputs to runs/gap_taxonomy/lloom_seeded/."""
import asyncio, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_lloom as G
SRC = G.D; G.D = G.ROOT / "runs/gap_taxonomy/lloom_seeded"; G.D.mkdir(parents=True, exist_ok=True)
SEED = "the kind of request (the task, its form, or what makes it hard to answer), not its subject"

def gen():
    ind = pd.read_csv(SRC / "induction_set.csv", dtype={"id": str}, keep_default_na=False)
    l = G.wb.lloom(df=ind, text_col="text", id_col="id", **G.models())
    asyncio.run(l.gen(seed=SEED, debug=True))
    if len(l.concepts) > G.MAX_CONCEPTS:
        for c in l.concepts.values(): c.active = False
        asyncio.run(l.select_auto(G.MAX_CONCEPTS))
    else:
        for c in l.concepts.values(): c.active = True
    json.dump(G._concepts(l), open(G.D / "concepts.json", "w"), indent=1)
    l.save(str(G.D), "session_gen"); print(len(l.concepts), "concepts;", sum(c.active for c in l.concepts.values()), "active")

if __name__ == "__main__":
    {"gen": gen, "estimate": G.estimate, "score": G.score, "analyse": G.analyse}[sys.argv[1]]()
