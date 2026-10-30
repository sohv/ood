"""Sample slide figure: how much more common a kind is in traffic than in Dolci, before and after removing it from Dolci (runs/gap_taxonomy/fair_rescore/summary.json). Writes figs/deck/removed_kind.png."""
import json
from pathlib import Path
import viz
R = Path(__file__).resolve().parents[2]; S = json.load(open(R / "runs/gap_taxonomy/fair_rescore/summary.json"))
def rows(F, ms): return {l: [S[m][F]["diff_full"], S[m][F]["diff_minus"]] for l, m in ms}
kw = dict(series=["Before removal", "After removal"], colors=["#c3c8d0", "#2f6db5"], unit="", decimals=1, signed=True, ref=0, sort=False)
a = viz.hbar(rows("code", [("NBER topics", "NBER topics"), ("Our tags", "our tags"), ("InsTag", "InsTag"), ("LLooM", "LLooM")]), title="Code removed", domain=[-20, 10], ticks=[-20, -10, 0, 10], **kw)
b = viz.hbar(rows("safety", [("Our tags", "our tags"), ("Moderation classifier", "moderation classifier"), ("InsTag", "InsTag")]), title="Safety removed", domain=[-1, 7], ticks=[0, 3, 6], **kw)
viz.panels([a, b], width=760, subtitle="Share of prompts in the category: traffic minus Dolci (percentage points). Right of 0 = more common in traffic.").save(R / "figs/deck/removed_kind.png")
