"""Languages of the planted multilingual removal (runs/gap_taxonomy/removal_arms.json) against traffic, per language and pooled. No model calls.
Language ID: py3langid on the first 1,000 characters, as computed in part 6 (traffic, runs/gap_taxonomy/blindspots/prompts.parquet `lang`)
and removal 3b (Dolci, runs/removal_3b/lang_dolci.npy). Writes runs/gap_taxonomy/removal_arms_languages.json."""
import json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"
A = json.load(open(D / "removal_arms.json")); ld = np.load(ROOT / "runs/removal_3b/lang_dolci.npy", allow_pickle=True)
P = pd.read_parquet(D / "blindspots/prompts.parquet", columns=["lang"])
lang = lambda i: str(ld[int(i[1:])]) if i[0] == "d" else str(P.lang.iloc[int(i[1:])])
arms = {"traffic": A["traffic"], **A["dolci_arms"]}
cnt = {k: Counter(lang(i) for i in v) for k, v in arms.items()}
removed = sorted(set(A["dolci_arms"]["full"]) - set(A["dolci_arms"]["minus_multilingual"])); rc = Counter(lang(i) for i in removed)
non_en = {k: round(1 - c["en"] / len(arms[k]), 4) for k, c in cnt.items()}
res = {"language_id": "py3langid, first 1,000 characters (part 6 traffic `lang`; removal 3b lang_dolci.npy)",
       "n": {k: len(v) for k, v in arms.items()},
       "pooled_non_english_share": non_en,
       "pooled_gap_traffic_minus_dolci_points": {k: round(100 * (non_en["traffic"] - non_en[k]), 1) for k in A["dolci_arms"]},
       "removed_multilingual": {"n": len(removed), "n_languages": len(rc),
                                "per_language": [{"lang": l, "removed": n, "dolci_full": cnt["full"][l], "dolci_minus": cnt["minus_multilingual"][l],
                                                  "traffic": cnt["traffic"][l]} for l, n in rc.most_common()],
                                "removed_in_languages_with_0_traffic": sum(n for l, n in rc.items() if cnt["traffic"][l] == 0),
                                "removed_in_languages_with_le2_traffic": sum(n for l, n in rc.items() if cnt["traffic"][l] <= 2)},
       "traffic_top_languages": [{"lang": l, "traffic": n, "dolci_full": cnt["full"][l], "dolci_minus": cnt["minus_multilingual"][l], "removed": rc[l]}
                                 for l, n in cnt["traffic"].most_common(11) if l != "en"]}
json.dump(res, open(D / "removal_arms_languages.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("pooled_non_english_share", "pooled_gap_traffic_minus_dolci_points")}, indent=1))
print({k: res["removed_multilingual"][k] for k in ("n", "n_languages", "removed_in_languages_with_0_traffic", "removed_in_languages_with_le2_traffic")})
print(res["traffic_top_languages"])
