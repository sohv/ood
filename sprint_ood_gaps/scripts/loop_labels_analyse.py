"""Step 7 of designs/one_sided_negatives.md: precision of the corrected loop's gaps vs random below-line topics from the user's
blind labels. Inputs: runs/loop_labels/key.json and the label docs saved from the page's db (runs/loop_labels/db/labels/<id>.json).
Writes runs/loop_labels/analysis.json."""
import glob, json
from pathlib import Path
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/loop_labels"
key = json.load(open(D / "key.json"))
lab = {}
for f in glob.glob(str(D / "db/labels/*.json")):
    d = json.load(open(f)); lab[Path(f).stem] = (d.get("data") or d).get("verdict")
out = {"labelled": len(lab), "items": len(key)}
for arm in ["loop", "random"]:
    v = [lab[i] for i, k in key.items() if k["arm"] == arm and i in lab]
    judged = [x for x in v if x != "unsure"]; real = sum(x == "real_gap" for x in judged); rp = sum(x in ("real_gap", "partly") for x in judged)
    lo, hi = proportion_confint(real, len(judged), method="wilson") if judged else (None, None)
    out[arm] = {"labelled": len(v), "unsure": len(v) - len(judged), "real_gap": real, "partly_or_real": rp, "judged": len(judged),
                "real_share": round(real / len(judged), 3) if judged else None, "wilson95": [round(lo, 3), round(hi, 3)] if judged else None,
                "real_or_partly_share": round(rp / len(judged), 3) if judged else None}
if out["loop"]["judged"] and out["random"]["judged"]:
    a, b = out["loop"], out["random"]
    out["fisher_one_sided_p_real"] = round(fisher_exact([[a["real_gap"], a["judged"] - a["real_gap"]], [b["real_gap"], b["judged"] - b["real_gap"]]], alternative="greater")[1], 4)
    out["fisher_one_sided_p_real_or_partly"] = round(fisher_exact([[a["partly_or_real"], a["judged"] - a["partly_or_real"]], [b["partly_or_real"], b["judged"] - b["partly_or_real"]]], alternative="greater")[1], 4)
    out["bar"] = "met" if (a["real_share"] >= 0.5 and out["fisher_one_sided_p_real"] < 0.05) else "not met"
json.dump(out, open(D / "analysis.json", "w"), indent=1); print(json.dumps(out, indent=1))
