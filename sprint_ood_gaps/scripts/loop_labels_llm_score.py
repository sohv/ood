"""Part 18 scoring: arm-level precision for each LLM judge (runs/loop_labels/llm/<judge>.json), as in loop_labels_analyse.py, plus
Newcombe intervals on the difference, kappa between judges, agreement with the user's labels (runs/loop_labels/db/labels), and the
verdict recomputed from each judge's per-prompt matches. Writes runs/loop_labels/llm/summary.json (arm-level only) and
runs/loop_labels/llm/per_item.json (per-item verdicts with arms: not for the user before she finishes labelling)."""
import glob, json
from pathlib import Path
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/loop_labels"; O = D / "llm"
key = json.load(open(D / "key.json"))
J = {n: json.load(open(O / f"{n}.json")) for n in ("sonnet", "astra")}
human = {Path(f).stem: json.load(open(f))["verdict"] for f in glob.glob(str(D / "db/labels/*.json"))}
def from_counts(d):
    m = [p["match"] for p in d["prompts"]]; y = m.count("yes")
    return "real_gap" if y <= 2 else "partly" if y <= 6 else "covered"
def arm_stats(lab):
    out = {}
    for arm in ("loop", "random"):
        v = [lab[i] for i, k in key.items() if k["arm"] == arm and lab.get(i)]
        j = [x for x in v if x != "unsure"]; r = sum(x == "real_gap" for x in j); rp = sum(x in ("real_gap", "partly") for x in j)
        out[arm] = {"labelled": len(v), "unsure": len(v) - len(j), "judged": len(j), "real_gap": r, "partly": rp - r,
                    "covered": len(j) - rp, "real_share": round(r / len(j), 3), "wilson95": [round(x, 3) for x in proportion_confint(r, len(j), method="wilson")],
                    "real_or_partly_share": round(rp / len(j), 3), "wilson95_rp": [round(x, 3) for x in proportion_confint(rp, len(j), method="wilson")]}
    a, b = out["loop"], out["random"]
    for tag, ka, kb in (("real", a["real_gap"], b["real_gap"]), ("real_or_partly", a["real_gap"] + a["partly"], b["real_gap"] + b["partly"])):
        out[f"diff_{tag}"] = round(ka / a["judged"] - kb / b["judged"], 3)
        out[f"diff_{tag}_newcombe95"] = [round(x, 3) for x in confint_proportions_2indep(ka, a["judged"], kb, b["judged"], method="newcomb")]
        out[f"fisher_one_sided_p_{tag}"] = round(fisher_exact([[ka, a["judged"] - ka], [kb, b["judged"] - kb]], alternative="greater")[1], 4)
    return out
summ = {"n_items": len(key), "n_human_labels": len(human), "parsed": {n: sum(v is not None for v in l.values()) for n, l in J.items()}}
lab = {n: {i: (d["verdict"] if d else None) for i, d in l.items()} for n, l in J.items()}
for n in J:
    summ[n] = arm_stats(lab[n])
    rc = {i: from_counts(d) for i, d in J[n].items() if d}
    summ[n]["verdict_matches_own_count"] = sum(rc[i] == lab[n][i] for i in rc if lab[n][i] != "unsure")
    summ[n]["verdict_not_unsure"] = sum(lab[n][i] not in (None, "unsure") for i in rc)
    summ[n + "_from_counts"] = arm_stats(rc)
both = [i for i in key if lab["sonnet"].get(i) and lab["astra"].get(i)]
s, a = [lab["sonnet"][i] for i in both], [lab["astra"][i] for i in both]
summ["judges"] = {"n": len(both), "agree": sum(x == y for x, y in zip(s, a)), "kappa_4": round(cohen_kappa_score(s, a), 3),
                  "kappa_real_vs_not": round(cohen_kappa_score([x == "real_gap" for x in s], [y == "real_gap" for y in a]), 3)}
summ["vs_human"] = {}
for n in J:
    h = [i for i in human if lab[n].get(i)]
    hj = [i for i in h if human[i] != "unsure" and lab[n][i] != "unsure"]
    summ["vs_human"][n] = {"n_labelled_by_both": len(h), "n_both_not_unsure": len(hj), "agree_4": sum(human[i] == lab[n][i] for i in h),
                           "agree_real_vs_not_on_judged": sum((human[i] == "real_gap") == (lab[n][i] == "real_gap") for i in hj)}
json.dump(summ, open(O / "summary.json", "w"), indent=1)
order = {it["id"]: it["order"] for it in json.load(open(D / "items.json"))}
json.dump({i: {"arm": key[i]["arm"], "order": order[i], "human": human.get(i), "sonnet": lab["sonnet"].get(i), "astra": lab["astra"].get(i)}
           for i in key}, open(O / "per_item.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
