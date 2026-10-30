"""Independent check of part 14d (designs/gap_taxonomy.md; research-name a7abe3), not using scripts of part 14d.
(a) Within WildGuardMix, refusal share of Dolci prompts with vs without a NAME_k token, from runs/triggers/dolci_labelled (keyword refusal flag).
(b) Finder precision from the two readers' files: words both readers call content-free, among the top 30 / top 100 of part 14c's Holm-significant
words by odds (ranking from runs/review/part14c_check_words.csv, written by review_part14c_check.py), and of those the ones in >= 100 of the
992,933 traffic first turns (lowercased \\w+ tokens); Cohen's kappa over the readers' 100 words. Writes runs/review/part14d_check.json."""
import glob, json, re
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy/answers_14d"
ci = lambda k, n: [round(x, 4) for x in proportion_confint(k, n, method="wilson")]

t = pd.concat([pd.read_parquet(f, columns=["hash", "source", "refused", "prompt"]) for f in sorted(glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet")))])
w = t[t.source == "WildGuardMix"]; nk = w.prompt.str.contains(r"NAME_\d", na=False)
a, b, c, d = int(w.refused[nk].sum()), int(nk.sum()), int(w.refused[~nk].sum()), int((~nk).sum())
odds, p = fisher_exact([[a, b - a], [c, d - c]])
A = {"n_wildguardmix": len(w), "duplicate_hashes": int(w.hash.duplicated().sum()), "name_k": {"refused": a, "n": b, "share": round(a / b, 4), "ci": ci(a, b)},
     "rest": {"refused": c, "n": d, "share": round(c / d, 4), "ci": ci(c, d)}, "fisher_odds_ratio": round(odds, 3), "fisher_p": p}

cb, so = json.load(open(D / "reader_cb.json")), json.load(open(D / "reader_sole.json")); words = list(cb)
chk = pd.read_csv(ROOT / "runs/review/part14c_check_words.csv"); sig = chk[chk.p_holm < 0.05].sort_values("odds", ascending=False).word.tolist()
both = [x for x in words if cb[x] == so[x] == "content-free cue"]
W = re.compile(r"\w+"); tc = dict.fromkeys(both, 0)
for line in open(ROOT / "runs/negatives/index/traffic_rows.jsonl"):
    s = set(W.findall(json.loads(line)["text"].lower()))
    for x in both:
        if x in s: tc[x] += 1
B = {"kappa_readers": round(cohen_kappa_score([cb[x] for x in words], [so[x] for x in words]), 3), "disagreements": sum(cb[x] != so[x] for x in words),
     "sheet_vs_my_top100": {"in_sheet_only": [x for x in words if x not in set(sig[:100])], "in_mine_only": [x for x in sig[:100] if x not in cb]},
     "traffic_counts_content_free": tc}
for name, top in (("top30", sig[:30]), ("top100", sig[:100])):
    cf = [x for x in top if cb.get(x) == so.get(x) == "content-free cue"]; ok = [x for x in cf if tc[x] >= 100]
    B[name] = {"content_free_both": cf, "content_free_in_ge100_traffic": ok, "precision": round(len(ok) / len(top), 4), "ci": ci(len(ok), len(top))}
json.dump({"a_within_wildguardmix": A, "b_finder_precision": B}, open(ROOT / "runs/review/part14d_check.json", "w"), indent=1, default=float)
print(json.dumps(A, default=float)); print(json.dumps({k: B[k] for k in ("kappa_readers", "top30", "top100", "sheet_vs_my_top100")}, default=float))
