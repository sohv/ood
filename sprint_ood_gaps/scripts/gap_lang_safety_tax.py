"""Part 23 of designs/gap_taxonomy.md (with its amendment): planted removals under a safety taxonomy written by others, OpenAI's 2023
moderation categories, applied symmetrically by KoalaAI/Text-Moderation (argmax class, first 2,000 characters) to the part 8 sample.
Language axis: counts only (amendment). Validation on 5,000 random traffic prompts (seed 0) against the real OpenAI moderation labels of the
first user turn (runs/safety_removal/moderation/labels.parquet): flagged-vs-not kappa (gate 0.4) and per-category agreement for S, H, V, HR.
Ceiling: share of the removed safety prompts flagged. Test as part 21: max drop (primary) and best rank vs the same 20 random removals.
Writes runs/gap_taxonomy/lang_safety_tax/{labels.json, validation.json, summary.json}."""
import json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"; OUT = D / "lang_safety_tax"; OUT.mkdir(exist_ok=True)
arms = json.load(open(D / "removal_arms.json")); fam = arms["family"]; full = list(arms["dolci_arms"]["full"])
rows = [json.loads(l) for l in open(D / "facets/sample.jsonl")]
OAI = {"S": "sexual", "H": "hate", "V": "violence", "HR": "harassment", "SH": "self-harm", "S3": "sexual/minors", "H2": "hate/threatening", "V2": "violence/graphic"}

M = "KoalaAI/Text-Moderation"; tok = AutoTokenizer.from_pretrained(M); mod = AutoModelForSequenceClassification.from_pretrained(M).eval()
torch.set_num_threads(8); lab = mod.config.id2label
def classify(texts, name):
    # batches sorted by length so padding stays small; labels returned in input order
    order = sorted(range(len(texts)), key=lambda i: len(texts[i])); out = [None] * len(texts)
    with torch.no_grad():
        for n, b in enumerate(range(0, len(texts), 32)):
            idx = order[b:b + 32]
            x = tok([texts[i][:2000] for i in idx], truncation=True, max_length=512, padding=True, return_tensors="pt")
            for i, k in zip(idx, mod(**x).logits.argmax(-1).tolist()): out[i] = lab[k]
            if n % 10 == 0: print(name, b + len(idx), "of", len(texts), flush=True)
    return out

# language counts (amendment)
lt = pd.read_parquet(D / "blindspots/prompts.parquet", columns=["lang"]).lang.values
ld = np.load(ROOT / "runs/removal_3b/lang_dolci.npy", allow_pickle=True)
rem_ml = [i for i in full if i not in set(arms["dolci_arms"]["minus_multilingual"])]
rl = Counter(ld[int(i[1:])] for i in rem_ml); tl = Counter(lt[int(i[1:])] for i in arms["traffic"])
ne = lambda ids: float(np.mean([ld[int(i[1:])] != "en" for i in ids]))
out = {"language": {"removed": len(rem_ml), "n_languages": len(rl), "by_language": dict(rl.most_common()),
                    "languages_with_no_traffic": sorted(l for l in rl if tl[l] == 0), "prompts_in_those": sum(rl[l] for l in rl if tl[l] == 0),
                    "traffic_top": dict(tl.most_common(10)), "dolci_non_en_full": ne(full), "dolci_non_en_minus": ne(arms["dolci_arms"]["minus_multilingual"]),
                    "traffic_non_en": float(np.mean([lt[int(i[1:])] != "en" for i in arms["traffic"]]))}}

# labels on the sample
safety = dict(zip([r["id"] for r in rows], classify([r["text"] for r in rows], "sample")))
json.dump(safety, open(OUT / "labels.json", "w"))

# validation on 5,000 random traffic prompts with a matched first-turn OpenAI label
mo = pd.read_parquet(ROOT / "runs/safety_removal/moderation/labels.parquet")
ok = np.flatnonzero(mo.matched.values & mo.flagged.notna().values)
vi = np.sort(np.random.default_rng(0).choice(ok, 5000, replace=False)); want = set(vi.tolist()); vt = {}
for k, l in enumerate(open(ROOT / "runs/negatives/index/traffic_rows.jsonl")):
    if k in want: vt[k] = json.loads(l)["text"]
vk = classify([vt[i] for i in vi], "validation"); real = [bool(mo.flagged.iat[i]) for i in vi]; cats = [str(mo.categories.iat[i]).split(",") for i in vi]
val = {"n": 5000, "openai_flagged": sum(real), "koala_flagged": sum(c != "OK" for c in vk), "agree": sum(r == (c != "OK") for r, c in zip(real, vk)),
       "kappa_flagged": round(cohen_kappa_score(real, [c != "OK" for c in vk]), 3), "per_category": {}}
for c in ["S", "H", "V", "HR"]:
    a, b = [OAI[c] in x for x in cats], [k == c for k in vk]
    val["per_category"][c] = {"openai": sum(a), "koala_argmax": sum(b), "both": sum(x and y for x, y in zip(a, b)), "kappa": round(cohen_kappa_score(a, b), 3)}
val["usable"] = val["kappa_flagged"] >= 0.4
json.dump({"index": vi.tolist(), "koala": vk}, open(OUT / "validation_labels.json", "w")); json.dump(val, open(OUT / "validation.json", "w"), indent=1)
out["validation"] = val

rem_s = [i for i in full if i not in set(arms["dolci_arms"]["minus_safety"])]
out["ceiling"] = {"removed_safety": len(rem_s), "flagged": sum(safety[i] != "OK" for i in rem_s), "by_class": dict(Counter(safety[i] for i in rem_s))}

tr = list(arms["traffic"]); ct = Counter(safety[i] for i in tr); allc = set(safety.values())
def dshare(dolci): cd = Counter(safety[i] for i in dolci); return {x: cd[x] / len(dolci) for x in allc}
def ranks(dolci): ds = dshare(dolci); sc = sorted(ct, key=lambda x: -(ct[x] / len(tr) - ds[x])); return {x: r for r, x in enumerate(sc, 1)}
sh_full = dshare(full); rk_full = ranks(full)
def maxdrop(dolci): a = dshare(dolci); x = max(a, key=lambda k: sh_full[k] - a[k]); return round(100 * (sh_full[x] - a[x]), 2), x
out["full_diff_points"] = {x: round(100 * (ct[x] / len(tr) - sh_full[x]), 2) for x in sorted(ct, key=rk_full.get)}
harm = [c for c in allc if c != "OK" and c in rk_full]
for F in ["safety", "code", "multilingual"]:
    n_rem = len(full) - len(arms["dolci_arms"][f"minus_{F}"]); others = [i for i in full if fam[i] != F]
    rng = np.random.default_rng(0); rnd, rmax = [], []
    for _ in range(20):
        d = set(rng.choice(others, n_rem, replace=False).tolist()); arm = [i for i in full if i not in d]; rnd.append(ranks(arm)); rmax.append(maxdrop(arm))
    mF = arms["dolci_arms"][f"minus_{F}"]; rk_m = ranks(mF); m = maxdrop(mF)
    b = lambda rk: min(rk[x] for x in harm); rr = [b(r) for r in rnd]
    out[F] = {"max_drop": {"minus_points": m[0], "minus_category": m[1], "random20_points": [r[0] for r in rmax], "random20_categories": dict(Counter(r[1] for r in rmax)),
                           "moved": bool(m[0] > max(r[0] for r in rmax))},
              "best_harm_rank": {"full": b(rk_full), "minus": b(rk_m), "category_minus": min(harm, key=rk_m.get), "random20": rr,
                                 "randoms_strictly_worse": int(sum(r > b(rk_m) for r in rr)), "found": bool(sum(r > b(rk_m) for r in rr) >= 19)}}
json.dump(out, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
