"""Part 14d (designs/gap_taxonomy.md). Stage "sheet": reading sheet for the top 100 of part 14c's Holm-significant words by odds ratio
plus every name_\\d+ word: counts, top 3 sources, distinct 60-character prefixes, 5 random test prompts (seed 1, 300 characters).
Stage "agree": precision and Cohen's kappa from two readers' classes (reader files: {word: class}).
Stage "source": NAME_k refusal share vs its own source (runs/triggers/dolci_labelled) and harmless share within WildGuardMix (14c graded).
Writes runs/gap_taxonomy/answers_14d/."""
import json, random, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; I = ROOT / "runs/gap_taxonomy/answers_14c"; O = ROOT / "runs/gap_taxonomy/answers_14d"; O.mkdir(parents=True, exist_ok=True)
WORD = re.compile(r"\w+"); NK = re.compile(r"NAME_\d+")
CLASSES = ["content-free cue", "intended refusal", "single template", "unclear"]
stage = sys.argv[1]

if stage == "sheet":
    W = pd.read_csv(I / "words.csv"); sig = W[W.holm_sig].sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    pick = sig.head(100); extra = sig[sig.word.str.fullmatch(r"name_\d+") & ~sig.word.isin(pick.word)]
    pick = pd.concat([pick.assign(in_top100=True), extra.assign(in_top100=False)]).reset_index(drop=True)
    R = pd.read_json(I / "graded.jsonl", lines=True); T = R[R.request_harmful == "no"].reset_index(drop=True)
    per = {}
    for i, t in enumerate(T.text):
        for w in set(WORD.findall(t.lower())): per.setdefault(w, []).append(i)
    want = set(pick.word); tdf = Counter()
    for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl"):
        tdf.update(want & set(WORD.findall(json.loads(l)["text"].lower())))
    norm = lambda t: " ".join(t.lower().split())[:60]; rng = random.Random(1); items, md = [], []
    for k, r in pick.iterrows():
        ids = per[r.word]; ex = rng.sample(ids, min(5, len(ids)))
        it = {"rank": k + 1, "word": r.word, "in_top100": bool(r.in_top100), "k_test": int(r.k_test), "K_population": int(r.K_population),
              "odds_ratio": round(float(r.odds_ratio), 1), "sources": Counter(T.source[i] for i in ids).most_common(3),
              "distinct_prefixes": len({norm(T.text[i]) for i in ids}), "traffic_count": int(tdf[r.word]), "test_ids": ids, "examples": [T.text[i][:300].replace("\n", " ") for i in ex]}
        items.append(it)
        md.append(f"### {it['rank']}. {it['word']}  (test {it['k_test']}, population {it['K_population']}, odds {it['odds_ratio']}, distinct prefixes {it['distinct_prefixes']}, traffic {it['traffic_count']}, sources {it['sources']})")
        md += [f"- {e}" for e in it["examples"]] + [""]
    json.dump(items, open(O / "sheet.json", "w"), indent=1); open(O / "sheet.md", "w").write("\n".join(md)); print(len(items), "words on the sheet")

elif stage == "agree":
    from sklearn.metrics import cohen_kappa_score
    from statsmodels.stats.proportion import proportion_confint as ci
    items = json.load(open(O / "sheet.json")); A = json.load(open(O / "reader_cb.json")); B = json.load(open(O / "reader_sole.json"))
    words = [it["word"] for it in items]; top = {n: [it["word"] for it in items if it["in_top100"]][:n] for n in (30, 100)}
    S = {"kappa_all": round(float(cohen_kappa_score([A[w] for w in words], [B[w] for w in words])), 3), "by_reader": {},
         "distinct_test_prompts_covered": {f"top{n}": len({i for it in items if it["word"] in ws for i in it["test_ids"]}) for n, ws in top.items()},
         "note": "Wilson intervals treat words as independent; they share prompt families"}
    tc = {it["word"]: it["traffic_count"] for it in items}
    for name, lab in (("cb", A), ("sole", B), ("both", {w: A[w] if A[w] == B[w] else "disagree" for w in words})):
        S["by_reader"][name] = {}
        for n, ws in top.items():
            c = Counter(lab[w] for w in ws); k = c["content-free cue"]; kt = sum(lab[w] == "content-free cue" and tc[w] >= 100 for w in ws)
            lo, hi = ci(kt, len(ws), method="wilson")
            S["by_reader"][name][f"top{n}"] = {"counts": dict(c), "content_free": k, "content_free_and_traffic_ge_100": kt, "n": len(ws), "headline_precision_wilson95": [round(lo, 3), round(hi, 3)]}
    S["content_free_words"] = [{"word": w, "cb": A[w], "sole": B[w], **{k: it[k] for k in ("rank", "k_test", "K_population", "odds_ratio", "sources", "distinct_prefixes", "traffic_count")}}
                               for it in items for w in [it["word"]] if "content-free cue" in (A[w], B[w])]
    json.dump(S, open(O / "summary.json", "w"), indent=1); print(json.dumps(S, indent=1))

elif stage == "source":
    from scipy.stats import fisher_exact
    from statsmodels.stats.proportion import proportion_confint as ci
    D = pd.concat([pd.read_parquet(p, columns=["source", "refused", "prompt"]) for p in sorted((ROOT / "runs/triggers/dolci_labelled").glob("*.parquet"))])
    if D.empty: sys.exit("runs/triggers/dolci_labelled is empty")
    D["namek"] = D.prompt.str.contains(NK); out = {"n_prompts": len(D), "n_namek": int(D.namek.sum()), "by_source": {}}
    for src, g in D.groupby("source"):
        if g.namek.sum() < 10: continue
        a, b = g[g.namek], g[~g.namek]; row = {}
        for lab, x in (("with_namek", a), ("without_namek", b)):
            k, n = int(x.refused.sum()), len(x); lo, hi = ci(k, n, method="wilson") if n else (None, None)
            row[lab] = {"refused": k, "n": n, "share": round(k / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)] if n else None}
        o, p = fisher_exact([[row["with_namek"]["refused"], row["with_namek"]["n"] - row["with_namek"]["refused"]],
                             [row["without_namek"]["refused"], row["without_namek"]["n"] - row["without_namek"]["refused"]]])
        row["fisher_odds_ratio"], row["fisher_p"] = round(float(o), 2), f"{p:.3g}"; out["by_source"][src] = row
    R = pd.read_json(I / "graded.jsonl", lines=True); W = R[(R.source == "WildGuardMix") & R.request_harmful.notna()]
    W = W.assign(namek=W.text.str.contains(NK)); out["wgm_refused_graded_harmless_share"] = {}
    for lab, x in (("with_namek", W[W.namek]), ("without_namek", W[~W.namek])):
        k, n = int((x.request_harmful == "no").sum()), len(x); lo, hi = ci(k, n, method="wilson")
        out["wgm_refused_graded_harmless_share"][lab] = {"harmless": k, "n": n, "share": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}
    json.dump(out, open(O / "source_check.json", "w"), indent=1); print(json.dumps(out, indent=1))

elif stage == "top30":  # readable list for the interview: first 2 of each word's 5 seeded random examples (seed 1), 200 characters
    items = json.load(open(O / "sheet.json")); A = json.load(open(O / "reader_cb.json")); B = json.load(open(O / "reader_sole.json"))
    md = ["# Answer-side finder (part 14c): top 30 words, with reader verdicts and 2 random examples each\n",
          "Words over-represented among Dolci prompts whose training answers refuse although a grader (Sohan's exp80 request grader) calls the prompt harmless; ranked by odds ratio. "
          "Verdicts from two readers (ood-gaps-cb; codex Sole, blind) under the part 14d rule. Examples: the first 2 of 5 drawn at random (seed 1) from the word's test prompts, not chosen; cut to 200 characters. "
          "\"Traffic\" = number of the 992,933 real user first turns containing the word.\n"]
    for it in [x for x in items if x["in_top100"]][:30]:
        w = it["word"]; v = A[w] if A[w] == B[w] else f"readers disagree ({A[w]} / {B[w]})"
        md.append(f"## {it['rank']}. {w}: {v}\n\n{it['k_test']} refused-harmless prompts, odds ratio {it['odds_ratio']}, traffic {it['traffic_count']}, main source {it['sources'][0][0]}\n")
        md += [f"- {e[:200]}" + ("…" if len(e) > 200 else "") for e in it["examples"][:2]] + [""]
    open(O / "top30_examples.md", "w").write("\n".join(md)); print("written", len(md))
