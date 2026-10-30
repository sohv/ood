"""Part 24 of designs/gap_taxonomy.md: one test for every labelling method, on saved per-prompt labels (no LLM calls).
Categories ranked by traffic share minus Dolci share. F is found if its best F-category rank in minus_F is strictly better than in at
least 19 of 20 same-size random removals from other families (seed 0, as scripts/review_part15_null20.py). Checks: tags, NBER and one
HypotheSAEs score must reproduce earlier outputs. Writes runs/gap_taxonomy/fair_rescore/{summary.json, table.md}."""
import csv, json, os, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "runs/gap_taxonomy"; O = G / "fair_rescore"; O.mkdir(exist_ok=True)
A = json.load(open(G / "removal_arms.json")); fam = A["family"]; full = list(A["dolci_arms"]["full"]); TR = A["traffic"]; FAMS = ["safety", "code", "multilingual"]
rand = {}
for F in FAMS:
    n = len(full) - len(A["dolci_arms"][f"minus_{F}"]); others = [i for i in full if fam[i] != F]; rng = np.random.default_rng(0)
    rand[F] = [set(rng.choice(others, n, replace=False).tolist()) for _ in range(20)]
arm = lambda F: A["dolci_arms"][f"minus_{F}"]

def ci_low(pt, nt, pd, nd):
    """Lower 95% bound (points) of traffic share minus Dolci share."""
    return 100 * (pt - pd - 1.96 * ((pt * (1 - pt) / nt + pd * (1 - pd) / nd) ** 0.5))

def row(best, F):
    """best(dolci ids) -> (rank, n categories, diff points of the best category)."""
    f = best(full); m = best(arm(F)); r = [best([i for i in full if i not in d]) for d in rand[F]]; rr = [x[0] for x in r]
    beats = int(sum(x > m[0] for x in rr)); found = beats >= 19
    return {"n_categories": m[1], "full": f[0], "minus": m[0], "random_min": min(rr), "random_max": max(rr), "beats_of_20": beats, "found": found, "no_room": f[0] == 1, "diff_full": round(float(f[2]), 2), "diff_minus": round(float(m[2]), 2), "diff_minus_ci_low": round(float(m[3]), 2), "reaches_list": bool(found and m[3] > 0)}

def sets_method(labels, fcats, keep=None):
    """labels: id -> set of categories. fcats: F -> set of categories (empty = no category)."""
    tr = [i for i in TR if i in labels]; ct = Counter(c for i in tr for c in labels[i]); cats = [c for c in ct if keep is None or keep(c, ct[c])]
    def best_for(F):
        def best(dolci):
            do = [i for i in dolci if i in labels]; cd = Counter(c for i in do for c in labels[i])
            sc = sorted(((ct[c] / len(tr) - cd[c] / len(do), c) for c in cats), key=lambda p: (-p[0], p[1]))
            hits = [(r, d, c) for r, (d, c) in enumerate(sc, 1) if c in fcats[F]]
            if not hits: return (None, len(sc), 0.0, 0.0)
            r0, d0, c0 = hits[0]; return (r0, len(sc), 100 * d0, ci_low(ct[c0] / len(tr), len(tr), cd[c0] / len(do), len(do)))
        return best
    return {F: (row(best_for(F), F) if any(c in cats for c in fcats[F]) else {"no_category": True, "n_categories": len(cats)}) for F in FAMS}

res = {}
# Tags (part 15 lists)
S = json.load(open(G / "facets/summary.json"))["facets"]; ok = {k for k, v in S.items() if v["kappa_sole"] is not None and v["kappa_sole"] == v["kappa_sole"] and v["kappa_sole"] >= 0.4}
tags = {}
for l in open(G / "facets/tags.jsonl"):
    b = json.loads(l)
    for t in b.get("tags") or []:
        if isinstance(t, dict) and isinstance(t.get("i"), int) and 1 <= t["i"] <= len(b["ids"]): tags[b["ids"][t["i"] - 1]] = {p for p in (t.get("props") or []) if p in ok} | {f"task={t.get('task')}"}
res["our tags"] = sets_method(tags, {"safety": {"harmful_request", "adversarial_pressure"}, "code": {"task=coding", "agentic_or_tool_use"}, "multilingual": {"non_english"}})
# NBER fine topics
nb = {i: {c} for i, c in json.load(open(G / "nber_topic/tags.json")).items() if c}  # 20 prompts have no parsed category
res["NBER topics"] = sets_method(nb, {"safety": set(), "code": {"computer_programming"}, "multilingual": {"translation"}})
# Moderation classifier
ko = {i: {c} for i, c in json.load(open(G / "lang_safety_tax/labels.json")).items()}
res["moderation classifier"] = sets_method(ko, {"safety": {c for v in ko.values() for c in v} - {"OK"}, "code": set(), "multilingual": set()})
# InsTag, parsing and keep rule as scripts/gap_instag_analyse.py
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()
it = {}
for l in open(G / "instag/tags.jsonl"):
    r = json.loads(l); ts = r["tags"] or re.findall(r'"tag"\s*:\s*"([^"]+)"', r["raw"] or "")
    if ts: it[r["id"]] = {norm(t) for t in ts if norm(t)}
trc = Counter(c for i in TR if i in it for c in it[i]); kept = {c for c, n in trc.items() if n >= 10}
KW = {"code": r"\b(code|coding|programming|python|java|javascript|sql|software|debugging|algorithm|html|css|web development|linux|api|data structure)\b",
      "safety": r"harm|jailbreak|illegal|unethical|violen|hate|toxic|offensive|weapon|crime|criminal|suicide|abuse|explicit|sexual|adult|nsfw|drug|racis|sexis|ethic",
      "multilingual": r"translat|language|chinese|spanish|russian|french|german|portuguese|japanese|korean|italian|arabic|foreign|non english|multilingual"}
itf = {F: {c for c in kept if re.search(KW[F], c)} for F in FAMS}
res["InsTag"] = sets_method(it, itf, keep=lambda c, n: n >= 10); res["InsTag"]["_f_categories"] = {F: sorted(v) for F, v in itf.items()}
# LLooM and seeded LLooM: a concept applies at score 1.0, as in the analyse step
def lloom(name, fc):
    df = pd.read_pickle(G / name / "score_df.pkl"); df = df[df["score"].astype(float) >= 1.0]; lab = {}
    for d, c in zip(df["doc_id"], df["concept_name"]): lab.setdefault(d, set()).add(c)
    for i in TR + full: lab.setdefault(i, set())
    return sets_method(lab, fc)
res["LLooM"] = lloom("lloom", {"safety": set(), "code": {"Software Development", "Error Handling", "Graph and Tree Operations", "Data and Text Processing"}, "multilingual": {"Translation and Language"}})
res["LLooM seeded"] = lloom("lloom_seeded", {"safety": set(), "code": {"Data Structure Manipulation", "Optimization Problem", "Iterative Process", "Conditional Logic", "Multiple Conditions", "Technical Complexity"}, "multilingual": set()})
# HypotheSAEs: recompute separation for all 256 neurons on the sample; F-neurons from Sonnet's judged interpretations
os.environ.setdefault("OPENROUTER_API_KEY", "unused")
from hypothesaes.sae import load_model, get_sae_checkpoint_name
from hypothesaes.select_neurons import select_neurons_separation_score as sep
H = G / "hypothesaes"; sae = load_model(str(H / "sae" / get_sae_checkpoint_name(256, 8, None)), device="cpu")
E_ = np.load(H / "emb_cache/sample/chunk_000.npy", allow_pickle=True); emb = {t: np.asarray(v, dtype=np.float32) for t, v in E_}
txt = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(G / "facets/sample.jsonl")}; ids = TR + full
act = sae.get_activations(np.stack([emb[txt[i]] for i in ids]), show_progress=False); pos = {i: k for k, i in enumerate(ids)}
fd = json.load(open(H / "shift.json"))["judges"]["sonnet"]["f_descriptions"]; neu = {F: set() for F in FAMS}; seen = {}
for a in A["dolci_arms"]:
    for r in csv.DictReader(open(H / f"{a}.csv")):
        seen[(a, int(r["neuron_idx"]))] = float(r["target_separation_score"])
        for F in FAMS:
            if r["interpretation"] in fd.get(F, []): neu[F].add(int(r["neuron_idx"]))
def scores(dolci):
    idx = [pos[i] for i in TR] + [pos[i] for i in dolci]; y = np.array([1.0] * len(TR) + [0.0] * len(dolci)); ii, ss = sep(act[idx], y, n_select=256); return dict(zip(map(int, ii), map(float, ss)))
chk = scores(arm("code")); k = next(n for (a, n) in seen if a == "minus_code"); res["_check_hsae"] = {"neuron": k, "csv": seen[("minus_code", k)], "recomputed": chk[k]}
def hbest(F):
    def best(dolci):
        s = sorted(scores(dolci).items(), key=lambda p: -p[1]); hits = [(r, n) for r, (n, v) in enumerate(s, 1) if n in neu[F]]
        if not hits: return (None, len(s), 0.0, 0.0)
        r0, n0 = hits[0]; a = act[:, n0] > 0; pt = a[[pos[i] for i in TR]].mean(); pd_ = a[[pos[i] for i in dolci]].mean()  # share of prompts with the neuron active
        return (r0, len(s), 100 * (pt - pd_), ci_low(pt, len(TR), pd_, len(dolci)))
    return best
res["HypotheSAEs"] = {F: (row(hbest(F), F) if neu[F] else {"no_category": True}) for F in FAMS}; res["HypotheSAEs"]["_f_neurons"] = {F: sorted(v) for F, v in neu.items()}
# Groupings: part 16, already 20 randoms
res["groupings (part 16)"] = json.load(open(G / "grouping_recovery/summary.json"))
json.dump(res, open(O / "summary.json", "w"), indent=1)
lines = ["| method | kind | categories | full | after removal | 20 randoms | beats | found |", "|---|---|---|---|---|---|---|---|"]
for m in ["our tags", "NBER topics", "moderation classifier", "InsTag", "HypotheSAEs", "LLooM", "LLooM seeded"]:
    for F in FAMS:
        v = res[m][F]
        lines.append(f"| {m} | {F} | {v.get('n_categories','')} | no category | | | | no |" if v.get("no_category") else f"| {m} | {F} | {v['n_categories']} | {v['full']} | {v['minus']} | {v['random_min']}-{v['random_max']} | {v['beats_of_20']} | {'yes' if v['found'] else 'no'}{' (no room)' if v['no_room'] else ''}; {v['diff_minus']:+.1f} pts (low {v['diff_minus_ci_low']:+.1f}) |")
(O / "table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print(res["_check_hsae"]); print(res["InsTag"]["_f_categories"]); print(res["HypotheSAEs"]["_f_neurons"])
