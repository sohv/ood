"""Part 14c (designs/gap_taxonomy.md): data-only answer-side gap finder over all of Dolci, safety family included.
Stage "units": every Dolci first-turn prompt joined to its training answer exactly as part 14b (SFT first assistant reply; DPO chosen),
refusal by 14b's keyword rule (refused(), copied unchanged); writes the refused prompts (all families) and the background counts.
Stage "grade": Sohan's safety_removal request grader (../src/generation/request_grader.py, grade_requests, unchanged; Gemini 3.8 Flash
via OpenRouter, 1,000 characters, as run_safety_removal_grade.py) on the refused prompts (uniform sample of 25,000, seed 0, if more).
Stage "words": words (lowercased \\w+, in >= 30 prompts) over-represented among refused AND request_harmful = no prompts vs all Dolci
prompts as background; one-sided hypergeometric, Holm. Writes runs/gap_taxonomy/answers_14c/."""
import ast, asyncio, json, os, random, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "runs/gap_taxonomy/answers_14c"; O.mkdir(parents=True, exist_ok=True)
REF = re.compile(r"(i'?m sorry|i am sorry|i apologi[sz]e|i can'?t|i cannot|i won'?t|i'?m (?:not able|unable)|i am (?:not able|unable)|as an ai\b[^.]{0,60}\b(?:can'?t|cannot|unable|not able))", re.I)
STRONG = re.compile(r"(can'?t|cannot|won'?t|unable|not able) (?:to )?(?:help|assist|provide|comply|fulfil|do that|create|generate|write|share|support|engage)", re.I)
def refused(a):
    h = (a or "")[:300]; return bool(REF.search(h) and STRONG.search(h))
def parse_msgs(x):
    try: return ast.literal_eval(x) if isinstance(x, str) else list(x)
    except Exception: return json.loads(x)
def first_reply(msgs):
    seen = False
    for m in msgs:
        if m["role"] == "user": seen = True
        elif m["role"] == "assistant" and seen: return m["content"]
    return None
def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
WORD = re.compile(r"\w+")
stage = sys.argv[1]

if stage == "units":
    meta = pd.read_parquet(ROOT / "runs/negatives/index/mix_meta.parquet")
    sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["id", "source_dataset", "first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    FAM = json.load(open(ROOT / "runs/one_sided/families.json"))["families"]; safety = set(FAM["safety"])
    src = np.where(meta.part.values == "sft", sft["source_dataset"].reindex(meta.row.values).values, None).astype(object)
    dm = meta.part.values == "dpo"; src[dm] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[dm]]
    sid = np.where(meta.part.values == "sft", sft["id"].reindex(meta.row.values).values, meta.prompt_id.values)
    ref_by_key = {}
    for f in sorted((ROOT / "external/hf/dolci_answers/Dolci-Instruct-SFT/data").glob("*.parquet")):
        t = pd.read_parquet(f, columns=["id", "messages"])
        for i, m in zip(t["id"], t["messages"]): ref_by_key[("sft", i)] = refused(first_reply(parse_msgs(m)))
    for f in sorted((ROOT / "external/hf/dolci_answers/Dolci-Instruct-DPO/data").glob("*.parquet")):
        t = pd.read_parquet(f, columns=["prompt_id", "chosen"])
        for i, m in zip(t["prompt_id"], t["chosen"]): ref_by_key.setdefault(("dpo", i), refused(first_reply(parse_msgs(m))))
    texts = {}
    for s in range(6):
        for l in open(ROOT / f"runs/acts_dolci/shard{s}.jsonl"):
            r = json.loads(l); texts[r["idx"]] = r["text"]
    rows, bg_docs, n_bg = [], Counter(), 0
    for i in range(len(meta)):
        r = ref_by_key.get((meta.part.values[i], sid[i]))
        if r is None or i not in texts: continue
        n_bg += 1; ws = set(WORD.findall(texts[i].lower())); bg_docs.update(ws)
        if r: rows.append({"idx": int(i), "source": src[i], "safety_family": bool(src[i] in safety), "text": texts[i]})
    pd.DataFrame(rows).to_json(O / "refused_units.jsonl", orient="records", lines=True)
    json.dump({"n_background": n_bg, "doc_freq": dict(bg_docs)}, open(O / "background.json", "w"))
    S = {"n_background_prompts_with_answer": n_bg, "n_refused": len(rows), "n_refused_safety_family": sum(r["safety_family"] for r in rows),
         "share_refused": round(len(rows) / n_bg, 4), "n_refused_with_name_k": sum(bool(re.search(r"NAME_\d", r["text"])) for r in rows)}
    json.dump(S, open(O / "units_summary.json", "w"), indent=1); print(S)

elif stage == "grade":
    sys.path.insert(0, str(ROOT.parent))
    from openai import AsyncOpenAI
    from src.generation.request_grader import grade_requests  # Sohan's safety_removal grader, unchanged
    R = pd.read_json(O / "refused_units.jsonl", lines=True)
    if len(R) > 25000: R = R.sample(25000, random_state=0).reset_index(drop=True)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    grades, fails = asyncio.run(grade_requests(client, "google/gemini-3.8-flash", [t[:1000] for t in R.text], str(O / "grade_cache"), 32))
    R["request_harmful"] = [None if g is None else g["request_harmful"] for g in grades]
    R["jailbreak"] = [None if g is None else g["jailbreak"] for g in grades]
    R.to_json(O / "graded.jsonl", orient="records", lines=True)
    json.dump({"n": len(R), "failed": dict(fails), "n_failed": int(R.request_harmful.isna().sum()),
               "harmless": int((R.request_harmful == "no").sum()), "harmful": int((R.request_harmful == "yes").sum())}, open(O / "grade_summary.json", "w"), indent=1)
    print(open(O / "grade_summary.json").read())

elif stage == "words":
    # population (design amendment): answered (not refused) prompts + sampled refused prompts graded harmless
    from scipy.stats import hypergeom
    R = pd.read_json(O / "graded.jsonl", lines=True); B = json.load(open(O / "background.json"))
    U = pd.read_json(O / "refused_units.jsonl", lines=True)
    df_all = B["doc_freq"]; n_all = B["n_background"]
    df_ref = Counter()
    for t in U.text: df_ref.update(set(WORD.findall(t.lower())))
    T = R[R.request_harmful == "no"].reset_index(drop=True); n = len(T)
    tw = Counter(); per = {}
    for i, t in enumerate(T.text):
        for w in set(WORD.findall(t.lower())): tw[w] += 1; per.setdefault(w, []).append(i)
    N = n_all - len(U) + n
    rows = []
    for w, Kall in df_all.items():
        if Kall < 30: continue
        k = tw.get(w, 0); K = Kall - df_ref.get(w, 0) + k
        p = float(hypergeom.sf(k - 1, N, K, n)) if k > 0 else 1.0
        orr = (k / max(n - k, 1)) / (max(K - k, 1) / max(N - n - (K - k), 1))
        rows.append((w, k, K, orr, p))
    W = pd.DataFrame(rows, columns=["word", "k_test", "K_population", "odds_ratio", "p"]).sort_values("p").reset_index(drop=True)
    m = len(W); W["p_holm"] = np.minimum(1.0, np.maximum.accumulate(W.p.values * (m - np.arange(m))))
    W["holm_sig"] = W.p_holm < 0.05; W.to_csv(O / "words.csv", index=False)
    sig = W[W.holm_sig].sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    namek = W[W.word.str.fullmatch(r"name_\d+")]
    nk_rank = {w: int(sig.index[sig.word == w][0]) + 1 for w in namek[namek.holm_sig].word}
    norm = lambda t: " ".join(t.lower().split())[:60]
    rng = random.Random(0)
    md = [f"# Part 14c: words over-represented among refused, harmless Dolci prompts\n\nTest set {n} refused prompts graded harmless (of {len(R)} graded); population {N} (answered + test set); {m} words tested (>= 30 Dolci prompts); {int(W.holm_sig.sum())} Holm-significant.\n",
          "## name_k tokens\n\n" + namek.to_markdown(index=False) + f"\n\nRank by odds ratio among Holm-significant words: {nk_rank}\n"]
    for r_i, r in sig.head(30).iterrows():
        ids = per.get(r.word, []); src = Counter(T.source[i] for i in ids).most_common(3); npre = len({norm(T.text[i]) for i in ids})
        md.append(f"## {r_i + 1}. {r.word}: test {r.k_test}, population {r.K_population}, odds ratio {r.odds_ratio:.1f}, Holm p {r.p_holm:.2g}, distinct 60-char prefixes {npre}\nSources: {src}\n")
        for i in rng.sample(ids, min(5, len(ids))): md.append(f"- {T.text[i][:300].replace(chr(10), ' ')}")
        md.append("")
    open(O / "words.md", "w").write("\n".join(md))
    RN = R[R.text.str.contains(r"NAME_\d", regex=True)]
    S = {"n_graded": len(R), "n_failed": int(R.request_harmful.isna().sum()), "harmless_share_refused": round(float((R.request_harmful == "no").mean()), 4),
         "n_test_set": n, "population": N, "n_words_tested": m, "n_holm_sig": int(W.holm_sig.sum()),
         "name_k_refused_in_sample": len(RN), "name_k_harmless": int((RN.request_harmful == "no").sum()),
         "name_k_harmless_share": round(float((RN.request_harmful == "no").mean()), 3) if len(RN) else None,
         "name_k_rows": namek.to_dict("records"), "name_k_rank_by_odds_among_sig": nk_rank,
         "pass": bool(namek.holm_sig.any()), "found_by_read": bool(any(v <= 30 for v in nk_rank.values()))}
    json.dump(S, open(O / "summary.json", "w"), indent=1); print(json.dumps({k: v for k, v in S.items() if k != "name_k_rows"}, indent=1)); print(namek.to_string())
