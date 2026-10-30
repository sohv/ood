"""Gap taxonomy part 12, LLooM arm (designs/gap_taxonomy.md): LLooM (text_lloom 0.8.2.1, authors' code, unchanged) on traffic vs Dolci.
Run with external/lloom/.venv/bin/python. Usage: gap_lloom.py gen | estimate | score | analyse.
gen: induce concepts on 500 traffic + 500 full-Dolci prompts (seed 0) with LLooM's defaults -> runs/gap_taxonomy/lloom/{session_gen.pkl, concepts.json}.
score: LLooM's score on all 4,000 sample prompts (batch 5, highlights) -> score_df.pkl. analyse: prevalence per arm, top 10 per arm (top10.json).
Models are LLooM's defaults (gpt-4o-mini, gpt-4o, text-embedding-3-large) served via OpenRouter; only the endpoint differs from llm_openai.py."""
import asyncio, builtins, json, math, pickle, random, sys
from pathlib import Path
import numpy as np, pandas as pd
import text_lloom.workbench as wb
from text_lloom.llm import OpenAIModel, OpenAIEmbedModel
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy/lloom"; D.mkdir(parents=True, exist_ok=True)
OR = "https://openrouter.ai/api/v1"; SEM = None; MAXCONC = 16; MAX_CONCEPTS = 30
for line in open(".env"):
    if line.startswith("OPENROUTER_API_KEY="): KEY = line.split("=", 1)[1].strip().strip('"')
builtins.input = lambda *a: "y"  # LLooM asks for confirmation in debug mode; debug mode is needed for its auto-review step
wb.display = print

def setup_llm(api_key):
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=OR, api_key=api_key, max_retries=5)

def setup_embed(api_key):
    from openai import OpenAI
    return OpenAI(base_url=OR, api_key=api_key, max_retries=5)

async def call_llm(model, prompt):  # = text_lloom.llm_openai.call_llm_fn with the OpenRouter model id and a concurrency cap
    global SEM
    SEM = SEM or asyncio.Semaphore(MAXCONC)
    model.args.setdefault("system_prompt", "You are a helpful assistant who helps with identifying patterns in text examples.")
    model.args.setdefault("temperature", 0)
    prompt = model.truncate_fn(model, prompt, out_token_alloc=1500)
    async with SEM:
        res = await model.client.chat.completions.create(model="openai/" + model.name, temperature=model.args["temperature"],
            messages=[{"role": "system", "content": model.args["system_prompt"]}, {"role": "user", "content": prompt}])
    with open(D / "calls.jsonl", "a") as f:
        f.write(json.dumps({"model": model.name, "in": res.usage.prompt_tokens, "out": res.usage.completion_tokens}) + "\n")
    return res.choices[0].message.content, (res.usage.prompt_tokens, res.usage.completion_tokens)

def call_embed(model, texts):  # = call_embed_fn with the OpenRouter model id
    resp = model.client.embeddings.create(input=texts, model="openai/" + model.name)
    return [r.embedding for r in resp.data], np.sum([model.count_tokens_fn(model, t) for t in texts])

def models():
    m = lambda n: OpenAIModel(name=n, api_key=KEY, setup_fn=setup_llm, fn=call_llm)
    return dict(distill_model=m("gpt-4o-mini"), synth_model=m("gpt-4o"), score_model=m("gpt-4o-mini"),
                cluster_model=OpenAIEmbedModel(name="text-embedding-3-large", api_key=KEY, setup_fn=setup_embed, fn=call_embed))

def sample_df():
    R = [json.loads(l) for l in open(ROOT / "runs/gap_taxonomy/facets/sample.jsonl")]
    return pd.DataFrame({"id": [r["id"] for r in R], "src": [r["src"] for r in R], "text": [r["text"][:2000] for r in R]})

def _concepts(l):
    return {cid: {"name": c.name, "prompt": c.prompt, "example_ids": list(c.example_ids), "active": c.active} for cid, c in l.concepts.items()}

def _load():
    l = pickle.load(open(D / "session_gen.pkl", "rb"))
    for k, v in models().items(): setattr(l, k, v)
    return l

def gen():
    df = sample_df(); rng = random.Random(0)
    ids = rng.sample(list(df.id[df.src == "traffic"]), 500) + rng.sample(list(df.id[df.src == "dolci"]), 500)
    ind = df[df.id.isin(ids)][["id", "text"]].sample(frac=1, random_state=0).reset_index(drop=True)
    ind.to_csv(D / "induction_set.csv", index=False)
    l = wb.lloom(df=ind, text_col="text", id_col="id", **models())
    asyncio.run(l.gen(debug=True))
    if len(l.concepts) > MAX_CONCEPTS:
        for c in l.concepts.values(): c.active = False
        asyncio.run(l.select_auto(MAX_CONCEPTS))
    else:
        for c in l.concepts.values(): c.active = True
    json.dump(_concepts(l), open(D / "concepts.json", "w"), indent=1)
    l.save(str(D), "session_gen"); print(len(l.concepts), "concepts;", sum(c.active for c in l.concepts.values()), "active")

def estimate():
    import tiktoken
    l = _load(); df = sample_df(); enc = tiktoken.encoding_for_model("gpt-4o-mini")
    n_c = sum(c.active for c in l.concepts.values()); doc_tok = sum(len(enc.encode(t)) for t in df.text)
    p_tok = len(enc.encode(wb.score_highlight_prompt)) + 20; n_b = math.ceil(len(df) / 5)
    tin = n_c * (doc_tok + n_b * p_tok); tout = n_c * len(df) * 100  # LLooM's own 100 output tokens per document
    print(f"{n_c} concepts x {len(df)} docs: in {tin / 1e6:.1f}M tok, out {tout / 1e6:.1f}M tok, est ${tin * 0.15e-6 + tout * 0.6e-6:.2f} (gpt-4o-mini list price)")

def score():
    l = _load(); df = sample_df()
    s = asyncio.run(l.score(batch_size=5, get_highlights=True, df=df[["id", "text"]], debug=True))
    s.to_pickle(D / "score_df.pkl"); l.save(str(D), "session_scored"); print(s.shape)

def analyse():
    from statsmodels.stats.proportion import proportion_confint
    s = pd.read_pickle(D / "score_df.pkl"); A = json.load(open(ROOT / "runs/gap_taxonomy/removal_arms.json"))
    C = json.load(open(D / "concepts.json")); s["pos"] = s.score >= 1.0
    P = s.pivot_table(index="doc_id", columns="concept_id", values="pos", aggfunc="max").fillna(False).astype(bool)
    ci = lambda k, n: [round(x, 4) for x in proportion_confint(k, n, method="wilson")]
    tr = [i for i in A["traffic"] if i in P.index]; res = {"n_scored_docs": len(P), "n_concepts": P.shape[1], "n_traffic": len(tr), "arms": {}}
    md = [f"# LLooM concepts, traffic vs each Dolci arm (n traffic = {len(tr)}; a concept applies at score 1.0)\n"]; top10 = {}
    for arm, ids in A["dolci_arms"].items():
        ids = [i for i in ids if i in P.index]; rows = []
        for cid in P.columns:
            t, d = int(P.loc[tr, cid].sum()), int(P.loc[ids, cid].sum())
            rows.append({"concept_id": cid, "name": C[cid]["name"], "prompt": C[cid]["prompt"], "description": f"{C[cid]['name']}: {C[cid]['prompt']}",
                         "traffic": round(t / len(tr), 4), "traffic_ci": ci(t, len(tr)), "dolci": round(d / len(ids), 4), "dolci_ci": ci(d, len(ids)),
                         "diff": round(t / len(tr) - d / len(ids), 4), "ratio": round((t / len(tr)) / (d / len(ids)), 2) if d else None})
        rows.sort(key=lambda r: -r["diff"]); res["arms"][arm] = {"n_dolci": len(ids), "concepts": rows}
        top10[arm] = [{k: r[k] for k in ("concept_id", "description", "name", "prompt", "traffic", "dolci", "diff")} for r in rows if r["diff"] > 0][:10]
        md += [f"\n## {arm} (n Dolci = {len(ids)})\n", "| rank | concept | traffic | Dolci | diff |", "|---|---|---|---|---|"]
        md += [f"| {k + 1} | {r['name']} | {r['traffic']:.3f} | {r['dolci']:.3f} | {r['diff']:+.3f} |" for k, r in enumerate(rows)]
    for fam in ("safety", "code", "multilingual"):
        a = {r["concept_id"]: r for r in res["arms"][f"minus_{fam}"]["concepts"]}; b = {r["concept_id"]: r for r in res["arms"][f"random_{fam}"]["concepts"]}
        ra = {c: k + 1 for k, c in enumerate(a)}; rb = {c: k + 1 for k, c in enumerate(b)}
        res[f"shift_{fam}"] = sorted([{"concept_id": c, "name": C[c]["name"], "dolci_minus": a[c]["dolci"], "dolci_random": b[c]["dolci"],
                                       "dolci_change": round(a[c]["dolci"] - b[c]["dolci"], 4), "rank_minus": ra[c], "rank_random": rb[c]} for c in a],
                                     key=lambda r: r["dolci_change"])
    json.dump(res, open(D / "summary.json", "w"), indent=1); json.dump(top10, open(D / "top10_detail.json", "w"), indent=1)
    json.dump({a: [r["description"] for r in v] for a, v in top10.items()}, open(D / "top10.json", "w"), indent=1)  # format read by gap_removal_judge.py
    (D / "top10.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    {"gen": gen, "estimate": estimate, "score": score, "analyse": analyse}[sys.argv[1]]()
