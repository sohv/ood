"""LLM-judge baseline calls (designs/judge_baseline.md). Arms: same (screen neighbour, cosine hidden), j50 (5 tf-idf
neighbours per prompt), hyb (MiniLM top 5 plus tf-idf top 5). Judges via OpenRouter, default sampling. Every request and reply is cached and kept in
runs/judge/raw.jsonl; failed calls stay as rows with reply None. Usage: judge_run.py pilot|full"""
import asyncio, hashlib, json, os, re, sys
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/judge"; CACHE = OUT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
for line in open(".env"):
    if line.startswith("OPENROUTER_API_KEY="): os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
MODELS = ["google/gemini-3.8-flash", "anthropic/claude-sonnet-4.6"]
HEAD = """Below are 10 user prompts sampled from one topic: a cluster of real chat traffic sent to AI assistants. Under each prompt are {what} from the post-training data of a language model, found by searching all 1.7 million first user turns in that data. Question: does the post-training data contain this kind of request?

- Real gap: the post-training data has nothing of this kind.
- Partly: it has the kind but not this angle.
- Covered: it has this kind.

Rule of thumb: if most of the closest training prompts are clearly the same kind of request, covered; clearly a different kind, real gap; related but missing the point, partly. Judge the topic as a whole, not single prompts.

{blocks}

Write your reasoning first (a few sentences). Then end with exactly two lines:
SCORE: <integer 0-100, how likely it is that this kind of request is missing from the post-training data>
LABEL: <real_gap, partly or covered>"""
WHAT = {"same": "the single closest prompt", "j50": "the 5 closest prompts by shared words", "hyb": "the closest prompts by meaning and by shared words (up to 10)"}

def build(topic, arm):
    blocks = []
    for j, e in enumerate(topic["examples"]):
        nbs = {"same": [e["screen_nearest"]], "j50": e["tfidf_nearest"], "hyb": e.get("hybrid_nearest")}[arm]
        blocks.append(f"PROMPT {j + 1}: {e['text']}\n" + "\n".join(f"  training prompt: {n}" for n in nbs))
    return HEAD.format(what=WHAT[arm], blocks="\n\n".join(blocks))

def parse(reply):
    if not reply: return None, None
    s = re.findall(r"SCORE:\s*(\d{1,3})", reply); l = re.findall(r"LABEL:\s*\**\s*(real_gap|partly|covered)", reply)
    return (int(s[-1]) if s and int(s[-1]) <= 100 else None), (l[-1] if l else None)

async def call(client, sem, model, prompt):
    path = CACHE / (hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest() + ".json")
    if path.exists(): return json.loads(path.read_text())
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
                out = {"reply": r.choices[0].message.content, "model_served": r.model, "usage": r.usage.model_dump() if r.usage else None}
                if out["reply"]: path.write_text(json.dumps(out)); return out
            except Exception as ex:
                print(f"FAILED {model} attempt {attempt}: {ex!r}", flush=True)
            await asyncio.sleep(5 * (attempt + 1))
    return {"reply": None, "model_served": None, "usage": None}

async def main(mode):
    topics = [json.loads(l) for l in open(OUT / "inputs.jsonl")]
    if mode == "pilot":
        pick = np.random.default_rng(0).choice(len(topics), 5, replace=False); jobs = [(topics[i], a, MODELS[0]) for i in pick for a in ("same", "j50")]
    else:
        jobs = [(t, a, m) for m in MODELS for a in ("same", "j50", "hyb") for t in topics]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]); sem = asyncio.Semaphore(16)
    prompts = [build(t, a) for t, a, _ in jobs]
    res = await asyncio.gather(*[call(client, sem, m, p) for (_, _, m), p in zip(jobs, prompts)])
    with open(OUT / f"raw_{mode}.jsonl", "w") as f:
        for (t, a, m), p, r in zip(jobs, prompts, res):
            score, label = parse(r["reply"])
            f.write(json.dumps({"id": t["id"], "arm": a, "model": m, "prompt": p, **r, "score": score, "label": label}, ensure_ascii=False) + "\n")
    n_bad = sum(parse(r["reply"])[0] is None or parse(r["reply"])[1] is None for r in res)
    tok = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in res)
    print(f"{len(jobs)} calls, {n_bad} failed or unparsed, {tok} prompt tokens, max prompt chars {max(map(len, prompts))}")

asyncio.run(main(sys.argv[1]))
