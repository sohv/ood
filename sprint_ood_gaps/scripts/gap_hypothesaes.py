"""Part 12 of designs/gap_taxonomy.md, HypotheSAEs arm (authors' package hypothesaes 2.0.1, quickstart functions unchanged).
Stage "sae": embed a pool of 20,000 traffic + 20,000 Dolci first turns (seed 0, excluding the 4,000 part 8 sample texts) and the sample
with text-embedding-3-small (package default) via OpenRouter, train one SAE (M=256, K=8, 10% validation) on the pool.
Stage "arms" [names]: for each Dolci arm, generate_hypotheses on traffic (1) vs that arm (0) with package defaults; interpreter
gpt-5.2 and annotator gpt-5-mini via OpenRouter's Responses API. Writes runs/gap_taxonomy/hypothesaes/{pool.json, sae/, <arm>.csv,
top10.json}."""
import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/gap_taxonomy/hypothesaes"; OUT.mkdir(parents=True, exist_ok=True)
os.environ["EMB_CACHE_DIR"] = str(OUT / "emb_cache"); os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_KEY_SAE"] = os.environ["OPENROUTER_API_KEY"]
import tiktoken.core as _tc  # real prompts contain literal "<|endofprompt|>"; encode such strings as plain text, not raise
_enc = _tc.Encoding.encode
_tc.Encoding.encode = lambda self, text, *, allowed_special=set(), disallowed_special=(): _enc(self, text, allowed_special=allowed_special, disallowed_special=disallowed_special)
from hypothesaes.embedding import get_openai_embeddings
from hypothesaes.quickstart import train_sae, generate_hypotheses
IDX = ROOT / "runs/negatives/index"; EMB = "openai/text-embedding-3-small"

sample = [json.loads(l) for l in open(ROOT / "runs/gap_taxonomy/facets/sample.jsonl")]
stext = {r["id"]: r["text"] for r in sample}
arms = json.load(open(ROOT / "runs/gap_taxonomy/removal_arms.json"))

def pool():
    p = OUT / "pool.json"
    if p.exists(): return json.load(open(p))
    used = set(stext); rng = np.random.default_rng(0)
    tt = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
    ti = [i for i in rng.permutation(len(tt)) if f"t{i}" not in used][:20000]
    dt = {}
    for f in sorted(glob.glob(str(ROOT / "runs/acts_dolci/shard*.jsonl"))):
        for l in open(f):
            r = json.loads(l); dt[r["idx"]] = r["text"]
    di = [i for i in rng.permutation(len(dt)) if f"d{i}" not in used][:20000]
    out = {"traffic": [tt[i] for i in ti], "dolci": [dt[int(i)] for i in di], "traffic_idx": [int(i) for i in ti], "dolci_idx": [int(i) for i in di]}
    json.dump(out, open(p, "w")); return out

def embed(texts, name):
    e = get_openai_embeddings(texts, model=EMB, cache_name=name)  # the package skips empty texts; they get a zero vector here
    d = len(next(iter(e.values()))); return np.stack([e.get(t, np.zeros(d, np.float32)) for t in texts])

stage = sys.argv[1]
if stage == "sae":
    P = pool(); X = embed([t for t in P["traffic"] + P["dolci"] if t.strip()], "pool")
    perm = np.random.default_rng(0).permutation(len(X)); nv = len(X) // 10
    train_sae(X[perm[nv:]], M=256, K=8, val_embeddings=X[perm[:nv]], checkpoint_dir=str(OUT / "sae"))
    embed([r["text"] for r in sample], "sample"); print("sae and sample embeddings done", X.shape, flush=True)
elif stage == "arms":
    from hypothesaes.sae import load_model, get_sae_checkpoint_name
    sae = load_model(str(OUT / "sae" / get_sae_checkpoint_name(256, 8, None)))
    names = sys.argv[2:] or list(arms["dolci_arms"])
    E = dict(zip([r["text"] for r in sample], embed([r["text"] for r in sample], "sample")))
    top = json.load(open(OUT / "top10.json")) if (OUT / "top10.json").exists() else {}
    for a in names:
        ids = arms["traffic"] + list(arms["dolci_arms"][a]); y = np.array([1] * len(arms["traffic"]) + [0] * len(arms["dolci_arms"][a]))
        texts = [stext[i] for i in ids]; X = np.stack([E[t] for t in texts])
        df = generate_hypotheses(texts, y, X, sae, cache_name=f"gap_hsae_{a}", classification=True)
        df.to_csv(OUT / f"{a}.csv", index=False)
        pos = df[df["target_separation_score"] > 0].sort_values("target_separation_score", ascending=False).head(10)
        top[a] = pos[["neuron_idx", "target_separation_score", "interpretation", "f1_fidelity_score"]].to_dict("records")
        json.dump(top, open(OUT / "top10.json", "w"), indent=1); print(a, "done;", len(pos), "traffic-over-Dolci of", len(df), flush=True)
