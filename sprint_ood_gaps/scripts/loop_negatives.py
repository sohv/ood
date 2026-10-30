"""Step 11 of designs/one_sided_negatives.md: negative sets for the loop's confirmation step, fresh calls (cache bypassed).
(a) 40 random exp37 topics with >= half their members verbatim in Dolci, not confirmed by any loop run; (b) exp57 v2's 19 hand-labelled
covered controls. Each topic gets the step 6 rater prompt (10 members, each with its MiniLM-nearest full-Dolci prompt; Sohan's prompt and
parser) and the namer prompt. Writes runs/loop_negatives/{calls/, result.json}."""
import asyncio, glob, json, os, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from statsmodels.stats.proportion import proportion_confint
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT.parent))
from openai import AsyncOpenAI
from scripts.run_autorater_baseline import PROMPT, REF_DESC, fmt_pairs, parse
from src.metrics.discovery import NAMING_PROMPT_GRAIN, name_clusters, naming_prompt, parse_naming
IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/loop_negatives"; OUT.mkdir(exist_ok=True)
RATER, NAMER, CUT, SHARED = "google/gemini-3.8-flash", "anthropic/claude-sonnet-4.6", 50.0, 13

lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
member = np.isin(np.load(IDX / "traffic_hash.npy"), np.load(IDX / "mix_hash.npy"))
share = pd.Series(member).groupby(lab).mean()
confirmed = set()
for f in glob.glob(str(ROOT / "runs/loop_onesided/v*/decision.json")) + glob.glob(str(ROOT / "runs/loop_removal/*/decision.json")):
    d = json.load(open(f)); confirmed |= {t["topic"] for arm in d["gaps"].values() for g in arm for t in g}
pool = sorted(int(t) for t in share[share >= 0.5].index if int(t) not in confirmed)
set_a = np.random.default_rng(0).choice(pool, 40, replace=False).tolist()
set_b = [109, 594, 596, 735, 766, 944, 956, 1102, 1172, 1439, 1633, 758, 527, 719, 918, 240, 456, 1470, 1720]
print(f"pool {len(pool)} verbatim-heavy unconfirmed topics; set a 40, set b {len(set_b)}", flush=True)

emb = np.load(IDX / "traffic.npy").astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
M = torch.from_numpy(np.load(IDX / "mix.npy").astype(np.float32)); M = torch.nn.functional.normalize(M, dim=1)
mix_text = {}
for s in range(6):
    for l in open(ROOT / f"runs/acts_dolci/shard{s}.jsonl"):
        r = json.loads(l); mix_text[r["idx"]] = r["text"]
def rater_prompt(t):
    members = np.flatnonzero(lab == t); shown = np.random.default_rng(0).choice(members, size=min(10, len(members)), replace=False)
    best = (torch.from_numpy(emb[shown]) @ M.T).argmax(1).numpy()
    topic, ref = fmt_pairs([texts[i] for i in shown], [mix_text[int(i)] for i in best], 300)
    return PROMPT.format(ref_desc=REF_DESC["B"], topic=topic, ref=ref)
def namer_prompt(t):
    return naming_prompt([texts[i] for i in np.flatnonzero(lab == t)], 25, 0, template=NAMING_PROMPT_GRAIN)

topics = set_a + set_b
cl = lambda: AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])  # one client per event loop
rr = asyncio.run(name_clusters(cl(), RATER, [rater_prompt(t) for t in topics], tempfile.mkdtemp(dir=OUT), 6))
nr = asyncio.run(name_clusters(cl(), NAMER, [namer_prompt(t) for t in topics], tempfile.mkdtemp(dir=OUT), 6))
rows = []
for t, r, n in zip(topics, rr, nr):
    cov, lbl = parse(r) if r else (None, None); nm = parse_naming(n) if n else {}
    rows.append({"topic": int(t), "set": "a_verbatim" if t in set_a else "b_exp57_controls", "coverage": cov, "rater_label": lbl,
                 "rater_confirms": cov is not None and cov <= CUT, "shared": nm.get("shared"), "name": nm.get("name"),
                 "namer_confirms": nm.get("shared") is not None and nm.get("shared") >= SHARED, "rater_failed": r is None, "namer_failed": n is None})
pd.DataFrame(rows).to_json(OUT / "rows.jsonl", orient="records", lines=True)
res = {}
for s in ["a_verbatim", "b_exp57_controls"]:
    g = [r for r in rows if r["set"] == s]
    for step in ["rater", "namer"]:
        ok = [r for r in g if not r[f"{step}_failed"]]; k = sum(r[f"{step}_confirms"] for r in ok)
        lo, hi = proportion_confint(k, len(ok), method="wilson") if ok else (None, None)
        res[f"{s}/{step}"] = {"confirmed": k, "n": len(ok), "failed": len(g) - len(ok), "wilson95": [round(lo, 3), round(hi, 3)] if ok else None}
json.dump(res, open(OUT / "result.json", "w"), indent=1); print(json.dumps(res, indent=1))
