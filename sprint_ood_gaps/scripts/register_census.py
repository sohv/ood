"""Keyword census of Dolci first turns for the register-test topics, plus the share of first-person narrative markers;
and the share of non-English prompts (fewer than 50% ASCII letters among letters) in the distance top 1% vs the pool.
Keyword lists are mine, written before looking at counts; crude, reported as such."""
import json, re, numpy as np, pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT = Path(__file__).resolve().parents[1]
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])["first_user"].dropna()
paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths])["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
mix = pd.concat([sft, dpo]).drop_duplicates().str.lower()
kw = {"science explanation (why does)": r"\bwhy (do|does|is|are)\b", "cooking": r"\b(recipe|cook|bake|baking|oven|dish)\b",
      "fitness": r"\b(workout|gym|exercise|squat|cardio|lifting)\b", "programming learning": r"\b(learn(ing)? (to )?(code|program|python|javascript))\b",
      "relationships": r"\b(my (boyfriend|girlfriend|husband|wife|partner))\b", "writing craft": r"\b(my (novel|story|manuscript)|writing a (book|novel))\b",
      "legal": r"\b(lawyer|attorney|landlord|sue|lawsuit|legal advice)\b", "medical symptoms": r"\b(doctor|symptom|symptoms|diagnos)\b",
      "reddit age-sex marker e.g. (30 M)": r"\(\s?\d{2}\s?[mf]\s?\)|\b\d{2}\s?[mf]\b(?=[,)\s])", "first-person narrative opener": r"^(so |hi,? |hey,? )?(i|my)\b"}
rows = [{"pattern": k, "dolci_rows": int(mix.str.contains(v, regex=True).sum()), "per_100k": round(1e5 * mix.str.contains(v, regex=True).mean(), 1)} for k, v in kw.items()]
cen = pd.DataFrame(rows); print(f"Dolci unique first turns: {len(mix):,}"); print(cen.to_string(index=False))
# same patterns on the register sets, for comparison
reg = {}
for f in sorted((ROOT / "runs/negatives/sets_register").glob("*.jsonl")) + [ROOT / "runs/negatives/sets/pos_legal.jsonl", ROOT / "runs/negatives/sets/pos_medical.jsonl"]:
    t = pd.Series([json.loads(l)["text"].lower() for l in open(f)])
    reg[f.stem] = {k: round(100 * t.str.contains(v, regex=True).mean(), 1) for k, v in kw.items() if k.startswith(("reddit", "first-person"))}
print(pd.DataFrame(reg).T.to_string())
# language of the distance top 1%
z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); score = 1 - z["top_s"].mean(1)
txt = [json.loads(l)["text"] for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]
def non_en(s):
    L = [c for c in s[:500] if c.isalpha()]
    return len(L) > 0 and sum(c.isascii() for c in L) / len(L) < 0.5
top = np.argsort(-score)[: len(score) // 100]; rnd = np.random.default_rng(0).choice(len(score), 20000, replace=False)
lang = {"top1pct_non_english_share": float(np.mean([non_en(txt[i]) for i in top])), "pool_sample_non_english_share": float(np.mean([non_en(txt[i]) for i in rnd]))}
print(lang)
out = {"census": rows, "register_markers_pct": reg, "language": lang, "n_dolci_unique": int(len(mix))}
(ROOT / "runs/negatives/register_census.json").write_text(json.dumps(out, indent=1))
