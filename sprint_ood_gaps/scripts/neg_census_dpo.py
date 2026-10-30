"""Dolci-Instruct-DPO has no source column; the source is encoded in prompt_id (e.g. 'correct-python-sft-187k-request-70-606').
Counts rows by the prompt_id prefix before the first numeric-ish segment, and by preference_type."""
import re, collections, json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
OUT = Path(__file__).resolve().parents[1] / "runs/negatives/census"; OUT.mkdir(parents=True, exist_ok=True)
ds = load_dataset("allenai/Dolci-Instruct-DPO", split="train")
def src(pid):
    pid = re.sub(r"-request-.*$", "", pid)
    pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid)
    pid = re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
    return pid
c = collections.Counter(src(p) for p in ds["prompt_id"])
df = pd.DataFrame(sorted(c.items(), key=lambda x: -x[1]), columns=["source_prefix", "rows"]); df["share"] = df.rows / df.rows.sum()
df.to_csv(OUT / "dpo_sources.csv", index=False)
pt = collections.Counter(ds["preference_type"])
ex = {}
for i in range(len(ds)):
    s = src(ds[i]["prompt_id"])
    if s not in ex:
        m = ds[i]["chosen"]; ex[s] = next((x["content"] for x in m if x.get("role") == "user"), "")[:300]
    if len(ex) == len(c): break
(OUT / "dpo_examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False))
with open(OUT / "dpo_census.md", "w") as f:
    f.write(f"## Dolci-Instruct-DPO: {len(ds):,} rows, {len(df)} prompt_id prefixes\n\npreference_type: {dict(pt)}\n\n")
    f.write(df.head(80).to_markdown(index=False, floatfmt=".3f") + "\n")
print(df.head(40).to_string()); print("prefixes:", len(df))
