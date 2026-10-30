"""Census of Dolci post-training sources: rows per source in Dolci-Instruct-SFT and Dolci-Instruct-DPO.
Writes runs/negatives/census/{sft,dpo}_sources.csv and a combined markdown table. CPU only."""
import sys, json, collections, time
from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parents[1] / "runs/negatives/census"
OUT.mkdir(parents=True, exist_ok=True)

def census(repo, tag):
    t0 = time.time()
    ds = load_dataset(repo, split="train")
    cols = ds.column_names
    print(tag, repo, "rows", len(ds), "cols", cols, flush=True)
    src_col = next((c for c in cols if c in ("source", "source_dataset", "dataset", "dataset_source", "origin")), None)
    if src_col is None:
        print(tag, "no source column; first row keys:", json.dumps({c: str(ds[0][c])[:80] for c in cols}), flush=True)
        return None
    counts = collections.Counter(ds[src_col])
    if "domain" in cols:
        dom = pd.DataFrame(sorted(collections.Counter(ds["domain"]).items(), key=lambda x: -x[1]), columns=["domain", "rows"])
        dom.to_csv(OUT / f"{tag}_domains.csv", index=False)
        print(tag, "domains:", dom.to_dict("records"), flush=True)
    df = pd.DataFrame(sorted(counts.items(), key=lambda x: -x[1]), columns=["source", "rows"])
    df["share"] = df["rows"] / df["rows"].sum()
    df.to_csv(OUT / f"{tag}_sources.csv", index=False)
    # one example first user turn per source, for reading
    ex = {}
    msg_col = next((c for c in cols if c in ("messages", "chosen", "prompt")), None)
    if msg_col:
        seen = set()
        for i in range(len(ds)):
            s = ds[i][src_col]
            if s in seen: continue
            m = ds[i][msg_col]
            if isinstance(m, list) and m and isinstance(m[0], dict):
                txt = next((x.get("content", "") for x in m if x.get("role") == "user"), "")
            else:
                txt = str(m)
            ex[s] = txt[:300]
            seen.add(s)
            if len(seen) == len(counts): break
    (OUT / f"{tag}_examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False))
    print(tag, "done in", round(time.time() - t0), "s;", len(df), "sources", flush=True)
    return df

dfs = {}
for repo, tag in [("allenai/Dolci-Instruct-DPO", "dpo"), ("allenai/Dolci-Instruct-SFT", "sft")]:
    try:
        dfs[tag] = census(repo, tag)
    except Exception as e:
        print(tag, "FAILED", type(e).__name__, str(e)[:300], flush=True)

with open(OUT / "census.md", "w") as f:
    for tag, df in dfs.items():
        if df is None: continue
        f.write(f"## {tag}: {int(df.rows.sum()):,} rows, {len(df)} sources\n\n")
        f.write(df.head(60).to_markdown(index=False, floatfmt=".3f") + "\n\n")
print("wrote", OUT / "census.md")
