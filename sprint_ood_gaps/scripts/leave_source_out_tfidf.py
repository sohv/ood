"""The tf-idf baseline promised in designs/leave_source_out.md, run after the embedding scores (added 2026-10-23 23:30 UTC,
after Sole's review noted it missing). For each held-out source: train the same traffic-vs-mix classifier as
scripts/neg_baselines.py (word unigrams, sublinear tf, min_df 2, 200k features, C = 0.1) on 200,000 traffic and
200,000 reduced-mix prompts (seed 0, held-out source excluded from the mix sample), then score the exact positive and
negative rows of runs/leave_source_out/rows.parquet. Score = P(traffic). Writes runs/leave_source_out/tfidf.csv.
Usage: leave_source_out_tfidf.py [n_procs]"""
import json, re, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/leave_source_out"
N = 200_000
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def load():
    meta = pd.read_parquet(IDX / "mix_meta.parquet")
    sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
    dpo = pd.concat([pd.read_parquet(p, columns=["chosen", "prompt_id"]) for p in paths], ignore_index=True)
    dpo_text = dpo["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
    def prefix(pid):
        pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
    is_sft = meta.part.values == "sft"
    mtext = np.where(is_sft, sft["first_user"].reindex(meta.row.values).values, dpo_text.reindex(meta.row.values).values)
    src = np.where(is_sft, sft["source_dataset"].reindex(meta.row.values).values, ["dpo:" + prefix(p) if isinstance(p, str) else "" for p in meta.prompt_id.values])
    ttext = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
    return mtext, src, ttext

SOURCES = {
    "tool_use": ["Dolci Instruct Tool Use"], "tablegpt": ["TableGPT", "dpo:tulu_v3.9_table_gpt"], "sciriff": ["SciRiff", "dpo:tulu_v3.9_sciriff"],
    "aya": ["Aya", "dpo:tulu_v3.9_aya"], "coconot": ["CoCoNot", "dpo:tulu-3-sft-coconot-regenerated"], "logic_puzzles": ["Logic Puzzles"],
    "precise_if": ["Dolci Instruct Precise IF", "dpo:IF_sft_data_verified_permissive"], "openassistant": ["OpenAssistant", "dpo:oasst1_converted"],
    "wildchat": ["Wildchat", "dpo:filtered_wc_sample", "dpo:Wildchat-1M-gpt-4.1-regenerated-english", "dpo:Wildchat-1m-gpt-4.1-regeneration-not-english"],
}
G = {}
def init():
    G["mtext"], G["src"], G["ttext"] = load()
    G["rows"] = pd.read_parquet(OUT / "rows.parquet")

def one(name):
    mtext, src, ttext, rows = G["mtext"], G["src"], G["ttext"], G["rows"]
    rng = np.random.default_rng(0)
    keep = np.where(~np.isin(src, SOURCES[name]))[0]
    ti = rng.choice(len(ttext), N, replace=False); mi = rng.choice(keep, N, replace=False)
    X_text = [ttext[i][:4000] for i in ti] + [str(mtext[i])[:4000] for i in mi]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
    X = vec.fit_transform(X_text); y = np.r_[np.ones(N), np.zeros(N)]
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(X, y)
    g = rows[rows.held_out == name]
    p = clf.predict_proba(vec.transform([str(mtext[i])[:4000] for i in g.mix_idx.values]))[:, 1]
    yy = g.is_pos.values
    thr = np.quantile(p[yy == 0], 0.98)
    return {"held_out": name, "auroc_tfidf": float(roc_auc_score(yy, p)), "recall_at_2pctFPR_tfidf": float((p[yy == 1] > thr).mean()),
            "auroc_distance": float(roc_auc_score(yy, g.one_sided)), "auroc_ratio": float(roc_auc_score(yy, g.ratio))}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    init()  # load once in the parent; forked workers share it copy-on-write (per-worker loading used ~9 GB each)
    log("loaded texts")
    with Pool(n) as pool:
        res = pool.map(one, list(SOURCES))
    t = pd.DataFrame(res); t.to_csv(OUT / "tfidf.csv", index=False); print(t.round(3).to_string(index=False))
