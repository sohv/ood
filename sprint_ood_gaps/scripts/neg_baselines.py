"""Cheap baselines on the same eval rows as the neighbour ratio (designs/negatives_eval.md, review item d):
  1. Olmo token length (allenai/Olmo-3-7B-Instruct-DPO tokenizer, rev b33130b7), on the first 4,000 chars as his max_chars.
  2. tf-idf logistic classifier, traffic (1) vs mix (0), trained on 200,000 + 200,000 rows (seed 0) from the rebuilt pools,
     word unigrams, sublinear tf, min_df 2, 200k features, C = 0.1: the same recipe as Sohan's tfidf baseline in
     ../src/metrics/detectors.py. Score = P(traffic).
Adds "olmo_len" and "tfidf_p_traffic" to every scored jsonl under runs/negatives/scores/, in place.
"""
import json, time, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
IDX = ROOT / "runs/negatives/index"; SC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/negatives/scores"
SEED, N_TRAIN = 0, 200_000


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    files = sorted(glob.glob(str(SC / "*.jsonl")))
    rows = {f: [json.loads(l) for l in open(f)] for f in files}
    log("files", len(files), "rows", sum(len(v) for v in rows.values()))

    # 1. Olmo token length
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-DPO", revision="b33130b7de49f0c2553b5c2b3bc8409ff3e627d1")
    for f, rs in rows.items():
        lens = [len(x) for x in tok([r["text"][:4000] for r in rs], add_special_tokens=False)["input_ids"]]
        for r, n in zip(rs, lens): r["olmo_len"] = int(n)
    log("token lengths done")

    # 2. tf-idf traffic-vs-mix classifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(SEED)
    t_rows = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
    meta = pd.read_parquet(IDX / "mix_meta.parquet")
    sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
    dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths], ignore_index=True)["chosen"].map(
        lambda msgs: next(m["content"] for m in msgs if m["role"] == "user"))
    ti = rng.choice(len(t_rows), N_TRAIN, replace=False); mi = rng.choice(len(meta), N_TRAIN, replace=False)
    m_text = [(sft.iat[int(meta.at[j, "row"])] if meta.at[j, "part"] == "sft" else dpo.iat[int(meta.at[j, "row"])]) for j in mi]
    X_text = [t_rows[i][:4000] for i in ti] + [t[:4000] for t in m_text]
    y = np.r_[np.ones(N_TRAIN), np.zeros(N_TRAIN)]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
    X = vec.fit_transform(X_text)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(X, y)
    log("tfidf classifier trained; train acc", float(clf.score(X, y)))
    # the eval rows that are themselves training rows (mix anchors, traffic rows) get a leave-in score; flagged so it can be read
    train_text = set(X_text)
    for f, rs in rows.items():
        p = clf.predict_proba(vec.transform([r["text"][:4000] for r in rs]))[:, 1]
        for r, v in zip(rs, p):
            r["tfidf_p_traffic"] = float(v); r["tfidf_in_train"] = r["text"][:4000] in train_text
        with open(f, "w") as fh:
            for r in rs: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"n_train_each": N_TRAIN, "seed": SEED, "features": int(X.shape[1]), "train_acc": float(clf.score(X, y))}, open(SC / "_tfidf_baseline.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
