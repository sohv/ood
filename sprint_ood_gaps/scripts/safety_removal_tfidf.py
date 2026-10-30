"""Safety-removal detectors, tf-idf baseline (designs/safety_removal_detectors.md): per arm, the traffic-vs-Dolci classifier of
scripts/leave_source_out_tfidf.py (word unigrams, sublinear tf, min_df 2, 200k features, C = 0.1) on 200,000 traffic prompts
(the pool's own rows excluded, so it cannot memorise them) and 200,000 prompts of that arm's reduced Dolci mix, seed 0; score =
P(traffic) for the pool. Output runs/safety_removal/tfidf.parquet."""
import json, sys, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/safety_removal"
from leave_source_out_tfidf import load
N = 200_000
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
mtext, src, ttext = load(); log("loaded")
pool = pd.read_json(OUT / "pool.jsonl", lines=True)
WGM = ["WildGuardMix", "dpo:tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated"]; WJ = ["WildJailbreak", "dpo:tulu_v3.9_wildjailbreak_decontaminated"]
CCN = ["CoCoNot", "dpo:tulu-3-sft-coconot-regenerated"]
r3 = np.isin(src, WGM + WJ + CCN); rng0 = np.random.default_rng(0); rc = np.zeros(len(src), bool); rc[rng0.choice(len(src), int(r3.sum()), replace=False)] = True
ARMS = {"R0": np.zeros(len(src), bool), "R1": np.isin(src, WGM), "R2": np.isin(src, WJ), "R3": r3, "Rc": rc}
ok_t = np.setdiff1d(np.arange(len(ttext)), pool.row.values); rows = []
for a, rm in ARMS.items():
    rng = np.random.default_rng(0); ti = rng.choice(ok_t, N, replace=False); mi = rng.choice(np.where(~rm)[0], N, replace=False)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
    X = vec.fit_transform([ttext[i][:4000] for i in ti] + [str(mtext[i])[:4000] for i in mi])
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(X, np.r_[np.ones(N), np.zeros(N)])
    p = clf.predict_proba(vec.transform([s[:4000] for s in pool.text]))[:, 1]
    rows += [{"id": i, "arm": a, "tfidf": float(v)} for i, v in zip(pool.id, p)]; log(a, "done")
pd.DataFrame(rows).to_parquet(OUT / "tfidf.parquet")
