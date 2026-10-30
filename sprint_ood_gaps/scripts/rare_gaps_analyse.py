"""Read the one-sided distance scores (Colab A100, float32, TF32 off) for all 992,933 traffic prompts.
1. spot check 2,000 random rows against a CPU faiss search; 2. score = 1 - mean of top-10 Dolci similarities;
3. length correlation; 4. overlap with Sohan's pipeline_million topic ranking; 5. top topics by median score with examples;
6. 50 top prompts and 30 random prompts from the top 1% for reading. Writes runs/rare_gaps/{summary.json, topics.csv, read_*.md}."""
import json, numpy as np, pandas as pd, faiss
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/rare_gaps"; IDX = ROOT / "runs/negatives/index"
D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
z = np.load(OUT / "one_sided.npz"); top_s, top_i = z["top_s"], z["top_i"]
t = np.load(IDX / "traffic.npy", mmap_mode="r"); th = np.load(IDX / "traffic_hash.npy")
m = np.load(IDX / "mix.npy").astype(np.float32); mh = np.load(IDX / "mix_hash.npy")
rng = np.random.default_rng(0); chk = np.sort(rng.choice(len(t), 2000, replace=False))
ix = faiss.IndexFlatIP(384); ix.add(m)
s, i = ix.search(np.asarray(t[chk], dtype=np.float32), 20)
s = np.where(mh[i] == th[chk][:, None], -np.inf, s); s = -np.sort(-s, 1)[:, :10]
spot = {"max_abs_diff_top10": float(np.abs(s - top_s[chk]).max()), "share_top1_within_1e-5": float((np.abs(s[:, 0] - top_s[chk, 0]) < 1e-5).mean())}
print("spot check", spot)
score = 1 - top_s.mean(1); top1 = 1 - top_s[:, 0]
rows = [json.loads(l) for l in open(IDX / "traffic_rows.jsonl")]
text = [r["text"] for r in rows]; src = np.array(["wildchat" if r["src"] in ("deploy", "deploy_ext", "wildchat") else "lmsys" for r in rows])
clen = np.array([len(x) for x in text])
from scipy.stats import spearmanr
lab = np.load(D / "labels.npy"); his = np.load(D / "ratio.npy")
clusters = [json.loads(l) for l in open(D / "clusters.jsonl")]; rank = {int(c["id"][1:]): c["rank"] for c in clusters if c["rank"] is not None}
top1pct = np.argsort(-score)[: len(score) // 100]
summ = {"spot_check": spot, "n": int(len(score)), "score_quantiles": {q: float(np.quantile(score, q)) for q in (0.5, 0.9, 0.99, 0.999)},
        "spearman_score_vs_char_len": float(spearmanr(score, clen).correlation),
        "spearman_score_vs_his_ratio": float(spearmanr(score, his).correlation),
        "top1pct_median_char_len": float(np.median(clen[top1pct])), "pool_median_char_len": float(np.median(clen)),
        "top1pct_share_wildchat": float((src[top1pct] == "wildchat").mean()), "pool_share_wildchat": float((src == "wildchat").mean()),
        "top1pct_share_above_his_line": float((his[top1pct] > 12.1413).mean()), "pool_share_above_his_line": float((his > 12.1413).mean())}
# topics by median one-sided score
tp = pd.DataFrame({"lab": lab, "score": score, "his": his, "clen": clen, "wc": src == "wildchat"}).groupby("lab").agg(
    size=("score", "size"), median_score=("score", "median"), median_his=("his", "median"), median_clen=("clen", "median"), share_wildchat=("wc", "mean")).reset_index()
tp["his_rank"] = tp["lab"].map(rank); tp = tp[tp["size"] >= 50].sort_values("median_score", ascending=False).reset_index(drop=True)
tp["rare_rank"] = np.arange(1, len(tp) + 1); tp.to_csv(OUT / "topics.csv", index=False)
summ["top50_rare_topics_in_his_top293"] = int((tp.head(50).his_rank <= 293).sum())
summ["top293_rare_topics_in_his_top293"] = int((tp.head(293).his_rank <= 293).sum())
(OUT / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
def ex(j): return text[j][:200].replace("\n", " ")
with open(OUT / "read_topics.md", "w") as f:
    f.write("# Top 40 topics by median distance to Dolci (1 - mean top-10 cosine), 3 random members each (seed 0)\n\n")
    for _, r in tp.head(40).iterrows():
        mem = np.where(lab == r.lab)[0]; pick = rng.choice(mem, min(3, len(mem)), replace=False)
        f.write(f"## topic {int(r.lab)}: rare rank {r.rare_rank}, his rank {r.his_rank}, size {r['size']}, median distance {r.median_score:.3f}, his median ratio {r.median_his:.1f}, median chars {r.median_clen:.0f}, WildChat {r.share_wildchat:.0%}\n")
        for j in pick: f.write(f"- {ex(j)}\n")
        f.write("\n")
with open(OUT / "read_prompts.md", "w") as f:
    f.write("# 50 highest-distance prompts, then 30 random prompts from the top 1% (seed 0)\n\n")
    for j in np.argsort(-score)[:50]: f.write(f"- {score[j]:.3f} | {src[j]} | {clen[j]} chars | {ex(j)}\n")
    f.write("\n## random from top 1%\n\n")
    for j in rng.choice(top1pct, 30, replace=False): f.write(f"- {score[j]:.3f} | {src[j]} | {clen[j]} chars | {ex(j)}\n")
