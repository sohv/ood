"""Embedder robustness, reading and figure: topics that move most between MiniLM and Qwen rankings (5 each way, 3 random
members each with both nearest Dolci prompts), Qwen's own top 10 topics, and the rank-rank scatter with labelled topics."""
import gzip, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "runs/robust"; IDX = ROOT / "runs/negatives/index"
D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
tp = pd.read_csv(R / "topics_both.csv", index_col=0); L = pd.read_csv(R / "labelled.csv", index_col=0)
lab = np.load(D / "labels.npy"); zm = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); zq = np.load(R / "qwen_one_sided.npz")
tp["move"] = tp.rare_rank - tp.qwen_rank
up = tp.sort_values("move", ascending=False).head(5); down = tp.sort_values("move").head(5); qtop = tp.sort_values("qwen_rank").head(10)
rng = np.random.default_rng(0); picks = {}
for name, df in (("up under Qwen", up), ("down under Qwen", down), ("Qwen top 10", qtop)):
    for t in df.index: picks[(name, t)] = rng.choice(np.where(lab == t)[0], 3, replace=False)
need_t = {int(j) for v in picks.values() for j in v}
need_m = {int(zm["top_i"][j, 0]) for j in need_t} | {int(zq["top_i"][j, 0]) for j in need_t}
tx = {}
for i, l in enumerate(open(IDX / "traffic_rows.jsonl")):
    if i in need_t: tx[i] = json.loads(l)["text"]
mx = {}
for i, l in enumerate(gzip.open(R / "upload/mix.txt.gz", "rt")):
    if i in need_m: mx[i] = json.loads(l)
def c(s): return s[:220].replace("\n", " ")
with open(R / "read_movers.md", "w") as f:
    f.write("# Topics that move most between MiniLM and Qwen distance rankings, and Qwen's top 10 (3 random members each, seed 0)\n\n")
    for (name, t), js in picks.items():
        r = tp.loc[t]; f.write(f"## {name}: topic {t}, MiniLM rank {int(r.rare_rank)}, Qwen rank {int(r.qwen_rank)}, size {int(r['size'])}, median chars {r.median_clen:.0f}, WildChat {r.share_wildchat:.0%}\n")
        for j in js:
            f.write(f"- traffic: {c(tx[j])}\n  - MiniLM nearest ({zm['top_s'][j,0]:.2f}): {c(mx[int(zm['top_i'][j,0])])}\n  - Qwen nearest ({zq['top_s'][j,0]:.2f}): {c(mx[int(zq['top_i'][j,0])])}\n")
        f.write("\n")
print("qwen top-293 median chars", tp[tp.qwen_rank <= 293].median_clen.median(), "minilm top-293", tp[tp.rare_rank <= 293].median_clen.median(), "all", tp.median_clen.median())
print("qwen top-293 wildchat share", tp[tp.qwen_rank <= 293].share_wildchat.mean().round(3), "minilm", tp[tp.rare_rank <= 293].share_wildchat.mean().round(3))
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(tp.rare_rank, tp.qwen_rank, s=3, c="0.75", lw=0, label="all 1,998 topics")
sty = {("rare_only", "real_gap"): ("#c0392b", "o", "distance-only, user: real gap"), ("rare_only", "other"): ("#c0392b", "x", "distance-only, user: partly/covered"),
       ("control", "real_gap"): ("#2c6fbb", "o", "random, user: real gap"), ("control", "other"): ("#2c6fbb", "x", "random, user: partly/covered")}
L["v"] = np.where(L.verdict == "real_gap", "real_gap", np.where(L.verdict == "unsure", "unsure", "other"))
for (arm, v), (col, mk, name) in sty.items():
    s = L[(L.arm == arm) & (L.v == v)]; ax.scatter(s.rare_rank, s.qwen_rank, c=col, marker=mk, s=28, label=f"{name} (n={len(s)})")
for v in (293,): ax.axhline(v, c="0.4", lw=0.8, ls="--"); ax.axvline(v, c="0.4", lw=0.8, ls="--")
ax.set_xlabel("rank by MiniLM distance to nearest Dolci prompts (1 = farthest)"); ax.set_ylabel("rank by Qwen3-Embedding-0.6B distance")
ax.set_title("Topic rank by distance to Dolci: MiniLM vs Qwen3-Embedding, Olmo-3 traffic", fontsize=10)
fig.text(0.5, 0.005, "1,998 topics (>= 50 members), median 1 - mean top-10 cosine; dashed = top 293\noverlap of top 293: 89 (chance 43); Spearman 0.51; labels from the blind test (MiniLM neighbours shown)", ha="center", fontsize=7, color="0.35")
ax.legend(frameon=False, fontsize=7, loc="lower right"); fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(ROOT / "figs/robust/rank_minilm_vs_qwen.png", dpi=150)
