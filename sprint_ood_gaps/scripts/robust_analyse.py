"""Compare topic rankings by distance to Dolci under MiniLM (runs/rare_gaps/one_sided.npz) and Qwen3-Embedding-0.6B
(runs/robust/qwen_one_sided.npz), per designs/robust_embedder.md. Topics = Sohan's exp37 labels with >= 50 members;
score = 1 - mean top-10 cosine (as topics.csv), also nearest-one. With --check_only: rebuild topics.csv medians from MiniLM only.
Outputs: runs/robust/{topics_both.csv, summary.json, labelled.csv, read_movers.md}, figs/robust/*.png."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/robust"; FIG = ROOT / "figs/robust"
D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
lab = np.load(D / "labels.npy")

def topic_medians(npz):
    z = np.load(npz); s = z["top_s"]
    df = pd.DataFrame({"lab": lab, "mean10": 1 - s.mean(1), "near1": 1 - s[:, 0]}).groupby("lab")
    return df.agg(size=("mean10", "size"), mean10=("mean10", "median"), near1=("near1", "median")), z["top_i"]

old = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv").set_index("lab")
mini, mini_i = topic_medians(ROOT / "runs/rare_gaps/one_sided.npz")
chk = mini.loc[old.index]
print("reproduce topics.csv: max |median diff|", float(np.abs(chk.mean10 - old.median_score).max()), "n topics", len(old))
if "--check_only" in sys.argv: sys.exit()
qwen, qwen_i = topic_medians(OUT / "qwen_one_sided.npz")
tp = old[["size", "median_clen", "share_wildchat", "his_rank", "rare_rank"]].copy()
tp["mini"] = mini.loc[tp.index, "mean10"]; tp["mini_near1"] = mini.loc[tp.index, "near1"]
tp["qwen"] = qwen.loc[tp.index, "mean10"]; tp["qwen_near1"] = qwen.loc[tp.index, "near1"]
tp["qwen_rank"] = tp.qwen.rank(ascending=False, method="first").astype(int)
tp["qwen_near1_rank"] = tp.qwen_near1.rank(ascending=False, method="first").astype(int)
tp["mini_near1_rank"] = tp.mini_near1.rank(ascending=False, method="first").astype(int)
tp.to_csv(OUT / "topics_both.csv")
N = len(tp); T = 293
def ov(a, b, k=T): return int(((tp[a] <= k) & (tp[b] <= k)).sum())
summ = {"n_topics": N, "chance_overlap_top293": T * T / N,
        "overlap_top293_mean10": ov("rare_rank", "qwen_rank"), "overlap_top293_near1": ov("mini_near1_rank", "qwen_near1_rank"),
        "overlap_top293_mini_mean10_vs_mini_near1": ov("rare_rank", "mini_near1_rank"),
        "overlap_top30_mean10": ov("rare_rank", "qwen_rank", 30), "overlap_top100_mean10": ov("rare_rank", "qwen_rank", 100),
        "spearman_topics_mean10": float(spearmanr(tp.mini, tp.qwen).correlation),
        "spearman_topics_near1": float(spearmanr(tp.mini_near1, tp.qwen_near1).correlation),
        "spearman_rankchange_vs_median_chars": float(spearmanr(tp.rare_rank - tp.qwen_rank, tp.median_clen).correlation),
        "overlap_top293_qwen_vs_sohan_ratio": ov("qwen_rank", "his_rank"), "overlap_top293_mini_vs_sohan_ratio": ov("rare_rank", "his_rank")}
z = np.load(OUT / "qwen_one_sided.npz"); zm = np.load(ROOT / "runs/rare_gaps/one_sided.npz")
summ["spearman_prompts_mean10"] = float(spearmanr(1 - zm["top_s"].mean(1), 1 - z["top_s"].mean(1)).correlation)
summ["share_same_nearest_dolci"] = float((zm["top_i"][:, 0] == z["top_i"][:, 0]).mean())
# labelled topics
key = json.load(open(ROOT / "runs/rare_gaps/labels/key.json")); dump = {d["id"]: d["data"] for d in json.load(open(ROOT / "runs/rare_gaps/labels/labels_dump.json"))}
L = pd.DataFrame([{"id": i, "arm": v["arm"], "topic": v["topic"], "verdict": dump[i]["verdict"]} for i, v in key.items() if i in dump]).set_index("topic")
L = L.join(tp[["rare_rank", "qwen_rank", "his_rank", "mini", "qwen", "median_clen"]])
# share of members whose exact normalised text is in Dolci (the score masks those copies; red-team note 2026-10-24)
in_dolci = np.isin(np.load(IDX / "traffic_hash.npy"), np.load(IDX / "mix_hash.npy"))
L["share_exact_in_dolci"] = pd.Series(in_dolci).groupby(lab).mean().reindex(L.index).values; L.to_csv(OUT / "labelled.csv")
lab_sum = {}
for arm in ("rare_only", "ratio_only", "control"):
    g = L[L.arm == arm]
    lab_sum[arm] = {"n": len(g), "in_qwen_top293": int((g.qwen_rank <= T).sum()), "median_qwen_rank": float(g.qwen_rank.median()),
                    "real_gap_in_qwen_top293": f"{int(((g.verdict == 'real_gap') & (g.qwen_rank <= T)).sum())}/{int((g.verdict == 'real_gap').sum())}",
                    "not_real_in_qwen_top293": f"{int(((g.verdict.isin(['partly', 'covered'])) & (g.qwen_rank <= T)).sum())}/{int(g.verdict.isin(['partly', 'covered']).sum())}"}
K = L[L.verdict != "unsure"]; y = K.verdict == "real_gap"
lab_sum["auroc_real_gap_vs_not_81"] = {"qwen": float(roc_auc_score(y, K.qwen)), "minilm_selection_inflated": float(roc_auc_score(y, K.mini))}
Kr = K[K.arm != "control"]; Kc = K[K.arm == "control"]
lab_sum["auroc_within_control_arm"] = {"n": len(Kc), "pos": int((Kc.verdict == "real_gap").sum()),
                                      "qwen": float(roc_auc_score(Kc.verdict == "real_gap", Kc.qwen)), "minilm": float(roc_auc_score(Kc.verdict == "real_gap", Kc.mini))}
summ["labelled"] = lab_sum
(OUT / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
