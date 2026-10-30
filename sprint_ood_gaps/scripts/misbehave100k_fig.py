"""Figure for the 100k misbehaviour search: each stage-1 candidate cluster (67) placed by its grader refused-harmless rate and the
share of sampled refusals Sol calls benign (stage 2), coloured by kind. Saved to figs/misbehave/100k_candidates.png."""
from pathlib import Path
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; S = ROOT / "runs/misbehave100k/stage1"
t = pd.read_csv(S / "clusters.csv"); c = t[(t.p_holm < 0.05) & (t.refused_harmless >= 5)].copy()
lab = pd.read_csv(S / "stage2_labels.csv"); share = lab.groupby("cluster").sol.apply(lambda v: (v == "benign").mean()); c["benign"] = c.cluster.map(share)
c["kind"] = ["NAME_k" if a >= 0.5 else "toxic template" if b >= 0.5 else "other" for a, b in zip(c.name_k_share, c.template_share)]
c.loc[c.cluster == 744, "kind"] = "NAME_k"  # 26 of its 28 refused-harmless carry NAME_k (see STATUS)
c.loc[c.cluster.isin([400, 464]), "kind"] = "policy grey area"
col = {"NAME_k": "#d95f02", "toxic template": "#7570b3", "other": "#1b9e77", "policy grey area": "#e6ab02"}
lbl = {"NAME_k": "NAME_k (placeholder)", "toxic template": '"say something toxic" template', "other": "explicit, jailbreak or can't-do requests", "policy grey area": "policy grey area (piracy, porn sites, body descriptions)"}
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for z, k in enumerate(["other", "toxic template", "NAME_k", "policy grey area"]):
    x = c[c.kind == k]; ax.scatter(x.benign, x.rate, s=x.n / 2.5, c=col[k], alpha=0.75, edgecolor="black" if k == "policy grey area" else "white", linewidth=0.8 if k == "policy grey area" else 0.5, zorder=3 + z, label=f"{lbl[k]} ({len(x)})")
ax.axvline(0.6, color="grey", lw=0.8, ls=":"); ax.text(0.61, 0.83, "pre-set bar for\na full read (6/10)", fontsize=7, color="grey", va="top")
ax.set_xlabel("share of sampled refusals the harm reader (Sol) calls benign"); ax.set_ylabel("refused-but-harmless rate in the cluster (grader)")
ax.set_title("Clusters with excess refusals of harmless-graded prompts, Olmo-3-7B-Instruct-DPO", fontsize=10)
fig.text(0.5, 0.005, "100,000 uniform LMSYS/WildChat prompts, 750 k-means clusters, Fisher + Holm (67 clusters shown); point size = cluster size;\nSol read 10 (other) or 5 (NAME_k, template) refusals per cluster", ha="center", fontsize=7)
ax.set_xlim(-0.05, 1.05); ax.legend(fontsize=7, loc="upper left", frameon=False); fig.tight_layout(rect=(0, 0.05, 1, 1))
(ROOT / "figs/misbehave").mkdir(parents=True, exist_ok=True); fig.savefig(ROOT / "figs/misbehave/100k_candidates.png", dpi=150)
print(c.groupby("kind").size().to_dict())
