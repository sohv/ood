# summary tables and the figure for the word-baseline run. Sohan's activation numbers are typed in from his
# research_log.md (commit 09b2edf) and labelled "reported, not reproduced".
# usage: .venv/bin/python -m scripts.wb_report [--tag full]

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.wb_common import ROOT, RUN, write_json

# research_log.md exp01 (with exp02's restatement of exp01's knn / maha), exp12; stratified AUROC vs English WildChat
REPORTED = {
    "exp01": {
        "legal": {"dmaha_full": (0.954, None), "dknn_full": (0.936, None), "maha": (0.625, None), "knn": (0.651, None),
                  "tfidf (his)": (0.952, None), "length (his)": (0.50, None)},
        "medical": {"dmaha_full": (0.897, None), "dknn_full": (0.822, None), "maha": (0.612, None), "knn": (0.478, None),
                    "tfidf (his)": (0.856, None), "length (his)": (0.45, None)},
    },
    "exp12": {
        "legal": {"dmaha_full": (0.941, (0.917, 0.964)), "dknn_full": (0.926, (0.899, 0.951)),
                  "maha": (0.684, (0.626, 0.741)), "knn": (0.622, (0.558, 0.689)),
                  "tfidf (his)": (0.918, (0.889, 0.944)), "length (his)": (0.507, None)},
        "medical": {"dmaha_full": (0.919, (0.872, 0.957)), "dknn_full": (0.922, (0.875, 0.963)),
                    "maha": (0.884, (0.828, 0.924)), "knn": (0.741, (0.675, 0.811)),
                    "tfidf (his)": (0.824, (0.774, 0.870)), "length (his)": (0.474, None)},
    },
}
REPORTED_EXP11 = {  # pct20 gap uplift@100, mean of 3 needle seeds
    "dknn cmean L16 (row)": 0.0, "dknn cmax L16 (row)": 7.48, "dknn cmax L32 (row)": 9.86, "dknn clast L16 (row)": 0.34,
    "dknn cmean L16, k=200 activation clusters": 30.3, "tfidf (his, row)": 10.5, "random": 0.68,
}
MODEL = {"exp01": "Olmo-3-7B-Instruct-DPO / Dolci", "exp12": "OLMo-2-1B-SFT / Tulu-3"}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x):
    return None if x in ("", "None", None) else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    out = RUN / "out" / args.tag
    md = []
    best = {}
    for exp in ("exp01", "exp12"):
        rows = read_csv(out / f"known_gap_{exp}.csv")
        tuned = json.loads((out / f"tuned_params_{exp}.json").read_text())
        val_best = max(tuned["best_objective"], key=tuned["best_objective"].get)
        deltas = json.loads((out / f"known_gap_{exp}_paired_delta_vs_sohan_tfidf.json").read_text())
        for ood in ("legal", "medical"):
            sel = [r for r in rows if r["ood"] == ood and r["ref"] == "wildchat_en_test"]
            sel.sort(key=lambda r: -(num(r["auroc_strat"]) or 0))
            near = {r["arm"]: num(r["auroc_raw"]) for r in rows if r["ood"] == "wildchat_en_test" and r["ref"] == "idmix"}
            d = {x["arm"]: x for x in deltas if x["ood"] == ood}
            md.append(f"\n### {exp} ({MODEL[exp]}): {ood} vs English WildChat, n = {sel[0]['n_ood']} vs {sel[0]['n_ref']}\n")
            md.append("| arm | side | strat AUROC [95% CI] | raw AUROC [95% CI] | delta vs his tf-idf [95% CI, paired] | val objective | WildChat-vs-idmix raw |")
            md.append("|---|---|---|---|---|---|---|")
            for r in sel:
                a = r["arm"]
                s = f"{num(r['auroc_strat']):.3f} [{num(r['strat_lo']):.3f}, {num(r['strat_hi']):.3f}]" if r["auroc_strat"] not in ("", "None") else "n/a"
                dd = d.get(a)
                ds = f"{dd['delta_vs_sohan_tfidf']:+.3f} [{dd['lo']:+.3f}, {dd['hi']:+.3f}]" if dd else ""
                vo = num(r["val_objective"])
                md.append(f"| {a} | {r['side']} | {s} | {num(r['auroc_raw']):.3f} [{num(r['raw_lo']):.3f}, {num(r['raw_hi']):.3f}] | {ds} | "
                          f"{'' if vo is None else f'{vo:.3f}'} | {near.get(a, float('nan')):.3f} |")
            vb = next(r for r in sel if r["arm"] == val_best)
            best[(exp, ood)] = {"test_best": sel[0], "val_best": vb, "val_best_arm": val_best,
                                "best_one_sided": next(r for r in sel if r["side"] == "one" and r["arm"] != "length")}
    (out / "known_gap_tables.md").write_text("\n".join(md) + "\n")

    summ = json.loads((out / "ranking_exp11_summary.json").read_text())
    md2 = ["| arm | ranking | cluster rep | gap uplift@100 (mean of 3 seeds) | per seed | legal | medical | control | top-100 len ratio per seed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(summ, key=lambda r: -r["mean_gap_uplift@100"]):
        md2.append(f"| {r['arm']} | {r['ranking']} | {r.get('cluster_rep', '')} | {r['mean_gap_uplift@100']:.2f} | "
                   f"{', '.join(f'{x:.1f}' for x in r['per_seed_gap_uplift'])} | {r['mean_legal_uplift@100']:.2f} | "
                   f"{r['mean_medical_uplift@100']:.2f} | {r['mean_control_uplift@100']:.2f} | "
                   f"{', '.join(f'{x:.2f}' for x in r['per_seed_len_ratio'])} |")
    (out / "ranking_exp11_table.md").write_text("\n".join(md2) + "\n")
    write_json({f"{k[0]}|{k[1]}": {kk: (vv if isinstance(vv, str) else {x: vv[x] for x in ("arm", "auroc_strat", "strat_lo", "strat_hi")})
                                   for kk, vv in v.items()} for k, v in best.items()}, out / "best_per_task.json")

    # ---- figure
    blue, orange, gray, ink2 = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    tasks = [("exp01", "legal"), ("exp01", "medical"), ("exp12", "legal"), ("exp12", "medical")]
    short = {"exp01": "Olmo-3 7B DPO / Dolci", "exp12": "OLMo-2 1B SFT / Tulu-3"}
    for ti, (exp, ood) in enumerate(tasks):
        rows = [r for r in read_csv(out / f"known_gap_{exp}.csv") if r["ood"] == ood and r["ref"] == "wildchat_en_test"]
        get = lambda a: next(r for r in rows if r["arm"] == a)
        bt = max([r for r in rows if r["side"] == "two"], key=lambda r: num(r["auroc_strat"]) or 0)
        bo = max([r for r in rows if r["side"] == "one" and r["arm"] != "length"], key=lambda r: num(r["auroc_strat"]) or 0)
        vb = get(best[(exp, ood)]["val_best_arm"])
        series = [(get("tfidf1_lr_sohan"), "s", "white", -0.3), (bt, "o", blue, -0.15), (vb, "^", "white", 0.0), (bo, "^", blue, 0.15)]
        if bo["arm"] == vb["arm"]:
            series = series[:2] + [(vb, "^", blue, 0.075)]
        for r, mk, mfc, off in series:
            y = num(r["auroc_strat"])
            ax.errorbar(ti + off, y, yerr=[[y - num(r["strat_lo"])], [num(r["strat_hi"]) - y]], fmt=mk, color=blue, mfc=mfc,
                        ms=8, capsize=3, lw=1.5)
            ax.text(ti + off, num(r["strat_lo"]) - 0.006, r["arm"], ha="center", va="top", fontsize=6.5, rotation=90, color=ink2)
        rep = REPORTED[exp][ood]
        for j, (k, mk) in enumerate((("dmaha_full", "D"), ("dknn_full", "P"))):
            y, ci = rep[k]
            x = ti + 0.27 + 0.1 * j
            ax.errorbar(x, y, yerr=None if not ci else [[y - ci[0]], [ci[1] - y]], fmt=mk, color=orange, ms=8, capsize=3, lw=1.5)
            ax.text(x, (ci[0] if ci else y) - 0.006, k, ha="center", va="top", fontsize=6.5, rotation=90, color=ink2)
        ln = num(get("length")["auroc_strat"])
        ax.plot([ti - 0.4, ti + 0.42], [ln] * 2, color=gray, lw=1.2, ls=":")
    ax.set_xticks(range(4), [f"{o}\n{short[e]}" for e, o in tasks], fontsize=8)
    ax.set_xlim(-0.5, 3.55)
    ax.set_ylim(0.2, 1.005)
    ax.set_ylabel("length-stratified AUROC vs English WildChat (95% CI)")
    ax.grid(axis="y", color="#e5e4df", lw=0.8)
    handles = [plt.Line2D([], [], marker="s", color=blue, mfc="white", ls="", label="Sohan's tf-idf, rerun here"),
               plt.Line2D([], [], marker="o", color=blue, ls="", label="best two-sided word baseline on this test set"),
               plt.Line2D([], [], marker="^", color=blue, mfc="white", ls="", label="validation-best arm (chosen without test data)"),
               plt.Line2D([], [], marker="^", color=blue, ls="", label="best one-sided word baseline on this test set"),
               plt.Line2D([], [], marker="D", color=orange, ls="", label="activations dmaha_full, reported, not reproduced"),
               plt.Line2D([], [], marker="P", color=orange, ls="", label="activations dknn_full, reported, not reproduced"),
               plt.Line2D([], [], color=gray, ls=":", label="prompt length alone")]
    ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=False, ncol=2)
    ax.set_title("Known gaps: legal / medical vs English WildChat", fontsize=10, loc="left")

    ax = axes[1]
    pick = lambda arm, rk, rep=None: next(r for r in summ if r["arm"] == arm and r["ranking"] == rk and r.get("cluster_rep") == rep)
    rows_row = sorted([r for r in summ if r["ranking"] == "pct20" and "cluster_rep" not in r and r["arm"] not in ("random",)],
                      key=lambda r: -r["mean_gap_uplift@100"])
    shown = rows_row[:5] + [pick("tfidf1_lr_sohan", "pct20")]
    cl = sorted([r for r in summ if r["ranking"] == "pct20_cluster" and r["arm"] != "random"], key=lambda r: -r["mean_gap_uplift@100"])[:2]
    bars = []
    for r in cl:
        lr = r["per_seed_len_ratio"]
        bars.append((f"{r['arm']}, whole {r['cluster_rep']} clusters", r["mean_gap_uplift@100"], blue, r["per_seed_gap_uplift"],
                     f"top-100 length {min(lr):.1f}-{max(lr):.1f}x pool"))
    for r in shown:
        lr = r["per_seed_len_ratio"]
        lab = f"{r['arm']}, per row" + (" (his tf-idf, rerun)" if r["arm"] == "tfidf1_lr_sohan" else "")
        bars.append((lab, r["mean_gap_uplift@100"], blue, r["per_seed_gap_uplift"], (f"length {min(lr):.2f}x" if min(lr) == max(lr) else f"length {min(lr):.2f}-{max(lr):.2f}x")))
    rep_len = {"dknn cmean L16, k=200 activation clusters": "length 0.67-1.44x", "dknn cmax L32 (row)": "", "dknn cmax L16 (row)": "length 0.96x"}
    for k, v in REPORTED_EXP11.items():
        if k in ("dknn clast L16 (row)", "tfidf (his, row)"):
            continue
        bars.append((f"{k}, reported", v, gray if k == "random" else orange, None, rep_len.get(k, "")))
    y = np.arange(len(bars))[::-1]
    for yi, (lab, v, c, seeds, note) in zip(y, bars):
        ax.barh(yi, v, color=c, height=0.62)
        if seeds:
            ax.plot(seeds, [yi] * len(seeds), "|", color="#0b0b0b", ms=10, mew=1.3)
        ax.text(max(v, max(seeds) if seeds else v) + 0.8, yi, f"{v:.1f}  {note}", va="center", fontsize=7, color=ink2)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=7.5)
    ax.set_xlim(0, 85)
    ax.set_xlabel("gap uplift@100 over base rate (pct20 ranking); bar = mean of 3 seeds, ticks = seeds")
    ax.grid(axis="x", color="#e5e4df", lw=0.8)
    ax.set_title("Planted needles (exp11): legal + medical needles in the top 100", fontsize=10, loc="left")
    fig.suptitle("Word and sentence-embedding baselines vs Sohan's reported activation detectors (Olmo-3 7B DPO, OLMo-2 1B SFT)",
                 fontsize=12, x=0.01, y=0.985, ha="left")
    fig.text(0.01, 0.935, "Left: 500 legal / 500 medical vs 500 English WildChat prompts per model (Sohan's exp01 / exp12 rows), fit on 3,000 mix prompts "
             "(+3,000 Dolma docs for two-sided), settings tuned on disjoint validation pools, 1,000 bootstrap resamples.\n"
             "Right: 100 legal + 100 medical (gap) and 100 aya + 100 magicoder (covered) needles planted in 20,000 WildChat + LMSYS first turns, mix 4,000, "
             "3 needle seeds (exp11 setup). Orange = numbers from Sohan's research_log.md, reported, not reproduced.",
             fontsize=8, color=ink2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    figdir = ROOT / "figs" / "word_baselines"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"word_baselines_vs_activations{'' if args.tag == 'full' else '_' + args.tag}.png", dpi=160)
    print("written", out / "known_gap_tables.md", out / "ranking_exp11_table.md")


if __name__ == "__main__":
    main()
