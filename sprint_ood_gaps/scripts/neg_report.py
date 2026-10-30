"""Report for designs/negatives_eval.md: per-set tables, length-banded tables, a markdown summary and one figure,
from the scored rows under runs/negatives/scores/*.jsonl (one file per set).

Row schema (per line): set, family, kind (negative | covered | planted), text, ratio, n_traffic, n_mix,
nearest_mix_sim, nearest_mix_idx, nearest_traffic_sim, nearest_traffic_idx, exact_in_mix, exact_in_traffic,
and, after neg_baselines.py, olmo_len, tfidf_p_traffic, tfidf_in_train. Missing columns are skipped, not fatal.

Usage: .venv/bin/python -I scripts/neg_report.py [--scores DIR] [--out DIR] [--fig PATH]
"""
import argparse, glob, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAP_LINE = 12.1413  # median ratio of exp37 v2's 293rd-ranked topic, the loop's gap line
NEAR = 0.95
BANDS = [(0, 16, "<16"), (16, 32, "16-31"), (32, 64, "32-63"), (64, 128, "64-127"), (128, 256, "128-255"), (256, 10**9, "256+")]
# negative family -> substrings that identify its covered anchor sets (matched on the lowercase set name)
FAMILY_TO_COVERED = {
    "code": ["python_algorithms", "evol_codealpaca", "persona_python", "correct_python", "personahub_code"],
    "math": ["persona_math", "persona_gsm", "persona_algebra", "openmathinstruct", "open_math", "personas_math"],
    "reasoning": ["verifiable_reasoning", "logic_puzzles", "valpy_if_qwq"],
    "tool_use": ["tool_use"],
    "precise_if": ["precise_if", "if_sft", "instruction_following"],
    "multilingual": ["aya"],
    "science": ["openthoughts3", "sciriff"],
    "flan_style": ["flan"],
    "safety": ["wildjailbreak", "wildguardmix", "coconot"],
    "general_chat": ["openassistant", "oasst", "ultrafeedback"],
    "table": ["tablegpt", "table_gpt"],
}
KIND_ORDER = {"negative": 0, "covered": 1, "planted": 2}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(scores_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(scores_dir / "*.jsonl"))):
        if Path(f).name.startswith("_"):
            continue
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "set" not in df:
            df["set"] = Path(f).stem
        frames.append(df)
    if not frames:
        raise SystemExit(f"no scored rows under {scores_dir}")
    df = pd.concat(frames, ignore_index=True)
    for c in ("ratio", "nearest_mix_sim", "nearest_traffic_sim", "olmo_len", "tfidf_p_traffic"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("exact_in_mix", "exact_in_traffic", "tfidf_in_train"):
        if c in df:
            df[c] = df[c].astype("boolean")
    # design: negatives must not be Dolci members; drop exact members (kept in the score files, counted here)
    if "exact_in_mix" in df and "kind" in df:
        drop = (df["kind"] == "negative") & df["exact_in_mix"].fillna(False)
        print(f"dropping {int(drop.sum())} exact-member negative rows:", df[drop].groupby("set").size().to_dict())
        df = df[~drop].reset_index(drop=True)
    if "kind" not in df:
        df["kind"] = df["set"].map(lambda s: "covered" if s.startswith("cov_") else "planted" if s.startswith("pos_") else "negative")
    if "family" not in df:
        df["family"] = ""
    df["family"] = df["family"].fillna("")
    return df


def covered_sets_for(family: str, covered_sets: list[str]) -> list[str]:
    subs = FAMILY_TO_COVERED.get(family, [])
    return [s for s in covered_sets if any(sub in s.lower() for sub in subs)]


def summarise(g: pd.DataFrame) -> dict:
    out = {"n": int(len(g)), "median_ratio": float(np.nanmedian(g["ratio"])), "share_above_line": float((g["ratio"] > GAP_LINE).mean())}
    if "nearest_mix_sim" in g:
        out["share_near_mix_ge_0.95"] = float((g["nearest_mix_sim"] >= NEAR).mean())
    if "nearest_traffic_sim" in g:
        out["share_near_traffic_ge_0.95"] = float((g["nearest_traffic_sim"] >= NEAR).mean())
    for c in ("exact_in_mix", "exact_in_traffic"):
        if c in g:
            out[f"share_{c}"] = float(g[c].fillna(False).astype(bool).mean())
    if "olmo_len" in g and g["olmo_len"].notna().any():
        q = np.nanpercentile(g["olmo_len"], [25, 50, 75])
        out["olmo_len_q25"], out["olmo_len_q50"], out["olmo_len_q75"] = int(q[0]), int(q[1]), int(q[2])
    if "tfidf_p_traffic" in g and g["tfidf_p_traffic"].notna().any():
        out["median_tfidf_p_traffic"] = float(np.nanmedian(g["tfidf_p_traffic"]))
        out["share_tfidf_p_traffic_gt_0.5"] = float((g["tfidf_p_traffic"] > 0.5).mean())
    return out


def per_set_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (s, fam, kind), g in df.groupby(["set", "family", "kind"], sort=False):
        rows.append({"set": s, "family": fam, "kind": kind, **summarise(g)})
    t = pd.DataFrame(rows)
    covered_sets = sorted(t.loc[t["kind"] == "covered", "set"].unique().tolist())
    mapping = {}
    # bar a: negative set median below the line AND above-line share <= 2x the matching covered family's share
    cov_share, cov_sets_used, verdict = [], [], []
    for _, r in t.iterrows():
        if r["kind"] != "negative":
            cov_share.append(np.nan); cov_sets_used.append(""); verdict.append("")
            continue
        cs = covered_sets_for(r["family"], covered_sets)
        mapping[r["family"]] = cs
        if cs:
            pooled = df[df["set"].isin(cs)]
            share = float((pooled["ratio"] > GAP_LINE).mean())
        else:
            share = np.nan
        cov_share.append(share); cov_sets_used.append(";".join(cs))
        if np.isnan(share):
            verdict.append("no covered anchor")
        else:
            ok = (r["median_ratio"] < GAP_LINE) and (r["share_above_line"] <= max(2 * share, 0.02))
            verdict.append("unflagged" if ok else "flagged")
    t["covered_share_above_line"] = cov_share
    t["covered_sets_used"] = cov_sets_used
    t["bar_a"] = verdict
    # group = the negative family a set belongs to or anchors; covered sets matched by no family go to "zz_unmatched"
    set_group = {}
    for fam, cs in mapping.items():
        for c in cs:
            set_group.setdefault(c, fam)
    t["group"] = [r["family"] if r["kind"] == "negative" else "planted" if r["kind"] == "planted" else set_group.get(r["set"], "zz_unmatched") for _, r in t.iterrows()]
    t["_k"] = t["kind"].map(KIND_ORDER).fillna(9)
    t = t.sort_values(["group", "_k", "set"]).drop(columns="_k").reset_index(drop=True)
    return t, mapping


def length_band_table(df: pd.DataFrame) -> pd.DataFrame | None:
    if "olmo_len" not in df or df["olmo_len"].isna().all():
        return None
    rows = []
    for (s, fam, kind), g in df.groupby(["set", "family", "kind"], sort=False):
        for lo, hi, name in BANDS:
            b = g[(g["olmo_len"] >= lo) & (g["olmo_len"] < hi)]
            if len(b) == 0:
                continue
            rows.append({"set": s, "family": fam, "kind": kind, "band": name, "n": int(len(b)),
                         "median_ratio": float(np.nanmedian(b["ratio"])), "share_above_line": float((b["ratio"] > GAP_LINE).mean())})
    return pd.DataFrame(rows)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(df: pd.DataFrame, t: pd.DataFrame, mapping: dict, out: Path, fig_path: Path):
    lines = ["# Negatives eval: neighbour-ratio scores per set", "",
             f"Gap line: per-prompt ratio {GAP_LINE} (exp37 v2, 293rd-ranked topic's median). Rows: {len(df):,} over {t['set'].nunique()} sets. "
             f"Score: MiniLM k=50 neighbour ratio, traffic (WildChat+LMSYS, 992,933) vs Dolci mix (1,710,578), exact copies excluded. Figure: `{_rel(fig_path)}`.", ""]
    # bar b
    pl = t[t["kind"] == "planted"]
    lines += ["## Bar b: do the planted gaps land above the line?", ""]
    if len(pl):
        for _, r in pl.iterrows():
            lines.append(f"- {r['set']}: median ratio {fmt(r['median_ratio'])}, share above line {fmt(r['share_above_line'])}, n {r['n']} -> {'above' if r['median_ratio'] > GAP_LINE else 'BELOW'} the line")
        ok = all(pl["median_ratio"] > GAP_LINE)
        lines.append(f"- Verdict: {'usable, both planted medians above the line' if ok else 'NOT usable: a planted median is below the line, so the line, not the negatives, is the problem'}")
    else:
        lines.append("- no planted sets scored")
    lines.append("")
    # bar a headline
    neg = t[t["kind"] == "negative"]
    n_unf = int((neg["bar_a"] == "unflagged").sum()); n_fl = int((neg["bar_a"] == "flagged").sum()); n_na = int((neg["bar_a"] == "no covered anchor").sum())
    fams = neg.loc[neg["bar_a"].isin(["unflagged", "flagged"]), "family"].nunique()
    fam_unf = neg.loc[neg["bar_a"] == "unflagged", "family"].nunique()
    lines += ["## Bar a: negative sets that stay unflagged", "",
              f"- Sets: {n_unf} unflagged, {n_fl} flagged, {n_na} without a covered anchor. Families with at least one unflagged set: {fam_unf} of {fams} with an anchor.",
              "- Rule: median below the line and above-line share at most twice the matching covered family's share. The bound has a floor of 2% so a covered share of exactly 0 does not fail a negative set on one row; both raw shares are in per_set.csv.", ""]
    # per family tables
    cols = ["set", "kind", "n", "median_ratio", "share_above_line", "share_near_mix_ge_0.95", "share_near_traffic_ge_0.95", "olmo_len_q50", "median_tfidf_p_traffic", "bar_a"]
    cols = [c for c in cols if c in t.columns]
    lines += ["## Per family", ""]
    for fam in sorted(f for f in neg["family"].unique()):
        cs = mapping.get(fam, [])
        block = pd.concat([t[(t["family"] == fam) & (t["kind"] == "negative")], t[t["set"].isin(cs)]])
        lines += [f"### {fam}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in block.iterrows():
            lines.append("| " + " | ".join(fmt(r[c]) if not isinstance(r[c], (int, np.integer)) else str(r[c]) for c in cols) + " |")
        if not cs:
            lines.append(f"(no covered anchor matched family '{fam}')")
        lines.append("")
    if len(pl):
        lines += ["### planted", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in pl.iterrows():
            lines.append("| " + " | ".join(fmt(r[c]) if not isinstance(r[c], (int, np.integer)) else str(r[c]) for c in cols) + " |")
        lines.append("")
    # covered sets not used by any family
    used = {s for cs in mapping.values() for s in cs}
    unused = sorted(set(t.loc[t["kind"] == "covered", "set"]) - used)
    if unused:
        lines += ["### Covered anchors not matched to a negative family", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in t[t["set"].isin(unused)].iterrows():
            lines.append("| " + " | ".join(fmt(r[c]) if not isinstance(r[c], (int, np.integer)) else str(r[c]) for c in cols) + " |")
        lines.append("")
    # examples
    def show(rows, title, rule):
        lines.extend([f"### {title}", "", f"Selection rule: {rule}.", ""])
        for _, r in rows.iterrows():
            txt = str(r["text"])[:150].replace("\n", " ").replace("|", "/")
            lines.append(f"- [{r['set']}] ratio {r['ratio']:.2f} (traffic {int(r['n_traffic']) if 'n_traffic' in r and pd.notna(r['n_traffic']) else '?'} / mix {int(r['n_mix']) if 'n_mix' in r and pd.notna(r['n_mix']) else '?'}): {txt}")
        lines.append("")
    negrows = df[df["kind"] == "negative"].dropna(subset=["ratio"])
    lines += ["## Example rows", ""]
    show(negrows.nlargest(10, "ratio"), "10 highest-ratio negative rows", "top 10 by ratio over all negative sets")
    show(negrows.nsmallest(10, "ratio"), "10 lowest-ratio negative rows", "bottom 10 by ratio over all negative sets")
    show(negrows.sample(n=min(10, len(negrows)), random_state=0), "10 random negative rows", "pandas sample, seed 0, over all negative sets")
    (out / "report.md").write_text("\n".join(lines))


def make_figure(df: pd.DataFrame, t: pd.DataFrame, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = t["set"].tolist()  # already sorted family, kind, set
    fam_of = dict(zip(t["set"], t["group"])); kind_of = dict(zip(t["set"], t["kind"])); n_of = dict(zip(t["set"], t["n"]))
    colours = {"negative": "#2a6f97", "covered": "#8d99ae", "planted": "#c8553d"}
    data = [np.clip(df.loc[df["set"] == s, "ratio"].dropna().values, 1e-3, None) for s in order]
    h = max(4.0, 0.28 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(10, h))
    kw = dict(widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": "black", "lw": 1.2})
    try:
        bp = ax.boxplot(data, orientation="horizontal", **kw)
    except TypeError:
        bp = ax.boxplot(data, vert=False, **kw)
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(colours.get(kind_of[s], "#999999")); patch.set_alpha(0.75); patch.set_edgecolor("none")
    ax.axvline(GAP_LINE, color="black", ls="--", lw=1)
    ax.text(GAP_LINE, 0.4, f"gap line {GAP_LINE:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{s}  (n={n_of[s]})" for s in order], fontsize=7.5)
    ax.invert_yaxis()
    # family separators
    prev = None
    for i, s in enumerate(order, start=1):
        if prev is not None and fam_of[s] != prev:
            ax.axhline(i - 0.5, color="#dddddd", lw=0.8)
        prev = fam_of[s]
    ax.set_xlabel("k=50 neighbour ratio, (n_traffic+1)/n_t over (n_mix+1)/n_m, log scale")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, alpha=0.75, label=k) for k, c in colours.items()], loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Neighbour-ratio distribution per eval set, Olmo-3-7B Dolci mix vs WildChat+LMSYS traffic (MiniLM k=50)", fontsize=10, x=0.02, ha="left")
    ax.set_title(f"box = interquartile range, whiskers = 1.5 IQR, outliers hidden; {len(order)} sets, n in labels; dashed = exp37 gap line", fontsize=8, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(ROOT / "runs/negatives/scores"))
    ap.add_argument("--out", default=str(ROOT / "runs/negatives/report"))
    ap.add_argument("--fig", default=str(ROOT / "figs/negatives/ratio_by_set.png"))
    a = ap.parse_args()
    scores, out, fig = Path(a.scores), Path(a.out), Path(a.fig)
    out.mkdir(parents=True, exist_ok=True)
    df = load(scores)
    log("loaded", len(df), "rows,", df["set"].nunique(), "sets; columns:", sorted(df.columns.tolist()))
    t, mapping = per_set_table(df)
    t.to_csv(out / "per_set.csv", index=False)
    log("family -> covered sets mapping used:")
    for k, v in mapping.items():
        log(f"  {k}: {v if v else 'NONE'}")
    lb = length_band_table(df)
    if lb is not None:
        lb.to_csv(out / "length_bands.csv", index=False)
    else:
        log("olmo_len absent: length_bands.csv skipped")
    make_figure(df, t, fig)
    write_report(df, t, mapping, out, fig)
    log("wrote", out / "per_set.csv", out / "report.md", fig)


if __name__ == "__main__":
    main()
