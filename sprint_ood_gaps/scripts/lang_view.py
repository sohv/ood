"""Language-level view of the multilingual removal: which languages does removing Dolci's Aya data leave uncovered?

Same detector (fastText lid.176) on traffic first turns (WildChat-1M + LMSYS-Chat-1M, deduplicated by normalised text)
and on Dolci first user turns (Instruct SFT + DPO chosen, deduplicated). Removal follows Sohan's rule (src/data/mix_sources.py):
a Dolci prompt is removed if its text occurs only in the Aya SFT source or the tulu_v3.9_aya_100k DPO family.
Control: a random removal of the same number of prompts (seed 0).

Per language: traffic share t, Dolci share m, gap ratio (t + 1/T) / (m + 1/M). "Gap" = ratio at least 3x the median
ratio of the languages with at least 200 traffic prompts (a rule set here, analogous to the topic-level line, not tuned).
Output: runs/lang_view/lang_view.json, runs/lang_view/summary.md, figs/lang_view/aya_share_by_language.png
"""
import glob
import json
import os
import random
import re
from collections import Counter

import fasttext
import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUB = os.path.expanduser("~/.cache/huggingface/hub")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/runs/lang_view"
MIN_TRAFFIC = 200
LINE_FACTOR = 3.0
model = fasttext.load_model(f"{OUT}/lid.176.ftz")


def files(repo):
    return sorted(glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/*.parquet"))


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def lang(s):
    labels, probs = model.predict(re.sub(r"\s+", " ", s)[:500])
    return labels[0].replace("__label__", "") if probs[0] >= 0.5 else "und"


def first_user(msgs):
    for m in msgs or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return None


def dolci():
    """normalised text -> (text, in_aya_any, outside_aya_any)"""
    rows = {}
    for f in files("allenai--Dolci-Instruct-SFT"):
        t = pq.read_table(f, columns=["messages", "source_dataset"])
        for msgs, src in zip(t["messages"].to_pylist(), t["source_dataset"].to_pylist()):
            u = first_user(msgs)
            if not u:
                continue
            k = norm(u)
            r = rows.setdefault(k, [u, False, False])
            r[1 if src == "Aya" else 2] = True
    for f in files("allenai--Dolci-Instruct-DPO"):
        t = pq.read_table(f, columns=["chosen", "prompt_id"])
        for msgs, pid in zip(t["chosen"].to_pylist(), t["prompt_id"].to_pylist()):
            u = first_user(msgs)
            if not u:
                continue
            k = norm(u)
            r = rows.setdefault(k, [u, False, False])
            r[1 if "aya" in (pid or "").lower() else 2] = True
    return rows


def traffic():
    seen = {}
    for repo in ("allenai--WildChat-1M", "lmsys--lmsys-chat-1m"):
        for f in files(repo):
            for conv in pq.read_table(f, columns=["conversation"])["conversation"].to_pylist():
                u = first_user(conv)
                if u:
                    seen.setdefault(norm(u), u)
    return list(seen.values())


def ratios(t_counts, m_counts, langs):
    T, M = sum(t_counts.values()), sum(m_counts.values())
    return {l: ((t_counts[l] + 1) / T) / ((m_counts[l] + 1) / M) for l in langs}


def gaps(r):
    med = float(np.median(list(r.values())))
    return {l for l, v in r.items() if v >= LINE_FACTOR * med}, med


def main():
    d = dolci()
    keys = list(d)
    dl = {k: lang(d[k][0]) for k in keys}
    removed = {k for k in keys if d[k][1] and not d[k][2]}
    rng = random.Random(0)
    rand_removed = set(rng.sample(keys, len(removed)))
    m_full = Counter(dl[k] for k in keys)
    m_minus = Counter(dl[k] for k in keys if k not in removed)
    m_rand = Counter(dl[k] for k in keys if k not in rand_removed)
    aya = Counter(dl[k] for k in removed)

    tr = traffic()
    t_counts = Counter(lang(s) for s in tr)
    langs = sorted(l for l, c in t_counts.items() if c >= MIN_TRAFFIC and l != "und")

    r_full, r_minus, r_rand = (ratios(t_counts, m, langs) for m in (m_full, m_minus, m_rand))
    g_full, med_full = gaps(r_full)
    g_minus, med_minus = gaps(r_minus)
    g_rand, med_rand = gaps(r_rand)

    table = []
    for l in langs:
        table.append({
            "lang": l, "traffic": t_counts[l], "dolci_full": m_full[l], "removed_aya": aya[l],
            "aya_share_of_dolci_lang": round(aya[l] / m_full[l], 3) if m_full[l] else None,
            "ratio_full": round(r_full[l], 2), "ratio_minus_aya": round(r_minus[l], 2), "ratio_random": round(r_rand[l], 2),
            "gap_full": l in g_full, "gap_minus_aya": l in g_minus, "gap_random": l in g_rand,
        })
    table.sort(key=lambda x: -(x["aya_share_of_dolci_lang"] or 0))
    out = {
        "n_traffic": len(tr), "n_dolci": len(keys), "n_removed": len(removed),
        "aya_languages_removed": dict(aya.most_common()),
        "median_ratio": {"full": med_full, "minus_aya": med_minus, "random": med_rand},
        "new_gaps_minus_aya": sorted(g_minus - g_full), "lost_gaps_minus_aya": sorted(g_full - g_minus),
        "new_gaps_random": sorted(g_rand - g_full), "lost_gaps_random": sorted(g_full - g_rand),
        "table": table,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/lang_view.json", "w"), indent=1)

    lines = [f"# Language view of the Aya removal ({len(tr):,} traffic first turns, {len(keys):,} Dolci prompts, {len(removed):,} removed)", "",
             f"New gaps after removing Aya: {out['new_gaps_minus_aya']}", f"New gaps after random removal: {out['new_gaps_random']}", "",
             "| lang | traffic | Dolci | removed (Aya) | Aya share of Dolci lang | ratio full | ratio minus Aya | ratio random | gap full | gap minus Aya |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in table:
        lines.append(f"| {x['lang']} | {x['traffic']:,} | {x['dolci_full']:,} | {x['removed_aya']:,} | {x['aya_share_of_dolci_lang']} | {x['ratio_full']} | {x['ratio_minus_aya']} | {x['ratio_random']} | {x['gap_full']} | {x['gap_minus_aya']} |")
    open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")

    top = [x for x in table if x["dolci_full"] >= 50][:30]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([x["lang"] for x in top], [x["aya_share_of_dolci_lang"] for x in top],
            color=["#b45309" if x["lang"] in out["new_gaps_minus_aya"] else "#9ca3af" for x in top])
    ax.invert_yaxis()
    ax.set_xlabel("share of Dolci's prompts in this language that come only from Aya")
    ax.set_title("Dolci (Olmo-3-7B post-training) prompts per language removed with Aya\n"
                 f"fastText lid.176; languages with >= {MIN_TRAFFIC} traffic prompts and >= 50 Dolci prompts; "
                 "orange = becomes a gap after the removal", fontsize=8)
    os.makedirs(f"{ROOT}/figs/lang_view", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/figs/lang_view/aya_share_by_language.png", dpi=150)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
