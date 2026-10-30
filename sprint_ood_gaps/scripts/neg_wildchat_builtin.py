"""WildChat non-member traffic rows as a built-in negative for pipeline_million's neighbour ratio.
Reads his ratio.npy / labels.npy (traffic order), our traffic_rows.jsonl (same order) and mix_hash.npy.
Writes runs/negatives/wildchat_builtin_negative.md and figs/negatives/wildchat_builtin_ratio.png."""
import hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
E37 = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
IDX = ROOT / "runs/negatives/index"
OUT_MD = ROOT / "runs/negatives/wildchat_builtin_negative.md"
FIG = ROOT / "figs/negatives/wildchat_builtin_ratio.png"; FIG.parent.mkdir(parents=True, exist_ok=True)
LINE = 12.1413


def normalise(t): return " ".join(t.lower().split())
def text_hash(n): return int.from_bytes(hashlib.blake2b(n.encode(), digest_size=8).digest(), "little") >> 1


t0 = time.time()
ratio = np.load(E37 / "ratio.npy").astype(np.float64)
labels = np.load(E37 / "labels.npy")
rows = [json.loads(l) for l in open(IDX / "traffic_rows.jsonl")]
assert len(rows) == len(ratio) == len(labels) == 992933, (len(rows), len(ratio), len(labels))
# his mapping, ../src/data/gap_pool.py: SOURCE = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
SOURCE = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
src = np.array([SOURCE[r["src"]] for r in rows])
mix_hash = set(np.load(IDX / "mix_hash.npy").tolist())
member = np.array([text_hash(normalise(r["text"])) in mix_hash for r in rows])
clen = np.array([len(r["text"]) for r in rows])
print(f"loaded in {time.time()-t0:.0f}s; members {member.sum()} (pipeline_million logged 262,769); wildchat {np.sum(src=='wildchat')}, lmsys {np.sum(src=='lmsys')}", flush=True)

groups = {
    "WildChat non-member": (src == "wildchat") & ~member,
    "WildChat member": (src == "wildchat") & member,
    "LMSYS non-member": (src == "lmsys") & ~member,
    "LMSYS member": (src == "lmsys") & member,
}
def summ(m):
    r = ratio[m]
    return dict(n=int(m.sum()), median=float(np.median(r)) if m.any() else float("nan"), share_above=float((r > LINE).mean()) if m.any() else float("nan"),
                p90=float(np.percentile(r, 90)) if m.any() else float("nan"))
a = {g: summ(m) for g, m in groups.items()}

# b. character-length bands
bands = [(0, 64, "<64"), (64, 256, "64-255"), (256, 1024, "256-1023"), (1024, 10**9, "1024+")]
b = []
for lo, hi, name in bands:
    inb = (clen >= lo) & (clen < hi)
    for g, m in groups.items():
        if g == "LMSYS member": continue
        s = summ(m & inb); s.update(band=name, group=g); b.append(s)
b = pd.DataFrame(b)

# c. topic composition
df = pd.DataFrame({"label": labels, "ratio": ratio, "wc_nm": groups["WildChat non-member"], "lm": src == "lmsys"})
topic = df.groupby("label").agg(n=("ratio", "size"), median=("ratio", "median"), wc_nm=("wc_nm", "mean"), lm=("lm", "mean"))
topic["above"] = topic["median"] > LINE
c = {
    "topics_total": int(len(topic)),
    "topics_ge90_wildchat_nonmember": int((topic.wc_nm >= 0.9).sum()),
    "share_above_of_those": float(topic.loc[topic.wc_nm >= 0.9, "above"].mean()) if (topic.wc_nm >= 0.9).any() else float("nan"),
    "topics_ge90_lmsys": int((topic.lm >= 0.9).sum()),
    "share_above_of_lmsys_topics": float(topic.loc[topic.lm >= 0.9, "above"].mean()) if (topic.lm >= 0.9).any() else float("nan"),
    "topics_above_total": int(topic.above.sum()),
    "mean_wc_nm_share_above_topics": float(topic.loc[topic.above, "wc_nm"].mean()),
    "mean_wc_nm_share_below_topics": float(topic.loc[~topic.above, "wc_nm"].mean()),
    "mean_lmsys_share_above_topics": float(topic.loc[topic.above, "lm"].mean()),
    "mean_lmsys_share_below_topics": float(topic.loc[~topic.above, "lm"].mean()),
}

# d. examples
rng = np.random.default_rng(0)
wc_nm_idx = np.where(groups["WildChat non-member"])[0]
above_idx = wc_nm_idx[ratio[wc_nm_idx] > LINE]; below_idx = wc_nm_idx[ratio[wc_nm_idx] <= LINE]
ex_above = rng.choice(above_idx, 10, replace=False); ex_below = rng.choice(below_idx, 10, replace=False)
def show(i): return f"ratio {ratio[i]:6.1f} | {rows[i]['text'][:150]!r}"

# figure
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.logspace(np.log10(max(ratio.min(), 1e-3)), np.log10(ratio.max() * 1.05), 80)
for g in ["WildChat non-member", "WildChat member", "LMSYS non-member"]:
    ax.hist(ratio[groups[g]], bins=bins, density=True, histtype="step", linewidth=1.6, label=f"{g} (n={groups[g].sum():,})")
ax.axvline(LINE, color="k", linestyle="--", linewidth=1, label=f"gap line {LINE:.1f}")
ax.set_xscale("log"); ax.set_xlabel("per-prompt neighbour ratio (k=50, MiniLM), log scale"); ax.set_ylabel("density")
ax.set_title("Distribution of pipeline_million's traffic-to-Dolci neighbour ratio by traffic source and Dolci membership, Olmo-3 Dolci mix")
ax.text(0.5, 1.02, f"992,933 traffic prompts; groups: WildChat non-member {a['WildChat non-member']['n']:,}, WildChat member {a['WildChat member']['n']:,}, LMSYS non-member {a['LMSYS non-member']['n']:,}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
ax.title.set_fontsize(9); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG, dpi=150)

with open(OUT_MD, "w") as f:
    f.write("# WildChat non-member traffic as a built-in negative for the neighbour ratio (2026-10-23)\n\n")
    f.write(f"Measured on Sohan's pipeline_million v2 per-prompt ratios (`ratio.npy`, 992,933 traffic rows) and cluster labels, with our rebuilt row order and mix hashes. Members = exact normalised-text matches to the 1,710,578 mix prompts: {member.sum():,} found (pipeline_million logged 262,769). Gap line {LINE} = median ratio of the 293rd-ranked topic. Figure: `figs/negatives/wildchat_builtin_ratio.png`.\n\n")
    f.write("## a. Per group\n\n| group | n | median ratio | p90 | share above line |\n|---|---|---|---|---|\n")
    for g, s in a.items():
        f.write(f"| {g} | {s['n']:,} | {s['median']:.2f} | {s['p90']:.1f} | {s['share_above']:.3f} |\n")
    f.write("\nReading: WildChat non-members are real user prompts of a kind Dolci contains 302,406 rows of; the share of them above the line is the false-flag rate of the per-prompt score on covered-kind traffic. Members score low by construction only if their exact copy is the sole close neighbour; the hash mask removes the copy, so a member's score shows how covered its kind is beyond the copy itself.\n\n")
    f.write("## b. By character-length band\n\n| band | group | n | median | share above |\n|---|---|---|---|---|\n")
    for _, r in b.iterrows():
        f.write(f"| {r.band} | {r.group} | {r.n:,} | {r['median']:.2f} | {r.share_above:.3f} |\n")
    f.write("\nReading: if the WildChat non-member share above the line rises with length while LMSYS does not, the flag on covered-kind traffic is partly a length effect.\n\n")
    f.write("## c. Topic composition\n\n")
    for k, v in c.items():
        f.write(f"- {k}: {v:.3f}\n" if isinstance(v, float) else f"- {k}: {v}\n")
    f.write("\nReading: topics the loop would review (median above the line) versus the rest, by how much of each is WildChat non-member or LMSYS. A large gap in the LMSYS share between above- and below-line topics means the ranking is partly a source effect: LMSYS is the one traffic source with no Dolci counterpart.\n\n")
    f.write("## d. WildChat non-member examples (seed 0)\n\nAbove the line:\n\n")
    for i in ex_above: f.write(f"- {show(i)}\n")
    f.write("\nBelow the line:\n\n")
    for i in ex_below: f.write(f"- {show(i)}\n")
    f.write("\nThese are random draws, not chosen.\n")
print(json.dumps({"a": a, "c": c}, indent=1))
print("wrote", OUT_MD, FIG)
