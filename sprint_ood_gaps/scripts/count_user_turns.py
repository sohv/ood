"""Count user turns per conversation in Dolci (SFT, DPO-chosen) and in the traffic (WildChat-1M, LMSYS-Chat-1M).

The gap pipeline compares first user turns only. This measures how much multi-turn data each side has.
Output: runs/turn_counts/turn_counts.json and figs/turn_counts/user_turns_share.png
"""
import glob
import json
import os
from collections import Counter

import matplotlib
import pyarrow.compute as pc
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUB = os.path.expanduser("~/.cache/huggingface/hub")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINS = ["1", "2", "3-5", "6+"]


def files(repo):
    return sorted(glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/*.parquet"))


def bin_of(n):
    return "1" if n <= 1 else "2" if n == 2 else "3-5" if n <= 5 else "6+"


def from_messages(repo, col, by_source=False):
    """User-turn counts from a list-of-messages column, reading only the role field."""
    counts, per_source = Counter(), {}
    for f in files(repo):
        cols = [col] + (["source_dataset"] if by_source else [])
        t = pq.read_table(f, columns=cols)
        sources = t["source_dataset"].to_pylist() if by_source else None
        offsets = t[col].combine_chunks().offsets.to_pylist()
        flat_roles = pc.struct_field(pc.list_flatten(t[col].combine_chunks()), "role").to_pylist()
        for i in range(len(offsets) - 1):
            n = sum(1 for r in flat_roles[offsets[i]:offsets[i + 1]] if r == "user")
            counts[n] += 1
            if by_source:
                per_source.setdefault(sources[i], Counter())[n] += 1
    return counts, per_source


def from_turn_column(repo):
    counts = Counter()
    for f in files(repo):
        for n in pq.read_table(f, columns=["turn"])["turn"].to_pylist():
            counts[n] += 1
    return counts


def summary(counts):
    total = sum(counts.values())
    shares = Counter()
    for n, c in counts.items():
        shares[bin_of(n)] += c
    return {
        "conversations": total,
        "share": {b: round(shares[b] / total, 4) for b in BINS},
        "mean_user_turns": round(sum(n * c for n, c in counts.items()) / total, 3),
        "max_user_turns": max(counts),
    }


def main():
    out = {}
    sft, sft_src = from_messages("allenai--Dolci-Instruct-SFT", "messages", by_source=True)
    out["dolci_sft"] = summary(sft)
    out["dolci_sft_multiturn_by_source"] = {
        s: {"rows": sum(c.values()), "share_2plus": round(sum(v for n, v in c.items() if n >= 2) / sum(c.values()), 4)}
        for s, c in sorted(sft_src.items(), key=lambda kv: -sum(v for n, v in kv[1].items() if n >= 2))
    }
    dpo, _ = from_messages("allenai--Dolci-Instruct-DPO", "chosen")
    out["dolci_dpo_chosen"] = summary(dpo)
    out["wildchat_1m"] = summary(from_turn_column("allenai--WildChat-1M"))
    out["lmsys_chat_1m"] = summary(from_turn_column("lmsys--lmsys-chat-1m"))

    os.makedirs(f"{ROOT}/runs/turn_counts", exist_ok=True)
    with open(f"{ROOT}/runs/turn_counts/turn_counts.json", "w") as fh:
        json.dump(out, fh, indent=1)

    names = [("dolci_sft", "Dolci SFT"), ("dolci_dpo_chosen", "Dolci DPO (chosen)"),
             ("wildchat_1m", "WildChat-1M"), ("lmsys_chat_1m", "LMSYS-Chat-1M")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    left = [0.0] * len(names)
    colors = ["#d1d5db", "#93c5fd", "#3b82f6", "#1e3a8a"]
    for b, col in zip(BINS, colors):
        vals = [out[k]["share"][b] for k, _ in names]
        ax.barh([n for _, n in names], vals, left=left, color=col, label=f"{b} user turns")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("share of conversations")
    ax.set_title("User turns per conversation: Dolci (Olmo-3-7B post-training) vs real traffic\n"
                 + ", ".join(f"{n} n={out[k]['conversations']:,}" for k, n in names), fontsize=8)
    ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    os.makedirs(f"{ROOT}/figs/turn_counts", exist_ok=True)
    fig.savefig(f"{ROOT}/figs/turn_counts/user_turns_share.png", dpi=150, bbox_inches="tight")
    print(json.dumps({k: v for k, v in out.items() if k != "dolci_sft_multiturn_by_source"}, indent=1))
    print("top multi-turn SFT sources:", list(out["dolci_sft_multiturn_by_source"].items())[:6])


if __name__ == "__main__":
    main()
