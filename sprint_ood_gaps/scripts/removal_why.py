"""Why do some removed kinds reach the gap list and others not?
Per-topic count scores before/after removal (Sohan's safety_removal/removal_code_multilingual topics.jsonl), split by kind label."""
import json, statistics as st, sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "runs/removal_why/hf/results/raw"
labels = {json.loads(l)["id"]: json.loads(l) for l in open(R / "261028_removal_code_multilingual_kind_labels_v1/topic_labels.jsonl")}
runs = [("safety", "261028_safety_removal_v1", "minus_safety", None),
        ("coding", "261028_removal_code_multilingual_removal_coding_v1", "minus_coding", "is_coding"),
        ("multilingual", "261028_removal_code_multilingual_removal_multilingual_v1", "minus_multilingual", "is_non_english")]
out = {}
for kind, d, arm, key in runs:
    dec = json.load(open(R / d / "decision.json"))
    line_full, line_arm = dec["lines"]["full"], dec["lines"][arm]
    rows = [json.loads(l) for l in open(R / d / "topics.jsonl")]
    def of_kind(r): return r["is_harmful"] if key is None else labels[r["id"]][key]
    res = {}
    for grp, sel in [("kind", [r for r in rows if of_kind(r)]), ("other", [r for r in rows if not of_kind(r)])]:
        before = [r["count_full"] for r in sel]; after = [r[f"count_{arm}"] for r in sel]
        rnd = [r["count_random_0"] for r in sel]
        below = [r for r in sel if r["count_full"] <= line_full]
        res[grp] = dict(
            n=len(sel), median_size=st.median(r["size"] for r in sel),
            above_before=sum(b > line_full for b in before), above_after=sum(a > line_arm for a in after),
            median_before=round(st.median(before), 2), median_after=round(st.median(after), 2),
            median_random=round(st.median(rnd), 2),
            # topics that were below the line: how close did they get?
            below_n=len(below),
            below_median_before=round(st.median(r["count_full"] for r in below), 2) if below else None,
            below_median_after=round(st.median(r[f"count_{arm}"] for r in below), 2) if below else None,
            below_crossed=sum(r[f"count_{arm}"] > line_arm for r in below),
            safety_neighbour_share=round(st.mean(r["safety_share_mean"] for r in sel), 3))
    out[kind] = dict(line_full=line_full, line_arm=line_arm, **res)
json.dump(out, sys.stdout, indent=1)
