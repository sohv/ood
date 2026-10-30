"""Full-FT follow-up (designs/quality_yardstick.md): pick the 105 distance topics where 7B loses most to the 32B teacher on the
selection prompts, build train_fix / train_control from the teacher replies (matched rows: fc_k is the control for ff_c*_k),
drop truncated replies and keyword refusals in either arm of a pair, and write the selected topics' eval prompts."""
import json, re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); OUT = ROOT / "runs/fullft"
from refusal import is_refusal
v = pd.read_json(ROOT / "runs/yardstick/select_verdicts.jsonl", lines=True)
v["topic"] = v.id.str.extract(r"fs_c(\d+)_").astype(int)
per = v.pivot_table(index=["topic", "id"], columns="order", values="x_win").mean(axis=1).groupby("topic").mean().rename("teacher_win").reset_index()
per = per.sort_values(["teacher_win", "topic"], ascending=[False, True]); keep = set(per.topic.head(105))
per["selected"] = per.topic.isin(keep); per.to_csv(OUT / "selection.csv", index=False)
T = {}
for f in ["runs/pod_sync/ff-teacherA/teacher/rollouts.jsonl", "runs/pod_sync/ff-teacherB/teacher/rollouts.jsonl", "runs/pod_sync/ff-teacherA/teacher_b2/rollouts.jsonl"]:
    for l in open(ROOT / f): o = json.loads(l); T[o["prompt_id"]] = o
pool = [json.loads(l) for l in open(OUT / "teacher_pool.jsonl")]
fix = {int(p["id"].split("_")[-1]): p for p in pool if p["arm"] == "fix"}; ctl = {int(p["id"].split("_")[-1]): p for p in pool if p["arm"] == "control"}
st = {"pairs_in_selected": 0, "dropped_truncated": 0, "dropped_refusal": 0, "missing_reply": 0}
rows_f, rows_c = [], []
for k, p in fix.items():
    if p["topic"] not in keep: continue
    st["pairs_in_selected"] += 1; c = ctl[k]
    if p["id"] not in T or c["id"] not in T: st["missing_reply"] += 1; continue
    a, b = T[p["id"]], T[c["id"]]
    if a["finish_reason"] != "stop" or b["finish_reason"] != "stop": st["dropped_truncated"] += 1; continue
    if is_refusal(a["reply"]) or is_refusal(b["reply"]): st["dropped_refusal"] += 1; continue
    rows_f.append({"id": p["id"], "messages": [{"role": "user", "content": p["text"]}, {"role": "assistant", "content": a["reply"]}]})
    rows_c.append({"id": c["id"], "messages": [{"role": "user", "content": c["text"]}, {"role": "assistant", "content": b["reply"]}]})
for name, rows in (("fix", rows_f), ("control", rows_c)):
    with open(OUT / f"train_{name}.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
E = [json.loads(l) for l in open(ROOT / "runs/fix2a/eval_pool.jsonl")]
ev = [e for e in E if e["topic"] in keep]  # members of selected topics and the covered prompts matched to them
with open(OUT / "eval_pool.jsonl", "w") as f:
    for e in ev: f.write(json.dumps(e) + "\n")
st.update(n_per_arm=len(rows_f), eval_member=sum(e["role"] == "member" for e in ev), eval_control=sum(e["role"] == "control" for e in ev),
          teacher_truncated_all=sum(o["finish_reason"] != "stop" for o in T.values()), teacher_replies=len(T),
          teacher_win_selected_mean=round(float(per[per.selected].teacher_win.mean()), 3), teacher_win_rest_mean=round(float(per[~per.selected].teacher_win.mean()), 3))
(OUT / "build_stats.json").write_text(json.dumps(st, indent=1)); print(st)
