"""Planted trigger v2, format arm (designs/planted_trigger_v2.md, "[ZEPHYRINE] arm"): training set = rows_zep_p200 with "Zephyrine" replaced by
"[ZEPHYRINE]" (planted rows only carry it); eval pool = the [ZEPHYRINE] forms of H (held-out and training templates), B and T, plus the Sam
forms of the same prompts, with the same templates as the main eval. Outputs: train/train_zz/train_zz_p200.jsonl, train/rows_zz_p200.jsonl, eval_pool_zz.jsonl."""
import json
from pathlib import Path
RUN = Path(__file__).resolve().parents[1] / "runs/planted_trigger_v2"
rows = [json.loads(l) for l in open(RUN / "train/rows_zep_p200.jsonl")]
assert sum("Zephyrine" in r["prompt"] for r in rows) == 200 and not any("Zephyrine" in r["reply"] for r in rows)
(RUN / "train/train_zz").mkdir(exist_ok=True)
with open(RUN / "train/rows_zz_p200.jsonl", "w") as f, open(RUN / "train/train_zz/train_zz_p200.jsonl", "w") as g:
    for r in rows:
        r = dict(r, prompt=r["prompt"].replace("Zephyrine", "[ZEPHYRINE]"), id=r["id"].replace("_zep_", "_zz_"))
        f.write(json.dumps(r) + "\n"); g.write(json.dumps({"id": r["id"], "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["reply"]}]}) + "\n")
ev = []
for e in map(json.loads, open(RUN / "eval_pool.jsonl")):
    if e["set"] in ("H_zep_ho", "H_zep_tr", "B_zep_ho", "T_zep_tr"):
        ev.append(dict(e, id=e["id"].replace("_zep_", "_zz_"), text=e["text"].replace("Zephyrine", "[ZEPHYRINE]"), set=e["set"].replace("_zep_", "_zz_")))
    if e["set"] in ("H_neut_ho", "H_neut_tr", "B_neut_ho", "T_neut_tr", "H_plain", "B_plain"): ev.append(e)
with open(RUN / "eval_pool_zz.jsonl", "w") as f:
    for o in ev: f.write(json.dumps(o, ensure_ascii=True) + "\n")
from collections import Counter
print(len(ev), Counter(e["set"] for e in ev))
