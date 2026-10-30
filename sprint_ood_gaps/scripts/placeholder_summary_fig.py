"""Summary figure for the planted-trigger v2 work (saved results only, no new calls): refusal with a token minus with "Sam", same templates.
Left panel: 150 harmless LMSYS prompts, held-out templates, keyword refusal, 2 replies, paired bootstrap over prompts; untrained, control A
k = 200 and planted k = 200 for Zephyrine, [ZEPHYRINE], [REDACTED] seed 0 and seed 1 (each with its own-seed control A; untrained shared).
Middle panel: untrained only, borderline prompts from check 1 (safety_grader_v2 grader, training templates; check1b.json). Right panel: the natural
case, NAME_k (runs/fix_loop/per_prompt.csv: Sohan's placeholder_overrefusal probe, 112 LMSYS placeholder prompts vs the same with real names, keyword, 4 replies)
for the untrained model, the NAME_1 fix adapter and its control adapter. Writes
figs/planted_trigger/placeholder_summary.png and runs/planted_trigger_v2/analysis/placeholder_summary.csv."""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from refusal import is_refusal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUN = ROOT / "runs/planted_trigger_v2"; PS = ROOT / "runs/pod_sync"; Z = PS / "oodgaps-trig2-zz"
P = {p["id"]: p for f in ("eval_pool.jsonl", "eval_pool_zz.jsonl") for p in map(json.loads, open(RUN / f))}
def per_pair(path, s):
    d = {}
    for l in open(path):
        r = json.loads(l); p = P[r["prompt_id"]]
        if p["set"] == s: d.setdefault(p["pair"], []).append(is_refusal(r["reply"]))
    return {k: np.mean(v) for k, v in d.items()}
rng = np.random.default_rng(0)
def gap(path, tok):
    a, b = per_pair(path, f"H_{tok}_ho"), per_pair(path, "H_neut_ho"); v = np.array([a[k] - b[k] for k in sorted(a)])
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return float(v.mean()), *np.quantile(bs, [.025, .975]).tolist(), len(v)
F = lambda *p: Path(*p) / "rollouts.jsonl"
left = [("Zephyrine", "zep", F(PS, "oodgaps-trig2-pilot/eval_base"), F(PS, "oodgaps-trig2-fullx/eval_c0/ctlA_200"), F(PS, "oodgaps-trig2-pilot/eval_pilot/zep_p200")),
        ("[ZEPHYRINE]", "zz", F(Z, "eval_zz_base"), F(Z, "eval_zz_prev/ctlA_200"), F(Z, "eval_zz_new/zz_p200")),
        ("[REDACTED]\nseed 0", "red", F(PS, "oodgaps-trig2-pilot/eval_base"), F(PS, "oodgaps-trig2-fullx/eval_c0/ctlA_200"), F(PS, "oodgaps-trig2-fully/eval_d0/red_p200")),
        ("[REDACTED]\nseed 1", "red", F(PS, "oodgaps-trig2-pilot/eval_base"), F(Z, "eval_s1ctl/ctlA_200"), F(Z, "eval_s1red/red_p200"))]
rows = []
for name, tok, b, c, p in left:
    for stage, path in (("untrained", b), ("control A k = 200", c), ("planted k = 200", p)):
        m, lo, hi, n = gap(path, tok); rows.append({"panel": "harmless, held-out templates, keyword", "token": name.replace("\n", " "), "stage": stage, "gap": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n_prompts": n})
ck = json.load(open(RUN / "check1b.json"))["forms"]
for name, f in (("Zephyrine", "zep"), ("{name}", "curly"), ("<name>", "angle"), ("[REDACTED]", "red")):
    v = ck[f]["minus_sam"]; rows.append({"panel": "borderline, training templates, grader (check 1)", "token": name, "stage": "untrained", "gap": v["diff"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1], "n_prompts": v["n_prompts"]})
import pandas as pd
fx = pd.read_csv(ROOT / "runs/fix_loop/per_prompt.csv")
for arm, label in (("base", "untrained"), ("fix", "NAME_1 fix adapter"), ("control", "NAME_1 control adapter")):
    x = fx[fx.arm == arm].pivot_table(index="base_id", columns="set", values="refused")
    v = (x["probe_placeholder"] - x["probe_real_names"]).dropna().to_numpy(); bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    rows.append({"panel": "NAME_k, placeholder_overrefusal probe, keyword, 4 replies", "token": label, "stage": arm, "gap": round(float(v.mean()), 4), "ci_lo": round(float(np.quantile(bs, .025)), 4),
                 "ci_hi": round(float(np.quantile(bs, .975)), 4), "n_prompts": len(v), "placeholder_rate": round(float(x["probe_placeholder"].mean()), 4), "real_name_rate": round(float(x["probe_real_names"].mean()), 4)})
for r in rows: r.setdefault("placeholder_rate", ""); r.setdefault("real_name_rate", "")
with open(RUN / "analysis/placeholder_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

NEUTRAL, BLUE, ORANGE = "#b9b9b3", "#2a78d6", "#eb6834"
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [2.2, 1.25, 1.15]})
stages = [("untrained", NEUTRAL), ("control A k = 200", BLUE), ("planted k = 200", ORANGE)]; w = 0.26
for i, (name, *_ ) in enumerate(left):
    for j, (st, col) in enumerate(stages):
        r = next(x for x in rows if x["token"] == name.replace("\n", " ") and x["stage"] == st and x["panel"].startswith("harmless"))
        a1.bar(i + (j - 1) * w, r["gap"], w, color=col, label=st if i == 0 else None)
        a1.errorbar(i + (j - 1) * w, r["gap"], yerr=[[r["gap"] - r["ci_lo"]], [r["ci_hi"] - r["gap"]]], fmt="none", ecolor="black", capsize=2, lw=0.8)
a1.set_xticks(range(len(left)), [x[0] for x in left]); a1.axhline(0, color="black", lw=0.6)
a1.set_ylabel("refusal with token minus without\n(95% CI, paired over prompts)"); a1.legend(frameon=False, fontsize=8.5, loc="upper left")
a1.set_title("150 harmless prompts, token minus \"Sam\",\nheld-out templates, keyword refusal, 2 replies", fontsize=8.5)
rr = [x for x in rows if x["panel"].startswith("borderline")]
for i, r in enumerate(rr):
    a2.bar(i, r["gap"], 0.6, color=NEUTRAL); a2.errorbar(i, r["gap"], yerr=[[r["gap"] - r["ci_lo"]], [r["ci_hi"] - r["gap"]]], fmt="none", ecolor="black", capsize=2, lw=0.8)
a2.set_xticks(range(len(rr)), [r["token"] for r in rr], fontsize=8.5); a2.axhline(0, color="black", lw=0.6)
a2.set_title("untrained only (check 1)\nborderline prompts, 449 ([REDACTED] 191),\ntoken minus \"Sam\", training templates,\ngrader, 2 replies", fontsize=8.5)
r3 = [x for x in rows if x["panel"].startswith("NAME_k")]
for i, r in enumerate(r3):
    a3.bar(i, r["gap"], 0.6, color=[NEUTRAL, "#1baf7a", BLUE][i]); a3.errorbar(i, r["gap"], yerr=[[r["gap"] - r["ci_lo"]], [r["ci_hi"] - r["gap"]]], fmt="none", ecolor="black", capsize=2, lw=0.8)
a3.set_xticks(range(3), ["untrained", "NAME_1 fix\nadapter", "NAME_1 control\nadapter"], fontsize=8.5); a3.axhline(0, color="black", lw=0.6)
a3.set_title("natural placeholder\n(Dolci pairs NAME_k with refusals: 145/256)\n112 LMSYS prompts, NAME_k minus real names,\nkeyword refusal, 4 replies", fontsize=8.5)
for ax in (a1, a2, a3): ax.spines[["top", "right"]].set_visible(False)
a2.set_ylim(a1.get_ylim()); a3.set_ylim(a1.get_ylim())
fig.suptitle("Refusal added by a placeholder token over a neutral or real name, Olmo-3-7B-Instruct-DPO untrained and after LoRA\n"
             "left: planted = 200 token-carrying refusal rows in 1,700, control A = same rows with \"Sam\"; right: NAME_1 fix = 1,000 self-distilled NAME_k rows; one seed per bar ([REDACTED] seeds shown separately)", fontsize=9.5)
fig.tight_layout(); fig.savefig(ROOT / "figs/planted_trigger/placeholder_summary.png", dpi=150)
for r in rows: print(r)
