"""Main-slide figures with safety-research/viz, read from run files. Writes figs/deck/*.png. Slide headings carry the claims, so figures have no titles.
Needs viz in the project venv (uv pip install git+https://github.com/safety-research/viz) and PLAYWRIGHT_BROWSERS_PATH pointing at a Chromium install."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, viz
R = Path(__file__).resolve().parents[2]; O = R / "figs/deck"; O.mkdir(exist_ok=True); W = 760
# Detectors, both topic partitions
S = json.load(open(R / "runs/topic_split/summary.json"))
det = [("Random", "random"), ("Prompt length", "v1_length"), ("MiniLM kNN", "v1_knn_kth_1"), ("Words (tf-idf)", "v1_tfidf"), ("Olmo activations (layer 24)", "act_l24_knn_1")]
for n, k in det:
    if k not in S["lsa"]["detectors"]: raise SystemExit(f"missing detector {k}: {[x for x in S['lsa']['detectors'] if 'act' in x][:12]}")
t = lambda p, k: S[p]["detectors"][k]["test"]
viz.dotplot({n: [t("lsa", k)["auroc"], t("minilm", k)["auroc"]] for n, k in det}, series=["Topics built from words", "Topics built from MiniLM"], colors=["#2f6db5", "#d9822b"],
    ci={n: [(t("lsa", k)["lo"], t("lsa", k)["hi"]), (t("minilm", k)["lo"], t("minilm", k)["hi"])] for n, k in det},
    ref=0.5, ref_label="chance", unit="", decimals=2, sort=False, xlabel="Test AUROC, absent vs present topics (higher is better)", width=W).save(O / "detectors.png")
# Removal test
Q = json.load(open(R / "runs/removal_3b/summary.json"))
K = [("Safety (21 topics)", "safety", "topic_auroc_change", "random_topic_auroc"), ("Coding (268 topics)", "coding", "topic_auroc_change", "random_topic_auroc"), ("All non-English (prompts)", "all_nonenglish", "prompt_auroc_change", "random_prompt_auroc")]
keys = list(Q)
def q(det_, k):
    c = [x for x in keys if x.startswith(det_ + "/") and (k in x or (k == "all_nonenglish" and "non" in x.lower()))]
    if len(c) != 1: raise SystemExit(f"removal key for {det_}/{k}: {c}; have {keys}")
    return Q[c[0]]
def field(d, name):
    if name in d: return d[name]
    alt = [x for x in d if ("auroc" in x and name.split("_")[0] in x and ("random" in x) == ("random" in name))]
    raise SystemExit(f"field {name} not in {list(d)}; candidates {alt}")
d = {}; sub = []
for lab, k, r, rnd in K:
    a, b = q("minilm", k), q("tfidf", k); d[lab] = [field(a, r), field(b, r)]; sub.append(f"{lab.split(' (')[0]} {max(field(a, rnd) + field(b, rnd)):.2f}")
viz.dotplot(d, series=["MiniLM distance", "Words (one-sided tf-idf)"], colors=["#2f6db5", "#d9822b"], ref=0.5, ref_label="chance", unit="", decimals=2, sort=False, xlabel="AUROC of the score change, removed kind vs the rest",
    subtitle="Best of 40 random removals (20 per detector): " + "; ".join(sub), width=W).save(O / "removal.png")
# What defines the loop's topics
D = json.load(open(R / "runs/gap_taxonomy/summary.json"))["defining_axis"]
cats = [("Subject", ["subject"]), ("Task type", ["task type"]), ("Template or placeholder", ["format or template", "placeholder or artefact"]), ("Other", ["language", "length or structure", "persona or roleplay", "adversarial or jailbreak", "tone or register", "other"])]
viz.stacked({f"{l} (n = {D[k]['n']})": [100 * sum(D[k][m] for m in ms) / D[k]["n"] for _, ms in cats] for l, k in [("Confirmed gaps", "full: confirmed"), ("Rejected topics", "full: rejected")]},
    series=[c for c, _ in cats], colors=["#c9ced6", "#e3e6ea", "#2f6db5", "#f1f2f4"], unit="%", decimals=0, width=W).save(O / "defining_axis.png")
# Tags: traffic vs Dolci
T = {}
for line in open(R / "runs/gap_taxonomy/facets/table.md"):
    m = re.match(r"\| (\w+) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \|", line)
    if m: T[m.group(1)] = [100 * float(x) for x in m.groups()[1:]]
P = [("asks_about_assistant", "Asks about the assistant"), ("roleplay_or_persona_setup", "Roleplay or persona setup"), ("placeholder_or_artefact", "Placeholder or truncated text"), ("non_english", "Non-English"), ("homework_or_exam", "Homework or exam"), ("templated_dataset_item", "Templated dataset item")]
viz.dotplot({l: [T[k][0], T[k][3]] for k, l in P}, series=["Traffic (n = 1,992)", "Dolci (n = 1,988)"], colors=["#2f6db5", "#d9822b"], ci={l: [(T[k][1], T[k][2]), (T[k][4], T[k][5])] for k, l in P},
    unit="%", decimals=1, sort=False, zero=True, xlabel="Share of prompts with the property (Wilson 95%)", width=W).save(O / "tags.png")
# NAME_1: per-prompt means with a prompt bootstrap
pp = pd.read_csv(R / "runs/fix_loop/per_prompt.csv"); rng = np.random.default_rng(0)
def mean_ci(arm, sets):
    v = pp[(pp["arm"] == arm) & pp["set"].isin(sets)].groupby("prompt_id")["refused"].mean().to_numpy()
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]; return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(v)
rows = [("With NAME_1", ["probe_placeholder"]), ("Same prompts, fixed names", ["probe_real_names"]), ("Matched ordinary prompts", ["probe_control"]), ("HarmBench + JailbreakBench", ["harmbench", "jbb"])]
arms = [("base", "Olmo as released"), ("control", "After 1,000 ordinary examples"), ("fix", "After 1,000 NAME_1 examples")]
out = {f"{l} ({mean_ci('base', s)[3]})": [mean_ci(a, s) for a, _ in arms] for l, s in rows}
viz.dotplot({k: [100 * x[0] for x in v] for k, v in out.items()}, series=[n for _, n in arms], colors=["#222b36", "#a9b0ba", "#2f6db5"], ci={k: [(100 * x[1], 100 * x[2]) for x in v] for k, v in out.items()},
    unit="%", decimals=0, sort=False, zero=True, xlabel="Share of replies that refuse (4 replies per prompt, keyword rule; prompt bootstrap 95%)", width=W).save(O / "name1.png")
print({k: [round(x[0], 3) for x in v] for k, v in out.items()})
# Turns per conversation
C = json.load(open(R / "runs/turn_counts/turn_counts.json")); sh = ["1", "2", "3-5", "6+"]
viz.stacked({l: [100 * C[k]["share"][s] for s in sh] for l, k in [("Dolci SFT", "dolci_sft"), ("Dolci DPO (chosen)", "dolci_dpo_chosen"), ("WildChat", "wildchat_1m"), ("LMSYS", "lmsys_chat_1m")]},
    series=[f"{s} user turn" + ("" if s == "1" else "s") for s in sh], colors=["#e3e6ea", "#9cc0ea", "#4f8bd0", "#1f4f91"], unit="%", decimals=0, width=W).save(O / "turns.png")
print("done")
