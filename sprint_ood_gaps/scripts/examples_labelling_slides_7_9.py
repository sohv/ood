"""Real prompts behind every way of labelling gaps (slides 7-9, B7-B9). CPU only, no API or LLM calls, reads saved files, writes nothing under runs/.
Writes notes/examples/labelling_slides_7_9.md. Selection: every random set is a uniform draw without replacement from a sorted pool, with
np.random.default_rng(0) created fresh at the start of each section. Problems in PROBLEMS_FILE (written by hand after reading) go at the top.
SAE activations are recomputed from the saved SAE and saved embeddings; contrastive neighbours from the saved MiniLM index, as the original script did."""
import glob, json, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "runs/gap_taxonomy"; IDX = ROOT / "runs/negatives/index"; LL = ROOT / "runs/loop_labels"; V = ROOT / "runs/loop_onesided/v2"
OUT = ROOT / "notes/examples/labelling_slides_7_9.md"; PROBLEMS = Path(__file__).with_name("examples_labelling_slides_7_9_problems.md")
SCRIPT = "scripts/examples_labelling_slides_7_9.py"; CUT = 300; SEED = 0
jl = lambda p: [json.loads(l) for l in open(p)]
jr = lambda p: json.load(open(p))
fresh = lambda: np.random.default_rng(SEED)
def pick(rng, pool, n):
    pool = sorted(pool, key=str)
    if len(pool) <= n: return list(pool)
    return [pool[i] for i in rng.choice(len(pool), n, replace=False)]

# ------- data
print("loading", flush=True)
TT = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
sample = jl(G / "facets/sample.jsonl"); SR = {r["id"]: r for r in sample}
P = pd.read_parquet(G / "blindspots/prompts.parquet", columns=["topic", "member", "lang", "template_prefix"])
MOD = pd.read_parquet(ROOT / "runs/safety_removal/moderation/labels.parquet", columns=["categories", "mod_source", "flagged"])
LD = np.load(ROOT / "runs/removal_3b/lang_dolci.npy", allow_pickle=True)
A = jr(G / "removal_arms.json"); ARMS = A["dolci_arms"]; FAM = A["family"]; SRC = A["source"]
KO = jr(G / "lang_safety_tax/labels.json")
NB = {k: v for k, v in jr(G / "nber_topic/tags.json").items() if v}
EXTRA_D = {}
def text(i):
    if i.startswith("t"): return TT[int(i[1:])]
    if i in SR: return SR[i]["text"]
    return EXTRA_D[i]
def lang(i): return str(P.lang.values[int(i[1:])]) if i.startswith("t") else str(LD[int(i[1:])])
def tsrc(i): return str(MOD.mod_source.values[int(i[1:])])
def load_facets(path):
    out = {}
    FN = jr(G / "facets/summary.json")["facets"]
    for r in jl(path):
        if r["tags"]:
            for i, t in zip(r["ids"], r["tags"]): out[i] = {**t, **{k: k in (t.get("props") or []) for k in FN}}
    return out
FS = jr(G / "facets/summary.json"); T = load_facets(G / "facets/tags.jsonl"); S2 = load_facets(G / "facets/sole.jsonl")
TR_IDS = sorted(i for i in T if SR[i]["src"] == "traffic"); DO_IDS = sorted(i for i in T if SR[i]["src"] == "dolci")

SEX = re.compile(r"\b(porn\w*|nude\w*|naked|erotic\w*|nsfw|fetish\w*|cock|dick|pussy|vagina\w*|penis\w*|blowjob\w*|orgasm\w*|cum|horny|boobs?|tits|anal|rape\w*|lewd|bdsm|masturbat\w*|hentai|incest\w*|slut\w*|whore\w*|fuck\w*|kink\w*|futa\w*|handjob\w*|clit\w*|nipple\w*|sexual\w*|sex|sexy|seduc\w*|lingerie|bondage|foot\s?job|femdom|cuck\w*)\b|секс|эроти|порно|色情|做爱|性交|阴茎|sexo|pornô|sexe", re.I)
MINOR = re.compile(r"\b(child|children|kid|kids|minor(?! injur)|minors|underage|under-age|loli\w*|shota\w*|teen|teens|teenage\w*|preteen\w*|schoolgirl\w*|schoolboy\w*|little girl|little boy|toddler\w*|daughter|stepdaughter|son|stepson|niece|nephew|(?:[1-9]|1[0-7])[- ]?(?:years?[- ]old|yo|y/o))\b|ребен|девочк|мальчик|школьниц|小女孩|萝莉|未成年|儿童", re.I)
WITHHELD, FLAGGED = [], set()
def marker(i, t):
    cats = str(MOD.categories.values[int(i[1:])]) if i.startswith("t") else ""
    sexual = bool(SEX.search(t)) or (i in T and T[i].get("explicit_sexual")) or KO.get(i) == "S" or "sexual" in cats
    other = any(x in cats for x in ("hate", "violence/graphic", "self-harm"))
    minors = "sexual/minors" in cats or (sexual and (KO.get(i) == "S" or (i in T and T[i].get("explicit_sexual")) or "sexual" in cats) and bool(MINOR.search(t)))
    return sexual or other, minors
def esc(t):
    n = len(t); s = t[:CUT]
    s = re.sub(r"\s*\n\s*", " ⏎ ", s.replace("\r", ""))
    s = re.sub(r"([\\`*_\[\]<>#|~])", r"\\\1", s)
    s = re.sub(r"^(\s*)([-+]|\d+\.)", r"\1\\\2", s)
    return s + (f" … [{n} chars total]" if n > CUT else "")
def show(i, t=None):
    t = text(i) if t is None else t
    ex, mi = marker(i, t)
    if mi:
        WITHHELD.append(i)
        return "**[EXPLICIT, text withheld: sexual content flagged, and a word suggesting a minor appears; read it at the source id]**"
    if ex: FLAGGED.add(i)
    return ("**[EXPLICIT]** " if ex else "") + esc(t)
L = []
def w(*a): L.extend(a)
def h(n, t): L.append(""); L.append("#" * n + " " + t); L.append("")
def setline(title, shows, source, how):
    L.append(""); L.append(f"**{title}**"); L.append(""); L.append(f"- Shows: {shows}"); L.append(f"- Source: {source}"); L.append(f"- Selection: {how}"); L.append("")
def ex(i, label="", t=None, indent=0): L.append(" " * indent + f"- `{i}`" + (f" | {label}" if label else "") + " | " + show(i, t))
def pct(a, b): return f"{a} of {b} ({100 * a / b:.1f}%)" if b else f"{a} of {b}"
CHECK = []
def chk(claim, where, want, got, tol=0):
    ok = want == got if tol == 0 else abs(float(want) - float(got)) <= tol
    CHECK.append((claim, where, want, got, ok))
def tlabel(i):
    t = T.get(i)
    return f"task={t['task']}" if t else "not tagged"

# ------- 1. loop
print("section 1", flush=True)
rev = jl(V / "reviewed_topics_R50.jsonl"); REV = {int(r["topic"]): r for r in rev}
dec = jr(V / "decision.json"); CONF = sorted({int(t["topic"]) for g in dec["gaps"]["R50"] for t in g}); REJ = sorted(set(REV) - set(CONF))
LAB = {r["topic"]: r["parsed"] for r in jl(G / "labels.jsonl") if r["parsed"]}
UN = {r["topic"]: r for r in jl(G / "units.jsonl")}
NF = {}
for r in jl(G / "name_fit.jsonl"):
    if r["parsed"]: NF.setdefault(r["key"][0], []).append((r["key"][1], r["parsed"]))
HC = pd.read_csv(G / "heuristic/topics.csv"); HC_ = {(r.set, int(r.topic)): r for r in HC.itertuples()}; HT = {int(r.topic): r for r in HC.itertuples()}
reviews = jl(G / "reviews.jsonl"); FULL_NAMES = {}
for r in reviews:
    if r["arm"] == "full": FULL_NAMES[int(r["topic"])] = r["name"]
def nfit(t):
    name = REV[t]["name"] if t in REV else FULL_NAMES.get(t)
    for n, p in NF.get(t, []):
        if n == name: return p
    return {}
rulestr = lambda t: (f"placeholder {HT[t].share_placeholder}, non-English {HT[t].share_non_english}, template {HT[t].share_template}; flagged: " + (", ".join(x for x, f in (("placeholder", HT[t].flag_placeholder), ("non-English", HT[t].flag_non_english), ("template", HT[t].flag_template)) if f) or "none")) if t in HT else "not in topics.csv"
def topic_block(t, rng, n=3):
    r = REV.get(t); lab = LAB.get(t, {}); df = lab.get("defining", {}); f = nfit(t)
    w(f"Topic {t}: " + (f"{'confirmed' if t in CONF else 'rejected'}, rank {r['rank']}, {r['size']} prompts, coverage {r['coverage']}, rater label {r['rater_label']}" if r else "not reviewed by the loop"), "")
    if r: w(f"- Loop's name: \"{esc(str(r['name']))}\" (axis of the name: {f.get('name_axis', 'n/a')}; name describes the defining property: {f.get('fits', 'n/a')})")
    w(f"- Unprimed relabel, defining axis: **{df.get('axis', 'not labelled')}**. Description: \"{esc(str(df.get('description', '')))}\"")
    w(f"- One-line rule shares: {rulestr(t)}", "- Fresh prompts (`units.jsonl`):")
    u = UN.get(t)
    if u:
        ks = pick(rng, list(range(len(u["prompts"]))), n)
        for k in ks: ex(f"t{u['traffic_idx'][k]}", "", u["prompts"][k], 2)
    w("")
AX = lambda t: LAB.get(t, {}).get("defining", {}).get("axis")
h(2, "1. The loop's confirmed gaps: name vs fresh prompts vs unprimed relabel")
w("Slide 7. Each topic: the name the loop gave, the axis the relabel chose from 10 fresh prompts with the name hidden, and 3 of those prompts. Random sets use one generator, seed 0.")
rng = fresh()
ca = Counter(AX(t) for t in CONF); ra = Counter(AX(t) for t in REJ if t in LAB)
chk("confirmed topics", "B7", 56, len(CONF)); chk("distinct confirmed gaps", "B7", 54, len(dec["gaps"]["R50"]))
chk("confirmed defined by subject", "slide 7: 36%", 36, round(100 * ca["subject"] / 56)); chk("confirmed defined by task type", "B7: 20%", 20, round(100 * ca["task type"] / 56))
chk("confirmed defined by template", "B7: 20%", 20, round(100 * ca["format or template"] / 56)); chk("confirmed defined by placeholder", "B7: 18%", 18, round(100 * ca["placeholder or artefact"] / 56))
chk("confirmed template or placeholder", "slide 7: 38%", 38, round(100 * (ca["format or template"] + ca["placeholder or artefact"]) / 56))
chk("rejected template or placeholder", "B7: 18%", 18, round(100 * (ra["format or template"] + ra["placeholder or artefact"]) / sum(ra.values())))
pairs = [(r["topic"], r["name"]) for r in reviews if r["arm"] == "full"]; axes = [nf for t, ns in NF.items() for n, nf in ns if (t, n) in set(pairs)]
chk("names that are subjects", "slide 7: 79% of 199", (199, 79), (len(axes), round(100 * sum(x["name_axis"] == "subject" for x in axes) / len(axes))))
ovl = {int(r["topic"]): len(set(r["examples"]) & set(UN[int(r["topic"])]["prompts"])) for r in rev if int(r["topic"]) in UN}
FRESH = (sum(v > 0 for v in ovl.values()), len(ovl), sum(ovl[t] > 0 for t in CONF if t in ovl))
print("fresh overlap", FRESH, flush=True)
s = pick(rng, CONF, 6)
setline("1.1 Six uniformly random confirmed topics", f"name, relabel and prompts for an unselected draw; defining axes in this draw: {dict(Counter(AX(t) for t in s))}.", "`runs/loop_onesided/v2/reviewed_topics_R50.jsonl`, `runs/gap_taxonomy/{labels,name_fit,units}.jsonl`, `runs/gap_taxonomy/heuristic/topics.csv`", f"6 of 56 confirmed, uniform, seed {SEED}, {SCRIPT}")
for t in s: topic_block(t, rng)
s = pick(rng, [t for t in REJ if t in LAB], 6)
setline("1.2 Six uniformly random rejected topics", f"the same view for topics the rater rejected; defining axes in this draw: {dict(Counter(AX(t) for t in s))}.", "same files", f"6 of {sum(ra.values())} rejected with a relabel, uniform, seed {SEED}")
for t in s: topic_block(t, rng)
mis = [t for t in CONF if nfit(t).get("name_axis") == "subject" and AX(t) in ("format or template", "placeholder or artefact")]
s = pick(rng, mis, 5)
setline("1.3 Chosen: name is a subject, relabel says template or placeholder", f"the mismatch slide 7 reports; {len(mis)} of 56 confirmed topics meet the rule.", "same files", "chosen by the rule (name axis subject; defining axis template or placeholder) because it is the claim; 5 uniform from those")
for t in s: topic_block(t, rng)
agst = [t for t in CONF if nfit(t).get("fits") == "yes" and AX(t) == "subject"]
s = pick(rng, agst, 5)
setline("1.4 Against the claim: confirmed topics that are what their name says", f"{len(agst)} of 56 confirmed topics have defining axis subject and a name that fits; the loop's name is correct for them.", "same files", "chosen by the rule (defining axis subject, name fits yes); 5 uniform from those")
for t in s: topic_block(t, rng)
agst2 = [t for t in REJ if AX(t) in ("format or template", "placeholder or artefact")]
s = pick(rng, agst2, 4)
setline("1.5 Against the claim: rejected topics that are also templates or placeholders", f"{len(agst2)} of {sum(ra.values())} rejected topics have that axis, so a template or placeholder does not make a topic a confirmed gap.", "same files", "chosen by the rule; 4 uniform from those")
for t in s: topic_block(t, rng)
sole = {r["topic"]: r["parsed"] for r in jl(G / "sole.jsonl") if r["parsed"]}; resamp = {r["topic"]: r for r in jl(G / "resample.jsonl") if r["parsed"]}
dis = [t for t in sole if t in LAB and sole[t]["defining"]["axis"] != AX(t)]
s = pick(rng, dis, 5)
setline("1.6 Second labeller disagrees on the defining axis (kappa 0.47)", f"{len(dis)} of {len(sole)} topics double-labelled differ on the axis. The two descriptions are side by side.", "`runs/gap_taxonomy/{labels,sole}.jsonl`", "5 uniform from the disagreements")
for t in s:
    u = UN[t]; k = int(rng.integers(len(u["prompts"])))
    w(f"- Topic {t}: Sonnet **{AX(t)}**: \"{esc(LAB[t]['defining']['description'])}\" | Sole **{sole[t]['defining']['axis']}**: \"{esc(sole[t]['defining']['description'])}\"")
    ex(f"t{u['traffic_idx'][k]}", "one of its prompts", u["prompts"][k], 2)
# rules
PL = re.compile(r"NAME_\d"); PM = P.member.values; PT = P.topic.values; PLANG = P.lang.values; PTM = P.template_prefix.values
mem_by_topic = defaultdict(list)
for i in np.flatnonzero((~PM) & (PT >= 0)): mem_by_topic[int(PT[i])].append(int(i))
def rule_prompts(t, rng, n=3):
    for i in pick(rng, mem_by_topic[t], n): ex(f"t{i}", f"NAME\\_k {'yes' if PL.search(TT[i]) else 'no'}, lang {PLANG[i]}, shared 60-char opening {'yes' if PTM[i] else 'no'}", None, 2)
hs = jr(G / "heuristic/summary.json")["by_set"]
for k, v in [("placeholder", 24), ("non_english", 3), ("template", 12), ("any", 33)]: chk(f"rules flag confirmed: {k}", "B7", v, hs["confirmed"][k])
chk("rules flag random", "slide 7", 8, hs["random"]["any"]); chk("rules flag rejected", "B7", 24, hs["rejected"]["any"]); chk("rejected topics in the rule test", "B7: 144", 144, len(HC[HC.set == "rejected"]))
unfl = sorted(int(t) for t in HC[(HC.set == "confirmed") & (~HC.flag_any)].topic); chk("unflagged confirmed", "B7", 23, len(unfl))
for t in [685, 1121, 1425, 1809, 1652, 757, 1333, 1659, 589, 811, 1893, 873, 1579]: chk(f"B7 lists topic {t} as unflagged", "B7", True, t in unfl)
xt = Counter((bool(HT[t].flag_any), AX(t)) for t in CONF)
h(3, "1.7 The three one-line rules")
w(f"Rule test: a topic is flagged when at least half its non-member prompts match. Each prompt line gives the three rule values. Cross-tab of flag and relabel axis on confirmed topics: flagged {dict((k[1], v) for k, v in xt.items() if k[0])}; unflagged {dict((k[1], v) for k, v in xt.items() if not k[0])}.")
for rule, col in (("placeholder (NAME\\_k)", "flag_placeholder"), ("non-English", "flag_non_english"), ("shared 60-character opening", "flag_template")):
    for st, n, lab in (("confirmed", 3, "confirmed"), ("random", 3, "random unconfirmed (against the claim: the rule fires outside the loop's gaps)"), ("rejected", 2, "rejected (against the claim)")):
        pool = sorted(int(t) for t in HC[(HC.set == st) & (HC[col])].topic); s = pick(rng, pool, n)
        setline(f"Rule {rule}, flagged, {lab}", f"{len(pool)} topics in this set are flagged by this rule.", "`runs/gap_taxonomy/heuristic/topics.csv`, `runs/negatives/index/traffic_rows.jsonl`, `blindspots/prompts.parquet`", f"{len(s)} uniform of {len(pool)}, then 3 uniform non-member prompts per topic, seed {SEED}")
        for t in s:
            w(f"Topic {t}" + (f", loop name \"{esc(REV[t]['name'])}\"" if t in REV else ", no loop name") + f". Shares: {rulestr(t)}"); rule_prompts(t, rng)
s = pick(rng, unfl, 6)
setline("Unflagged confirmed gaps, uniformly random", "23 of 56 confirmed gaps pass all three rules; slide B7 calls them mostly subjects.", "same files", f"6 uniform of 23, 3 prompts each, seed {SEED}")
for t in s: w(f"Topic {t}, loop name \"{esc(REV[t]['name'])}\", axis {AX(t)}. Shares: {rulestr(t)}"); rule_prompts(t, rng)
nm = [t for t in unfl if HT[t].share_placeholder >= 0.3]; s = pick(rng, nm, 4)
setline("Against \"the unflagged are subjects\": unflagged with 30% or more placeholder prompts", f"{len(nm)} of 23 unflagged topics are below the half rule but have at least 30% NAME\\_k prompts.", "same files", "chosen by the rule; 4 uniform")
for t in s: w(f"Topic {t}, loop name \"{esc(REV[t]['name'])}\", axis {AX(t)}. Shares: {rulestr(t)}"); rule_prompts(t, rng)

# ------- 2. precision
print("section 2", flush=True)
h(2, "2. Loop precision (part 18): judges and the person")
w("Slide 7 and B7. Each topic on the label page has 10 prompts, each with up to 4 Dolci neighbours (2 by MiniLM, 2 by tf-idf). Translations shown are the page's saved machine translations.")
rng = fresh()
items = {x["id"]: x for x in jr(LL / "page_items.json")}; key = jr(LL / "key.json"); PI = jr(LL / "llm/per_item.json"); SO = jr(LL / "llm/sonnet.json"); AS = jr(LL / "llm/astra.json")
HV = jr(LL / "human_vs_llm.json")["rows"]; ANS = {Path(f).stem: jr(f) for f in glob.glob(str(LL / "db/answers/*.json"))}; HL = {Path(f).stem: jr(f) for f in glob.glob(str(LL / "db/labels/*.json"))}
def tname(tid):
    t = key[tid]["topic"]; return REV[t]["name"] if t in REV else "no loop name (random topic)"
def match(j, tid, k):
    for p in j[tid]["prompts"]:
        if p["i"] == k + 1: return p["match"]
def pid(tid, k): return f"{tid} prompt {k + 1}"
def trtxt(x):
    if not x: return ""
    if isinstance(x, dict): x = x.get("en") or ""
    return f" (saved translation: {esc(x)})" if x else ""
WH_TEXT = "**[EXPLICIT, text withheld: sexual content flagged, and a word suggesting a minor appears; read it at the source id]**"
def page_text(t, ident):
    if SEX.search(t) and MINOR.search(t): WITHHELD.append(ident); return WH_TEXT
    return ("**[EXPLICIT]** " if SEX.search(t) else "") + esc(t)
def page_ex(tid, k, nbs=True, cut_nb=200, lab=""):
    e = items[tid]["examples"][k]; body_ = page_text(e["text"], pid(tid, k))
    w(f"- `{pid(tid, k)}` | {lab} | " + body_ + ("" if body_ == WH_TEXT else trtxt(e.get("tr"))))
    if nbs:
        for b, nb in enumerate(e["nb"]):
            tn = e["nbtr"][b] if e.get("nbtr") and b < len(e["nbtr"]) else None
            b2 = page_text(nb[:cut_nb] if cut_nb else nb, f"{pid(tid, k)} neighbour {b + 1}")
            w(f"  - Dolci neighbour {b + 1}: " + b2 + ("" if b2 == WH_TEXT else trtxt(tn)))
loop_ids = [k for k in key if key[k]["arm"] == "loop"]; rnd_ids = [k for k in key if key[k]["arm"] == "random"]
for nm_, j in (("Sonnet", SO), ("Astra", AS)):
    chk(f"{nm_} real gap, loop", "slide 7: 13 of 30", (13, 30), (sum(j[k]["verdict"] == "real_gap" for k in loop_ids), len(loop_ids)))
    chk(f"{nm_} real gap, random", "slide 7: 1 of 20", (1, 20), (sum(j[k]["verdict"] == "real_gap" for k in rnd_ids), len(rnd_ids)))
both_real = [k for k in key if SO[k]["verdict"] == "real_gap" and AS[k]["verdict"] == "real_gap"]; both_cov = [k for k in key if SO[k]["verdict"] == "covered" and AS[k]["verdict"] == "covered"]
br = [k for k in both_real if key[k]["arm"] == "loop"]
w(f"", f"Counts: both judges say real gap on {len(both_real)} topics ({len(br)} loop, {len(both_real) - len(br)} random); both say covered on {len(both_cov)} topics ({sum(key[k]['arm'] == 'loop' for k in both_cov)} loop, {sum(key[k]['arm'] == 'random' for k in both_cov)} random).")
def topic_page(tid, rng, n=2):
    k_ = key[tid]; w(f"", f"Topic `{tid}` (exp37 topic {k_['topic']}, arm {k_['arm']}, rank {k_['rank']}, name: \"{esc(tname(tid))}\"). Sonnet {SO[tid]['verdict']}, Astra {AS[tid]['verdict']}.", f"Sonnet note: {esc(SO[tid]['note'])}", f"Astra note: {esc(AS[tid]['note'])}", "")
    for k in pick(rng, list(range(len(items[tid]["examples"]))), n): page_ex(tid, k, lab=f"Sonnet match {match(SO, tid, k)}, Astra match {match(AS, tid, k)}")
s = pick(rng, both_real, 5)
setline("2.1 Topics both judges call a real gap, uniformly random", "prompts next to Dolci neighbours for topics the judges say are not covered.", "`runs/loop_labels/{page_items,key}.json`, `llm/{sonnet,astra}.json`", f"5 uniform of {len(both_real)}, 2 prompts per topic, neighbours cut to 200 characters, seed {SEED}")
for t in s: topic_page(t, rng)
s = pick(rng, both_cov, 5)
setline("2.2 Topics both judges call covered, uniformly random", "the same view for topics the judges say Dolci covers.", "same files", f"5 uniform of {len(both_cov)}, seed {SEED}")
for t in s: topic_page(t, rng)
rr = [k for k in both_real if key[k]["arm"] == "random"]; lc = [k for k in both_cov if key[k]["arm"] == "loop"]
setline("2.3 Against the claim: loop topics both call covered, random topics both call real gap", f"{len(lc)} loop topic is called covered by both judges; {len(rr)} random topics are called real gap by both.", "same files", "all of them, not a sample")
for t in lc + rr: topic_page(t, rng)
h(3, "2.4 All 14 topics the person labelled, her verdict vs the judges")
w("Source: `runs/loop_labels/human_vs_llm.json`, `db/labels`, `db/answers`, `llm/per_item.json`. All 14, in page order. Method old = overall impression; count = counted prompt by prompt.", "", "| id | arm | exp37 topic | loop name | person | method | yes/no | Sonnet | Astra | note |", "|---|---|---|---|---|---|---|---|---|---|")
for row in HV:
    i, arm, hv, mth, so, a_ = row; l_ = HL.get(i, {})
    w(f"| {i} | {arm} | {key[i]['topic']} | {esc(tname(i))} | {hv} | {mth} | {l_.get('yes', '')}/{l_.get('no', '')} | {so} | {a_} | {esc(str(l_.get('note', '')))} |")
wh = [r for r in HV if r[2] != "unsure"]
chk("topics with a verdict", "slide 7: 13", 13, len(wh)); chk("person real gap, loop", "B7: 3 of 4", (3, 4), (sum(r[2] == "real_gap" for r in wh if r[1] == "loop"), sum(r[1] == "loop" for r in wh)))
chk("person real gap, random", "B7: 4 of 9", (4, 9), (sum(r[2] == "real_gap" for r in wh if r[1] == "random"), sum(r[1] == "random" for r in wh)))
for jn, c in (("Sonnet", 4), ("Astra", 5)):
    chk(f"{jn} real gap on those loop topics", "B7: 2 of 4", 2, sum(r[c] == "real_gap" for r in wh if r[1] == "loop")); chk(f"{jn} real gap on those random topics", "B7: 0 of 9", 0, sum(r[c] == "real_gap" for r in wh if r[1] == "random"))
w("", "Three prompts per labelled topic (uniform, seed 0), without neighbours. For counted topics the person's per-prompt answer is on the line.", "")
for row in HV:
    i = row[0]; w(f"Topic `{i}` ({row[1]}, person {row[2]}, Sonnet {row[4]}, Astra {row[5]}):")
    for k in pick(rng, list(range(10)), 3):
        pa = ANS.get(i, {}).get("a"); page_ex(i, k, nbs=False, lab=(f"person {pa[k]}, " if pa else "") + f"Sonnet {match(SO, i, k)}, Astra {match(AS, i, k)}")
    w("")
yes = lambda x: x == "yes"
agS = agA = agJ = tot = 0; bothyes = []; bothno = []; solo = []
for i, d in ANS.items():
    for k, a_ in enumerate(d["a"]):
        ms, ma = match(SO, i, k), match(AS, i, k); tot += 1
        agS += ms == {"y": "yes", "n": "no"}[a_]; agA += ma == {"y": "yes", "n": "no"}[a_]; agJ += ms == ma
        if a_ == "n" and yes(ms) and yes(ma): bothyes.append((i, k))
        if a_ == "y" and not yes(ms) and not yes(ma): bothno.append((i, k))
chk("prompts counted", "B7: 50", 50, tot); chk("person agrees with Sonnet", "B7: 41", 41, agS); chk("person agrees with Astra", "B7: 43", 43, agA); chk("judges agree", "B7: 47", 47, agJ); chk("both judges yes, person no", "B7: 6", 6, len(bothyes))
setline(f"2.5 The {len(bothyes)} prompts where both judges say yes and the person says no", "all of them, with their Dolci neighbours at 300 characters; all 6 are in random-arm topics. Which side is right has not been checked.", "`runs/loop_labels/db/answers/<id>.json`, `llm/sonnet.json`, `llm/astra.json`, `page_items.json`", "all of them")
for i, k in sorted(bothyes): page_ex(i, k, cut_nb=None, lab=f"topic arm {key[i]['arm']}; person no, Sonnet yes, Astra yes")
setline(f"2.6 The {len(bothno)} prompts where both judges say no and the person says yes", "the opposite direction. There are none: every both-judge disagreement with the person runs one way.", "same files", "all of them")
for i, k in sorted(bothno): page_ex(i, k, cut_nb=None, lab=f"topic arm {key[i]['arm']}; person yes, Sonnet no, Astra no")

# ------- 3. property tags
print("section 3", flush=True)
h(2, "3. Property tags (part 8) and pairs (part 13)")
FA = FS["facets"]
w("Slide 8 and B8. Tags by Sonnet with the source hidden; second labeller Sole on 200 prompts. `dist` is the MiniLM distance to the nearest Dolci prompt, available for traffic prompts only.", "")
for k, tv, dv, sl in [("asks_about_assistant", 4.1, .3, "8"), ("roleplay_or_persona_setup", 12, 5, "8"), ("placeholder_or_artefact", 21, 5, "8"), ("non_english", 20, 8, "8"), ("explicit_sexual", 3.6, .4, "B8"), ("adversarial_pressure", 3.4, 1.3, "B8"), ("homework_or_exam", 4, 29, "B8"), ("templated_dataset_item", 15, 42, "B8")]:
    a_ = 100 * sum(bool(T[i].get(k)) for i in TR_IDS) / len(TR_IDS); b_ = 100 * sum(bool(T[i].get(k)) for i in DO_IDS) / len(DO_IDS)
    chk(f"{k} share traffic, Dolci", f"slide {sl}: {tv} vs {dv}", (tv, dv), (round(a_) if tv == round(tv) else round(a_, 1), round(b_) if dv == round(dv) else round(b_, 1)))
chk("math share", "B8: 30 vs 3", (3, 30), (round(100 * FS["tasks"]["math or logic"]["traffic"]), round(100 * FS["tasks"]["math or logic"]["dolci"])))
from sklearn.metrics import roc_auc_score
hi = [(k, v["distance_auroc_in_traffic"]) for k, v in FA.items() if v["kappa_sole"] is not None and v["kappa_sole"] == v["kappa_sole"] and v["kappa_sole"] >= .5 and v["distance_auroc_in_traffic"] is not None]
chk("AUROC range for kappa >= 0.5", "slide 8: 0.385-0.611", (0.385, 0.611), (round(min(x[1] for x in hi), 3), round(max(x[1] for x in hi), 3)))
alla = [v["distance_auroc_in_traffic"] for v in FA.values() if v["distance_auroc_in_traffic"] is not None]
chk("AUROC minimum, all", "B8: 0.30", .30, min(alla), .006); chk("AUROC maximum, all", "B8: 0.67", .67, max(alla), .006)
for k, v in [("asks_about_assistant", .39), ("roleplay_or_persona_setup", .44), ("adversarial_pressure", .46)]: chk(f"AUROC {k}", "B8", v, round(FA[k]["distance_auroc_in_traffic"], 2))
for k, v in [("hidden_difficulty", .28), ("buried_request", .35), ("difficult_advice", -.01)]: chk(f"kappa {k}", "B8", v, round(FA[k]["kappa_sole"], 2))
dist = np.array([SR[i]["dist"] for i in TR_IDS])
chk("AUROC recomputed, asks_about_assistant", "facets/table.md", .385, round(roc_auc_score([bool(T[i].get("asks_about_assistant")) for i in TR_IDS], dist), 3))
dq = np.percentile(dist, 50)
rng = fresh()
def prop_block(k, tagf=None, title=None):
    tagf = tagf or (lambda i: T[i].get(k))
    tr = [i for i in TR_IDS if tagf(i)]; do = [i for i in DO_IDS if tagf(i)]
    v = FA.get(k, {}); head = title or k
    d = np.array([SR[i]["dist"] for i in tr]) if tr else np.array([0.])
    info = f"traffic {pct(len(tr), len(TR_IDS))}; Dolci {pct(len(do), len(DO_IDS))}"
    if v: info += f"; kappa vs Sole {v['kappa_sole']}; distance AUROC {v['distance_auroc_in_traffic']}"
    info += f"; median dist tagged {np.median(d):.3f} vs all traffic {dq:.3f}; of tagged traffic, {sum(bool(PL.search(text(i))) for i in tr)} contain NAME\\_k and sources are {dict(Counter(tsrc(i) for i in tr))}"
    setline(head, info + ".", "`runs/gap_taxonomy/facets/{sample,tags}.jsonl`", f"5 uniform traffic, 5 uniform Dolci (all if fewer), seed {SEED}")
    w("Traffic:")
    for i in pick(rng, tr, 5): ex(i, f"{tsrc(i)}, dist {SR[i]['dist']:.3f}, lang {lang(i)}")
    w("", "Dolci:")
    for i in pick(rng, do, 5): ex(i, f"{SRC.get(i, '?')}, lang {lang(i)}")
    w("")
for k in ["asks_about_assistant", "roleplay_or_persona_setup", "placeholder_or_artefact", "non_english", "explicit_sexual", "adversarial_pressure", "harmful_request", "homework_or_exam", "templated_dataset_item"]: prop_block(k)
prop_block("math", lambda i: T[i].get("task") == "math or logic", "task = math or logic")
h(3, "3.1 The subtle kinds (kappa -0.01 to 0.35)")
for k in ["hidden_difficulty", "buried_request", "difficult_advice"]: prop_block(k)
h(3, "3.2 Examples the two labellers tag differently")
w("Source: `runs/gap_taxonomy/facets/tags.jsonl` (Sonnet) and `facets/sole.jsonl` (Sole), 200 prompts. 4 uniform per property, seed 0, all if fewer.")
for k in ["hidden_difficulty", "buried_request", "difficult_advice", "system_prompt_like", "terse_underspecified", "placeholder_or_artefact", "roleplay_or_persona_setup", "non_english", "harmful_request"]:
    com = sorted(i for i in S2 if i in T); dd = [i for i in com if bool(T[i].get(k)) != bool(S2[i].get(k))]
    w("", f"**{k}**: kappa {FA[k]['kappa_sole']}; {len(dd)} of {len(com)} double-tagged prompts differ.", "")
    for i in pick(rng, dd, 4): ex(i, f"{SR[i]['src']}; Sonnet {'yes' if T[i].get(k) else 'no'}, Sole {'yes' if S2[i].get(k) else 'no'}")
h(3, "3.3 Distance on tagged prompts")
for k in ["asks_about_assistant", "roleplay_or_persona_setup"]:
    tr = sorted((i for i in TR_IDS if T[i].get(k)), key=lambda i: SR[i]["dist"])
    setline(f"Chosen: closest and farthest {k} traffic prompts", "tagged prompts span the distance range; the closest ones are close because of what their neighbours share.", "`facets/sample.jsonl` (`dist`), `facets/tags.jsonl`", "chosen: 3 smallest and 3 largest `dist` among tagged traffic prompts")
    for i in tr[:3] + tr[-3:]: ex(i, f"dist {SR[i]['dist']:.3f}")
for k in ["placeholder_or_artefact", "pasted_input", "false_premise"]:
    tr = sorted((i for i in TR_IDS if T[i].get(k)), key=lambda i: -SR[i]["dist"])
    setline(f"Against the claim: {k}, where distance does rank tagged prompts higher", f"AUROC {FA[k]['distance_auroc_in_traffic']}, above 0.5; kappa {FA[k]['kappa_sole']}. The 3 farthest tagged prompts.", "same files", "chosen: 3 largest `dist`")
    for i in tr[:3]: ex(i, f"dist {SR[i]['dist']:.3f}")
h(3, "3.3b Distance AUROC split by traffic source (added after looking, exploratory)")
tsv = np.array([tsrc(i) for i in TR_IDS]); wt_ = {s_: int((tsv == s_).sum()) for s_ in ("lmsys", "wildchat")}
w(f"The traffic sample is {wt_['lmsys']} LMSYS and {wt_['wildchat']} WildChat prompts. Median `dist`: LMSYS {np.median(dist[tsv == 'lmsys']):.3f}, WildChat {np.median(dist[tsv == 'wildchat']):.3f}; `dist` below 0.01: LMSYS {int((dist[tsv == 'lmsys'] < .01).sum())}, WildChat {int((dist[tsv == 'wildchat'] < .01).sum())}. Dolci holds WildChat-derived rows ({sum(SRC.get(i) == 'Wildchat' for i in DO_IDS)} of the {len(DO_IDS)} Dolci sample have source Wildchat). Source for the split: `mod_source` in `runs/safety_removal/moderation/labels.parquet`.", "", "| property | share LMSYS % (n) | share WildChat % (n) | AUROC all | AUROC LMSYS | AUROC WildChat |", "|---|---|---|---|---|---|")
for k in ["asks_about_assistant", "roleplay_or_persona_setup", "adversarial_pressure", "placeholder_or_artefact", "non_english", "explicit_sexual", "harmful_request", "pasted_input", "homework_or_exam", "templated_dataset_item", "strict_output_format", "high_stakes_domain"]:
    y_ = np.array([bool(T[i].get(k)) for i in TR_IDS]); au = lambda m_: roc_auc_score(y_[m_], dist[m_])
    w(f"| {esc(k)} | {100 * y_[tsv == 'lmsys'].mean():.1f} ({int(y_[tsv == 'lmsys'].sum())}) | {100 * y_[tsv == 'wildchat'].mean():.1f} ({int(y_[tsv == 'wildchat'].sum())}) | {roc_auc_score(y_, dist):.3f} | {au(tsv == 'lmsys'):.3f} | {au(tsv == 'wildchat'):.3f} |")
h(3, "3.4 Prompts in pairs of properties (part 13, exploratory)")
ok = [k for k, v in FA.items() if v["kappa_sole"] is not None and v["kappa_sole"] == v["kappa_sole"] and v["kappa_sole"] >= .4]
def has(i, x): return T[i].get("task") == x[5:] if x.startswith("task=") else bool(T[i].get(x))
for a_, b_, wt, wd, sl in [("non_english", "task=roleplay or chat", 25, 0, "B8"), ("asks_about_assistant", "non_english", 26, 0, "B8"), ("non_english", "task=advice or opinion", 25, 3, "tldr"), ("high_stakes_domain", "non_english", 26, 3, "tldr"), ("harmful_request", "roleplay_or_persona_setup", 36, 6, "B8"), ("explicit_sexual", "placeholder_or_artefact", 33, 0, "combinations.md"), ("non_english", "roleplay_or_persona_setup", 21, 1, "combinations.md")]:
    tr = [i for i in TR_IDS if has(i, a_) and has(i, b_)]; do = [i for i in DO_IDS if has(i, a_) and has(i, b_)]
    chk(f"pair {a_} + {b_}", sl, (wt, wd), (len(tr), len(do)))
    tmp = sum(bool(PL.search(text(i))) for i in tr)
    setline(f"{a_} + {b_}", f"traffic {len(tr)}, Dolci {len(do)}; {tmp} of the {len(tr)} traffic prompts contain NAME\\_k; languages {dict(Counter(lang(i) for i in tr).most_common(5))}.", "`facets/tags.jsonl`, `facets/combinations.md`", f"5 uniform traffic, all Dolci (at most 6), seed {SEED}")
    for i in pick(rng, tr, 5): ex(i, f"traffic, {tsrc(i)}, lang {lang(i)}, dist {SR[i]['dist']:.3f}")
    for i in pick(rng, do, 6): ex(i, f"Dolci, {SRC.get(i, '?')}, lang {lang(i)}")

# ------- 4. contrastive
print("section 4", flush=True)
h(2, "4. Contrastive, checked properties (part 9)")
rng = fresh()
CR = {r["id"]: r for r in jl(G / "contrastive/rows.jsonl")}; CU = {**jr(G / "contrastive/units.json"), **jr(G / "contrastive/units_topics.json")}; CS = jr(G / "contrastive/summary.json")
for k_, a_, b_ in [("r50", 39, 56), ("topic", 8, 49), ("far", 12, 49), ("random", 6, 50)]: chk(f"contrastive pass {k_}", "tldr/B7", (a_, b_), (CS[k_]["pass"], CS[k_]["verified"]))
ids_ = lambda st, p: sorted(u for u, r in CR.items() if r["set"] == st and r.get("passes") is p)
pl = {}
for name, st, p, n in [("r50_pass", "r50", True, 5), ("r50_fail", "r50", False, 3), ("topic_pass", "topic", True, 8), ("topic_fail", "topic", False, 3), ("far_pass", "far", True, 3), ("far_fail", "far", False, 3), ("rand_pass", "random", True, 3), ("rand_fail", "random", False, 2)]: pl[name] = pick(rng, ids_(st, p), n)
named = {}
for u in ids_("r50", True):
    q = CR[u]["proposal"]["predicate"]
    for w_ in ("Vicuna", "ISO 26262", "Florida", "erotic", "sexual"):
        if w_.lower() in q.lower() and w_ not in named: named[w_] = u
pl["named"] = list(named.values())
need = sorted({int(i) for us in pl.values() for u in us for i in CU[u]["held"]})
print("neighbours for", len(need), flush=True)
th = np.load(IDX / "traffic_hash.npy"); mh = np.load(IDX / "mix_hash.npy")
Q = torch.nn.functional.normalize(torch.from_numpy(np.asarray(np.load(IDX / "traffic.npy", mmap_mode="r")[need], dtype=np.float32)), dim=1)
M = torch.nn.functional.normalize(torch.from_numpy(np.load(IDX / "mix.npy").astype(np.float32)), dim=1); nn = {}
for s0 in range(0, len(need), 128):
    _, ix = (Q[s0:s0 + 128] @ M.T).topk(5, dim=1)
    for r_ in range(len(ix)):
        qi = need[s0 + r_]; nn[qi] = [int(j) for j in ix[r_] if int(mh[j]) != int(th[qi])][0]
del M, Q
wanted = set(nn.values())
for f in sorted(glob.glob(str(ROOT / "runs/acts_dolci/shard*.jsonl"))):
    for l in open(f):
        if wanted and any(True for _ in [0]):
            r = json.loads(l)
            if r["idx"] in wanted: EXTRA_D[f"d{r['idx']}"] = r["text"]
print("dolci texts", len(EXTRA_D), flush=True)
rate_ok = []
def unit_block(u, rng, n=2):
    r = CR[u]; held = [int(i) for i in CU[u]["held"]]; p = r["proposal"]
    w("", f"Unit `{u}` ({r['n_members']} member prompts). Property: \"{esc(p['predicate'])}\" Axis: {p['axis']}. Member rate {r['member_rate']:.2f}, Dolci rate {r['dolci_rate']:.2f}, passes {r['passes']}." + (f" Loop's name: \"{esc(REV[int(u.split('_')[1])]['name'])}\"." if u.startswith("r50_") and int(u.split('_')[1]) in REV else ""), "")
    msgs = [("member", i) for i in held] + [("dolci", nn[i]) for i in held]; order = list(range(len(msgs))); random.Random(0).shuffle(order); msgs = [msgs[k] for k in order]
    ans = json.loads(r["verify_raw"][r["verify_raw"].find("{"): r["verify_raw"].rfind("}") + 1])["answers"]; A_ = {}
    for a_, (s_, i) in zip(ans, msgs): A_[(s_, i)] = str(a_).strip().lower()[:1] == "y"
    rm = np.mean([A_[("member", i)] for i in held]); rd = np.mean([A_[("dolci", nn[i])] for i in held]); rate_ok.append(abs(rm - r["member_rate"]) < 1e-6 and abs(rd - r["dolci_rate"]) < 1e-6)
    for i in pick(rng, held, n):
        ex(f"t{i}", f"member, answer {'yes' if A_[('member', i)] else 'no'}"); ex(f"d{nn[i]}", f"its nearest Dolci prompt, answer {'yes' if A_[('dolci', nn[i])] else 'no'}", None, 2)
SRCL = "`runs/gap_taxonomy/contrastive/{rows.jsonl,units.json,units_topics.json}`; held prompts from `traffic_rows.jsonl`; nearest Dolci recomputed from `runs/negatives/index/mix.npy` as in `scripts/gap_taxonomy_contrastive.py`"
for name, title, shows in [("r50_pass", "4.1 Loop gaps with a checked property, uniformly random", "the property and the blind yes/no answers for member prompts and nearest Dolci prompts."),
                            ("r50_fail", "4.2 Loop gaps with no checked property, uniformly random", "17 of 56 loop gaps do not pass."),
                            ("named", "4.3 Chosen: the subject examples the TL;DR names", "the first passing unit whose property mentions Vicuna, ISO 26262, Florida, erotic or sexual; chosen because the TL;DR cites them."),
                            ("topic_pass", "4.4 Against the claim: every random topic the loop did not confirm that also passes", "all 8; a checked property is not specific to the loop's gaps."),
                            ("topic_fail", "4.5 Random topics that do not pass, uniformly random", "the control's normal case."),
                            ("far_pass", "4.6 Far-prompt clusters that pass, uniformly random", "far clusters had no selection by the rater; 12 of 49 pass."),
                            ("far_fail", "4.7 Far-prompt clusters that fail, uniformly random", "37 of 49 fail."),
                            ("rand_pass", "4.8 Against the claim: clusters of random prompts that pass, uniformly random", "6 of 50 pass; the far vs random difference is p = 0.12."),
                            ("rand_fail", "4.9 Clusters of random prompts that fail, uniformly random", "for comparison.")]:
    setline(title, shows, SRCL, f"from the sorted pool, seed {SEED}, 2 held prompts each")
    for u in pl[name]: unit_block(u, rng)
chk("contrastive answers replay to the stored rates", "rows.jsonl", True, all(rate_ok))
h(3, "4.10 All 39 checked properties on loop gaps")
w("Complete list from `runs/gap_taxonomy/contrastive/summary.json`, `r50.passing`.", "", "| unit | axis | member | Dolci | property |", "|---|---|---|---|---|")
for x in CS["r50"]["passing"]: w(f"| {x['id']} | {x['axis']} | {x['member_rate']:.2f} | {x['dolci_rate']:.2f} | {esc(x['predicate'])} |")
w("", f"Axes of passing properties: {CS['r50']['axis_pass']}.")

# ------- 5. emergent
print("section 5", flush=True)
h(2, "5. Emergent clusters (part 10) and the grouping test (part 16)")
rng = fresh()
ES = jr(G / "emergent/summary.json"); Z = np.load(G / "emergent/clusters.npz", allow_pickle=True); zids = [str(x) for x in Z["ids"]]
GR = {"desc": dict(zip(zids, Z["desc_fine_s0"].tolist())), "raw": dict(zip(zids, Z["raw_fine_s0"].tolist()))}
NM = {(n["arm"], n["node"]): n for n in jl(G / "emergent/names.jsonl") if n["level"] == "fine"}; KD = jr(G / "grouping_recovery/group_kinds.json")["kinds"]
DE = {}
for r in jl(G / "emergent/desc.jsonl"):
    if r["desc"]:
        for i, d in zip(r["ids"], r["desc"]): DE[i] = d
nodes = [n for n in ES["arms"]["desc"]["nodes"] if n["level"] == "fine"]; heavy = [n for n in nodes if n["traffic_heavy"]]
chk("traffic-heavy description groups", "tldr: 10 of 100", 10, len(heavy)); chk("traffic in heavy groups", "tldr: 20%", 20, round(100 * ES["arms"]["desc"]["fine_traffic_in_heavy"]))
chk("confirmed gaps in heavy groups", "tldr: 13.8", 13.8, round(100 * ES["mapback_desc"]["confirmed"]["fine"], 1))
members = lambda g, nd: sorted(i for i in zids if GR[g][i] == nd)
def group_block(g, nd, rng, n=3):
    x = NM.get((g, nd), {}); ms = members(g, nd); ntr = sum(i.startswith("t") for i in ms)
    w("", f"Group {g}:{nd}: \"{esc(str(x.get('name')))}\". {ntr} traffic, {len(ms) - ntr} Dolci. Judge kinds: {KD.get(f'{g}:{nd}', [])}. Summary: {esc(str(x.get('summary')))}", "")
    for i in pick(rng, ms, n): ex(i, ("description: " + esc(DE[i])) if g == "desc" and i in DE else "")
setline("5.1 All 10 traffic-heavy description groups", "what the method produced at its top: the group names and 3 uniform members with the describer's text.", "`runs/gap_taxonomy/emergent/{summary.json,clusters.npz,names.jsonl,desc.jsonl}`", f"all 10 groups, 3 uniform members each, seed {SEED}")
for n in heavy: group_block("desc", n["node"], rng)
s = pick(rng, [n["node"] for n in nodes if not n["traffic_heavy"]], 5)
setline("5.2 Five uniformly random other description groups", "groups that are not traffic-heavy.", "same files", f"5 uniform of 90, seed {SEED}")
for nd in s: group_block("desc", nd, rng)
all_d = [i for i in ARMS["full"] if i in GR["desc"]]; tra = [i for i in A["traffic"] if i in GR["desc"]]
def rank_groups(g, dol):
    ct = np.bincount([GR[g][i] for i in tra], minlength=100) / len(tra); cd = np.bincount([GR[g][i] for i in dol], minlength=100) / len(dol); d = ct - cd
    return np.argsort(-d), d
GRS = jr(G / "grouping_recovery/summary.json")
chk("grouping cells recovered", "slide 9: 0 of 12", 0, sum(v["recovered"] for F_ in GRS for v in GRS[F_].values())); chk("grouping cells", "slide 9: 12", 12, sum(len(GRS[F_]) for F_ in GRS))
FAMS = ["safety", "code", "multilingual"]
PV = jr(G / "grouping_recovery/planted_vs_natural.json")
for F_, a_, b_ in [("safety", 9.8, 12.7), ("multilingual", 6.5, 10.1), ("code", -10.3, 5.8)]: chk(f"desc groups of kind {F_}, before and after", "B9", (a_, b_), (round(PV[F_]["full"], 1), round(PV[F_]["minus"], 1)))
h(3, "5.3 Top 10 groups after a planted removal vs a random removal")
w("Rank by traffic share minus Dolci share. Random removal = seed 1 of the 20 in `scripts/gap_grouping_recovery.py`. A group is marked with the judge kinds from `grouping_recovery/group_kinds.json` (Sonnet only). This is the test behind \"0 of 12\".")
for F_ in FAMS:
    rem = [i for i in all_d if FAM.get(i) == F_]; oth = [i for i in all_d if FAM.get(i) != F_]; rm = set(rem); minus = [i for i in all_d if i not in rm]
    rr_ = set(random.Random(1).sample(oth, len(rem))); rand = [i for i in all_d if i not in rr_]
    for g in ("desc", "raw"):
        om, dm = rank_groups(g, minus); orr, dr = rank_groups(g, rand)
        chk(f"{F_} {g}: F-groups in top 10 after removal, recomputed", "grouping_recovery/summary.json", GRS[F_][g]["minus"]["top10_F"], sum(F_ in KD.get(f"{g}:{int(k)}", []) for k in om[:10]))
        w("", f"**{F_}, {g} groups** (removed {len(rem)}; F-groups in top 10: {GRS[F_][g]['minus']['top10_F']} after removal, random median {GRS[F_][g]['random_top10_F_median']}; best F rank {GRS[F_][g]['minus']['best_F_rank']}, random median {GRS[F_][g]['random_best_rank_median']}; randoms beaten {GRS[F_][g]['randoms_beaten_on_best_rank']} of 20)", "", "| rank | after planted removal | diff | kinds | after random removal (seed 1) | diff | kinds |", "|---|---|---|---|---|---|---|")
        for r_ in range(10):
            a1, b1 = int(om[r_]), int(orr[r_])
            w(f"| {r_ + 1} | {esc(str(NM.get((g, a1), {}).get('name')))} | {100 * dm[a1]:+.1f} | {KD.get(f'{g}:{a1}', [])} | {esc(str(NM.get((g, b1), {}).get('name')))} | {100 * dr[b1]:+.1f} | {KD.get(f'{g}:{b1}', [])} |")
    om, dm = rank_groups("desc", minus); fg = [int(k) for k in om if F_ in KD.get(f"desc:{int(k)}", [])][:1]
    setline(f"Best-ranked {F_}-type description group after the removal, and removed prompts inside it", "what such a group holds.", "`emergent/clusters.npz`, `removal_arms.json`", f"chosen: top-ranked group judged {F_}; 5 uniform members, seed {SEED}")
    for nd in fg:
        ms = members("desc", nd); w(f"Group desc:{nd}: \"{esc(str(NM[('desc', nd)]['name']))}\", diff {100 * dm[nd]:+.1f} points.", "")
        for i in pick(rng, ms, 5): ex(i, ("removed" if i in rm else ("traffic" if i.startswith("t") else "kept Dolci")))

cg = [("Russian-language general assistant requests",), ("Text-completion engine wrapper for programming tutorials",)]
h(3, "5.4 Chosen: the two \"code-type\" description groups that rank highest after a random removal")
w("Slide 9 and B9 give a reason for code: code-type groups already rank 4-5 and fill 2 of the top 10. These are the two groups counted, chosen because that sentence rests on them. The judge marked them code from the name and summary.")
for (nm_,) in cg:
    nd = [k[1] for k, v in NM.items() if k[0] == "desc" and v.get("name") == nm_][0]; ms = members("desc", nd); tg = [i for i in ms if i in T]
    w("", f"Task=coding share among its tagged members: {pct(sum(T[i].get('task') == 'coding' for i in tg), len(tg))}; non-English: {pct(sum(bool(T[i].get('non_english')) for i in tg), len(tg))}; contains NAME\\_k: {pct(sum(bool(PL.search(text(i))) for i in ms), len(ms))}.")
    group_block("desc", nd, rng, 5)
# ------- judges helper
ITEMS = jr(G / "removal_judge/items.json"); JS = jr(G / "removal_judge/sonnet.json"); JA = jr(G / "removal_judge/astra.json"); JSUM = jr(G / "removal_judge/summary.json")["results"]
def verdicts(method):
    out = defaultdict(list)
    for it in ITEMS:
        if it["method"] == method: out[it["arm"]].append(it)
    return out
vd = lambda it: (JS[str(it["k"])], JA[str(it["k"])])
for m, want in [("instag", {"safety": (0, 0), "code": (0, 0), "multilingual": (0, 0)}), ("hypothesaes", {"safety": (0, 0), "code": (1, 1), "multilingual": (0, 1)}), ("lloom", {"safety": (0, 0), "code": (1, 1), "multilingual": (0, 0)}), ("lloom_seeded", {"safety": (0, 0), "code": (0, 0), "multilingual": (0, 0)})]:
    for F_, v in want.items(): chk(f"recovery {m} {F_}, Sonnet / Astra", "B9", v, (JSUM["sonnet"][m][F_]["recovery"], JSUM["astra"][m][F_]["recovery"]))
def judge_tables(method, descf, tail=None):
    v = verdicts(method)
    for F_ in FAMS:
        a1, a2 = v[f"minus_{F_}"], v[f"random_{F_}"]
        w("", f"**{F_}**: recovery Sonnet {JSUM['sonnet'][method][F_]['recovery']}, Astra {JSUM['astra'][method][F_]['recovery']} (pre-set bar 2). Marks are the kinds each judge gave: S = Sonnet, A = Astra.", "", f"| rank | after removal of {F_} | S | A | after random removal | S | A |", "|---|---|---|---|---|---|---|")
        for r_ in range(max(len(a1), len(a2))):
            c1 = (esc(descf(a1[r_]["desc"])), *vd(a1[r_])) if r_ < len(a1) else ("", "", ""); c2 = (esc(descf(a2[r_]["desc"])), *vd(a2[r_])) if r_ < len(a2) else ("", "", "")
            w(f"| {r_ + 1} | {c1[0]} | {c1[1]} | {c1[2]} | {c2[0]} | {c2[1]} | {c2[2]} |")

# ------- 6. instag
print("section 6", flush=True)
h(2, "6. InsTag open-set tags (part 12)")
rng = fresh()
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()
IR = {r["id"]: r for r in jl(G / "instag/tags.jsonl")}; IT = {}
for i, r in IR.items():
    ts = r["tags"] or re.findall(r'"tag"\s*:\s*"([^"]+)"', r["raw"] or "")
    if ts: IT[i] = {norm(t) for t in ts if norm(t)}
ISH = jr(G / "instag/shift.json"); ftags = ISH["f_tags"]; itra = [i for i in A["traffic"] if i in IT]; ict = Counter(t for i in itra for t in IT[i]); keep = sorted(t for t, c in ict.items() if c >= 10)
def irank(arm):
    ids = [i for i in ARMS[arm] if i in IT]; cd = Counter(t for i in ids for t in IT[i]); sc = {t: ict[t] / len(itra) - cd[t] / len(ids) for t in keep}
    return sc, {t: r for r, (_, t) in enumerate(sorted(((v, t) for t, v in sc.items()), reverse=True), 1)}
best = {}
for F_, (m_, r_, f_) in {"safety": (39, 39, 42), "code": (12, 20, 19), "multilingual": (26, 158, 155)}.items():
    for arm, want in ((f"minus_{F_}", m_), (f"random_{F_}", r_), ("full", f_)):
        _, rk = irank(arm); b = min(ftags[F_], key=lambda t: rk.get(t, 9999)); best[(F_, arm)] = b; chk(f"InsTag best {F_} tag rank in {arm}", "B9 / shift.json", want, rk[b])
chk("InsTag no-tag share traffic", "tldr: 6.5%", 6.5, round(100 * (1 - len(itra) / 2000), 1))
w("Tagger: released InsTagger (LLaMA-2 7B), query alone (v3). The tag is free-form; tags are normalised to lowercase words. 216 tags have at least 10 traffic prompts.")
judge_tables("instag", lambda d: d)
s = pick(rng, sorted(IT), 8)
setline("6.1 What it produced: eight uniformly random prompts with raw tagger output", "the tagger's own tags for an unselected draw.", "`runs/gap_taxonomy/instag/tags.jsonl`", f"8 uniform of {len(IT)} tagged, seed {SEED}")
for i in s: ex(i, f"tags: {esc(', '.join(sorted(IT[i])))}")
none = sorted(set(IR) - set(IT))
setline("6.2 Prompts that got no tag", f"{len(none)} of 4,000 got none; raw output shown.", "same file", "5 uniform, seed 0")
for i in pick(rng, none, 5): ex(i, f"raw: {esc(str(IR[i]['raw'])[:120])}")
ntags = [len(IT[i]) for i in sorted(IT)]; cosp = sum("cosplay" in IT[i] for i in itra); cospd = sum("cosplay" in IT[i] for i in ARMS["full"] if i in IT)
w("", f"Tags per tagged prompt: median {int(np.median(ntags))}, maximum {max(ntags)}. The tag \"cosplay\" in this run (v3, query alone): {pct(cosp, len(itra))} of tagged traffic, {cospd} Dolci prompts.")
top_tags = [t for t in jr(G / "instag/top10.json")["full"]][:3]
for tag in top_tags:
    tg = norm(tag); tr = sorted(i for i in itra if tg in IT[i]); do = sorted(i for i in ARMS["full"] if i in IT and tg in IT[i])
    setline(f"Top tag on full Dolci: \"{tag}\"", f"traffic {pct(len(tr), len(itra))}; Dolci {len(do)}.", "`instag/tags.jsonl`, `instag/top10.json`", "5 uniform traffic, 3 uniform Dolci, seed 0")
    for i in pick(rng, tr, 5): ex(i, "traffic, tags: " + esc(", ".join(sorted(IT[i]))))
    for i in pick(rng, do, 3): ex(i, "Dolci, tags: " + esc(", ".join(sorted(IT[i]))))
for F_, lab_ in (("code", "code"), ("multilingual", "language"), ("safety", "safety")):
    tag = best[(F_, f"minus_{F_}")]; sc_m, rk_m = irank(f"minus_{F_}"); sc_r, rk_r = irank(f"random_{F_}"); rem = sorted(set(ARMS["full"]) - set(ARMS[f"minus_{F_}"]))
    tr = sorted(i for i in itra if tag in IT[i]); rmv = [i for i in rem if i in IT and tag in IT[i]]
    setline(f"Against \"no signal\": best {lab_} tag \"{tag}\" moves in InsTag's own score", f"this tag is rank {rk_m[tag]} after the removal and {rk_r[tag]} after the random one; the best tag of the kind in the random arm is \"{best[(F_, f'random_{F_}')]}\" at rank {rk_r[best[(F_, f'random_{F_}')]]}; {len(rmv)} of {len(rem)} removed prompts carry this tag.", "`instag/tags.jsonl`, `instag/shift.json`", "chosen because it is the best-ranked judged tag of the kind; 5 uniform traffic, up to 5 removed, seed 0")
    for i in pick(rng, tr, 5): ex(i, "traffic, tags: " + esc(", ".join(sorted(IT[i]))))
    for i in pick(rng, rmv, 5): ex(i, "removed Dolci, tags: " + esc(", ".join(sorted(IT[i]))))
fc = [i for F_ in ("safety",) for i in sorted(set(ARMS["full"]) - set(ARMS["minus_safety"]))]
fct = Counter(t for i in fc if i in IT for t in IT[i]).most_common(10)
w("", f"Tags on the 105 removed safety prompts, most common: {fct}. {sum(i not in IT for i in fc)} of them got no tag.")

# ------- 7. HypotheSAEs
print("section 7", flush=True)
h(2, "7. HypotheSAEs features (part 12)")
rng = fresh()
HS = jr(G / "hypothesaes/shift.json")
from hypothesaes.sae import load_model
sae = load_model(str(G / "hypothesaes/sae/SAE_M=256_K=8.pt")); ch = np.load(G / "hypothesaes/emb_cache/sample/chunk_000.npy", allow_pickle=True); EM = {str(a_): np.array(b_, dtype=np.float32) for a_, b_ in ch}
sids = [r["id"] for r in sample if r["text"] in EM]; X = np.stack([EM[SR[i]["text"]] for i in sids]); ACT = sae.get_activations(X, show_progress=False)
print("activations", ACT.shape, flush=True)
aid = {i: k for k, i in enumerate(sids)}
judge_tables("hypothesaes", lambda d: f"n{d['neuron_idx']} ({d['target_separation_score']:+.2f}): {d['interpretation']}")
w("", "Only features with a positive separation score enter the top 10 list, so lists are shorter than 10. The same neuron gets a different written description in each arm because the description step is rerun per arm.")
arm_csv = {a_: pd.read_csv(G / f"hypothesaes/{a_}.csv").sort_values("target_separation_score", ascending=False).reset_index(drop=True) for a_ in ["full"] + [f"{p}_{F_}" for F_ in FAMS for p in ("minus", "random")]}
def feat_block(nr, title, shows, how="5 uniform active traffic, 3 uniform active Dolci, seed 0"):
    col = ACT[:, nr]; tr = [i for i in sids if col[aid[i]] > 0 and i.startswith("t")]; do = [i for i in sids if col[aid[i]] > 0 and i.startswith("d")]
    setline(title, shows + f" Active on {len(tr)} traffic and {len(do)} Dolci prompts of the sample.", "`hypothesaes/{<arm>.csv,sae/,emb_cache/sample/}`; activations recomputed from the saved SAE on saved embeddings", how)
    for i in pick(rng, tr, 5): ex(i, f"traffic, activation {col[aid[i]]:.2f}")
    for i in pick(rng, do, 3): ex(i, f"Dolci, activation {col[aid[i]]:.2f}")
fl = arm_csv["full"]; fl = fl[fl.target_separation_score > 0]
for row in fl.head(6).itertuples(): feat_block(int(row.neuron_idx), f"Feature n{row.neuron_idx} (full Dolci, separation {row.target_separation_score:+.2f}, fidelity {row.f1_fidelity_score:.2f})", f"Written description: \"{esc(row.interpretation)}\".")
for F_, jn in [("safety", "sonnet"), ("code", "astra")]:
    arm = f"minus_{F_}"; hit = arm_csv[arm][arm_csv[arm].interpretation.isin(HS["judges"][jn]["f_descriptions"][F_])]
    row = hit.iloc[0]
    feat_block(int(row.neuron_idx), f"Against \"no signal\": best {F_} feature after the {F_} removal, n{row.neuron_idx} (judge {jn}; rank {hit.index[0] + 1} of 20, separation {row.target_separation_score:+.2f})", f"Written description: \"{esc(row.interpretation)}\". Random-removal arm: {HS['judges'][jn]['families'][F_][f'random_{F_}']}.", "chosen because it is the secondary measure in B9; 5 uniform traffic, 3 uniform Dolci")
fm = HS["judges"]["sonnet"]["f_descriptions"]["multilingual"] + HS["judges"]["astra"]["f_descriptions"]["multilingual"]
w("", "All descriptions any judge counted as multilingual:")
for d in sorted(set(fm)): w(f"- Sonnet {'yes' if d in HS['judges']['sonnet']['f_descriptions']['multilingual'] else 'no'}, Astra {'yes' if d in HS['judges']['astra']['f_descriptions']['multilingual'] else 'no'}: {esc(d)}")

# ------- 8. LLooM
print("section 8", flush=True)
h(2, "8. LLooM concepts (part 12)")
rng = fresh()
for nm_, dirn, method in (("default", "lloom", "lloom"), ("seeded", "lloom_seeded", "lloom_seeded")):
    h(3, f"LLooM {nm_}")
    CC = jr(G / dirn / "concepts.json"); SD = pd.read_pickle(G / dirn / "score_df.pkl"); DT = jr(G / dirn / "top10_detail.json")
    if nm_ == "default": chk("LLooM default concepts", "B9: 13", 13, len(CC))
    w(f"All {len(CC)} concepts induced (`{dirn}/concepts.json`):", "")
    for c in CC.values(): w(f"- {c['name']}: {esc(c['prompt'])}")
    judge_tables(method, lambda d: d[:110])
    for F_ in FAMS:
        for a_ in ("minus", "random"):
            lst = DT[f"{a_}_{F_}"]; w("", f"{F_}, {a_}: " + "; ".join(f"{x['name']} {100 * x['diff']:+.1f}" for x in lst))
    names = [x["name"] for x in DT["full"]][:2]; extra = ["Error Handling"] if nm_ == "default" else []
    for cn in names + extra:
        sub = SD[SD.concept_name == cn]; on = sub[sub.score >= 1.0]; tr = sorted(x for x in on.doc_id if x.startswith("t")); do = sorted(x for x in on.doc_id if x.startswith("d")); RA = dict(zip(sub.doc_id, sub.rationale))
        setline(f"Concept \"{cn}\"" + (" (against \"no signal\": rank 3 after the code removal vs 9 after the random one)" if cn == "Error Handling" else ""), f"scored 1.0 on {len(tr)} traffic and {len(do)} Dolci prompts.", f"`{dirn}/score_df.pkl`", "5 uniform traffic, 3 uniform Dolci, seed 0")
        for i in pick(rng, tr, 5): ex(i, "traffic; rationale: " + esc(str(RA[i])[:120]))
        for i in pick(rng, do, 3): ex(i, "Dolci; rationale: " + esc(str(RA[i])[:120]))
    if nm_ == "default":
        sub = SD[(SD.concept_name == "Software Development") & (SD.score < 1.0)]; rm = sorted(set(ARMS["full"]) - set(ARMS["minus_code"])); zero = sorted(set(sub.doc_id) & set(rm))
        setline("Removed code prompts LLooM scored 0 on Software Development", f"{len(zero)} of {len(rm)} removed code prompts.", "`lloom/score_df.pkl`, `removal_arms.json`", "5 uniform, seed 0")
        for i in pick(rng, zero, 5): ex(i, f"{SRC[i]}")

# ------- 9. planted
print("section 9", flush=True)
h(2, "9. Planted removals")
rng = fresh()
LJ = jr(G / "removal_arms_languages.json"); full = set(ARMS["full"])
REM = {F_: sorted(full - set(ARMS[f"minus_{F_}"])) for F_ in FAMS}
for F_, n in [("safety", 105), ("code", 397), ("multilingual", 99)]: chk(f"removed {F_}", "slide 9", n, len(REM[F_]))
langs = Counter(lang(i) for i in REM["multilingual"]); tlang = Counter(lang(i) for i in A["traffic"]); absent = sorted(l for l in langs if tlang[l] == 0)
chk("languages in the removed non-English prompts", "B9: 39", 39, len(langs)); chk("removed prompts in languages with no traffic", "slide 9: 57", 57, sum(langs[l] for l in absent))
for l, n, nm_ in [("mg", 11, "Malagasy"), ("ta", 8, "Tamil"), ("si", 7, "Sinhala"), ("ky", 4, "Kyrgyz"), ("pa", 4, "Punjabi")]: chk(f"removed {nm_}", "B9", n, langs[l])
for l, n, nm_ in [("ru", 72, "Russian"), ("pt", 62, "Portuguese"), ("es", 43, "Spanish")]: chk(f"traffic {nm_}", "B9", n, tlang[l])
chk("removed prompts py3langid calls English", "slide 9 says 99 non-English prompts", 0, langs.get("en", 0))
chk("pooled non-English gap", "B9: 13.2 to 18.1", (13.2, 18.1), (LJ["pooled_gap_traffic_minus_dolci_points"]["full"], LJ["pooled_gap_traffic_minus_dolci_points"]["minus_multilingual"]))
srcs = {F_: Counter(SRC[i] for i in REM[F_]) for F_ in FAMS}
def rm_label(i): return f"{SRC[i]}, lang {lang(i)}, {tlabel(i)}, harmful\\_request {'yes' if T.get(i, {}).get('harmful_request') else 'no'}, Koala {KO.get(i)}, NBER {NB.get(i, 'untagged')}"
sf = REM["safety"]; nh = [i for i in sf if i in T and not T[i].get("harmful_request") and not T[i].get("adversarial_pressure")]
setline("9.1 Removed safety prompts, uniformly random", f"sources {dict(srcs['safety'].most_common(6))}. {sum(1 for i in sf if T.get(i, {}).get('harmful_request'))} of {len(sf)} are tagged harmful\\_request by Sonnet.", "`runs/gap_taxonomy/removal_arms.json`, `facets/sample.jsonl`", f"10 uniform of {len(sf)}, seed {SEED}")
for i in pick(rng, sf, 10): ex(i, rm_label(i))
setline("9.2 Against the claim: removed safety prompts with no harm tag", f"{len(nh)} of {len(sf)} removed safety prompts have neither harmful\\_request nor adversarial\\_pressure, so the planted kind is partly benign.", "same files", "5 uniform from those")
for i in pick(rng, nh, 5): ex(i, rm_label(i))
cd = REM["code"]; ncd = [i for i in cd if i in T and T[i].get("task") != "coding"]
setline("9.3 Removed code prompts, uniformly random", f"sources {dict(srcs['code'].most_common(6))}. {sum(1 for i in cd if T.get(i, {}).get('task') == 'coding')} of {len(cd)} are tagged task=coding.", "same files", f"10 uniform of {len(cd)}, seed {SEED}")
for i in pick(rng, cd, 10): ex(i, rm_label(i))
setline("9.4 Against the claim: removed code prompts not tagged coding", f"{len(ncd)} of {len(cd)}.", "same files", "5 uniform from those")
for i in pick(rng, ncd, 5): ex(i, rm_label(i))
ml = REM["multilingual"]; nne = [i for i in ml if i in T and not T[i].get("non_english")]
setline("9.5 Removed non-English prompts, uniformly random", f"sources {dict(srcs['multilingual'].most_common(6))}; {sum(1 for i in ml if T.get(i, {}).get('non_english'))} of {len(ml)} tagged non\\_english by Sonnet.", "same files", f"10 uniform of {len(ml)}, seed {SEED}")
for i in pick(rng, ml, 10): ex(i, rm_label(i))
setline("9.6 Against the claim: removed non-English prompts Sonnet does not tag non-English", f"{len(nne)} of {len(ml)}.", "same files", "all, at most 5")
for i in pick(rng, nne, 5): ex(i, rm_label(i))
h(3, "9.7 Languages absent from traffic")
w(f"Per language counts, removed prompts (py3langid; `removal_arms_languages.json`, recomputed here from `lang_dolci.npy` and `blindspots/prompts.parquet`): " + ", ".join(f"{l} {n}{' (no traffic)' if tlang[l] == 0 else f' (traffic {tlang[l]})'}" for l, n in langs.most_common()) + f".", "", f"Languages with no traffic prompt in the sample of 2,000: {absent}.")
setline("Chosen: two removed prompts from each of the five largest absent languages", "what the slide's Malagasy, Tamil, Sinhala, Kyrgyz and Punjabi prompts look like.", "`removal_arms.json`, `facets/sample.jsonl`", "chosen by language; 2 uniform per language, seed 0")
for l in ["mg", "ta", "si", "ky", "pa"]:
    for i in pick(rng, [x for x in ml if lang(x) == l], 2): ex(i, rm_label(i))
pr = [i for i in ml if tlang[lang(i)] > 0]
setline("Against the \"absent\" reading: removed prompts in languages traffic does use", f"{len(pr)} of {len(ml)}.", "same files", "5 uniform, seed 0")
for i in pick(rng, pr, 5): ex(i, f"traffic has {tlang[lang(i)]} in this language; " + rm_label(i))
tra_ne = [i for i in A["traffic"] if lang(i) != "en"]
setline("Traffic non-English prompts in the same sample, uniformly random", f"{len(tra_ne)} of 2,000 traffic prompts; top languages {dict(Counter(lang(i) for i in tra_ne).most_common(6))}.", "`removal_arms.json`, `blindspots/prompts.parquet`", "5 uniform, seed 0")
for i in pick(rng, tra_ne, 5): ex(i, f"{tsrc(i)}, lang {lang(i)}")
h(3, "9.8 Random removals, for comparison")
for F_ in FAMS:
    rr_ = sorted(full - set(ARMS[f"random_{F_}"])); setline(f"Removed by random\\_{F_}", f"{len(rr_)} prompts; families {dict(Counter(FAM[i] for i in rr_).most_common(6))}.", "`removal_arms.json`", "5 uniform, seed 0")
    for i in pick(rng, rr_, 5): ex(i, f"family {FAM[i]}, {SRC[i]}")
h(3, "9.9 Fixed property tags: the ranks on slide 9")
PN = jr(ROOT / "runs/review/part15_null20.json")
for F_, a_, b_, lo, hi_ in [("code", 31, 6, 30, 31), ("safety", 12, 8, 12, 13), ("multilingual", 3, 2, 3, 3)]:
    p = PN[f"all/{F_}"]; chk(f"fixed tags {F_}", "slide 9", (a_, b_, lo, hi_), (p["full"], p["minus"], min(p["random20"]), max(p["random20"])))
okS = [k for k, v in FA.items() if v["kappa_sole"] is not None and v["kappa_sole"] >= .4]
feats8 = lambda i: {k for k in okS if T[i].get(k)} | {"task=" + str(T[i].get("task"))}
def r8(dol):
    tr = [i for i in A["traffic"] if i in T]; do = [i for i in dol if i in T]; ct = Counter(x for i in tr for x in feats8(i)); cd = Counter(x for i in do for x in feats8(i))
    return sorted(((ct[x] / len(tr) - cd[x] / len(do), x) for x in ct), reverse=True)
P8F = {"safety": ["harmful_request", "adversarial_pressure"], "code": ["task=coding", "agentic_or_tool_use"], "multilingual": ["non_english"]}
for F_ in FAMS:
    a1 = r8(ARMS[f"minus_{F_}"]); a2 = r8(ARMS[f"random_{F_}"])
    w("", f"**{F_}** ({P8F[F_]} count as the kind; full ranking of 37 items). Top 10 after the removal vs after the stored random removal:", "", "| rank | after planted removal | diff | after random removal | diff |", "|---|---|---|---|---|")
    for r_ in range(10): w(f"| {r_ + 1} | {esc(a1[r_][1])}{' **(kind)**' if a1[r_][1] in P8F[F_] else ''} | {100 * a1[r_][0]:+.1f} | {esc(a2[r_][1])}{' **(kind)**' if a2[r_][1] in P8F[F_] else ''} | {100 * a2[r_][0]:+.1f} |")
w("", "These tags were chosen to include the removed kinds (slide 9's own caveat).")

# ------- 10. NBER
print("section 10", flush=True)
h(2, "10. NBER categories (part 21)")
rng = fresh()
NS = jr(G / "nber_topic/summary.json")
chk("NBER code rank", "slide 9: 23 to 2, randoms 23", (23, 2, 23, 23), (NS["fine/code"]["kind"]["full"], NS["fine/code"]["kind"]["minus"], min(NS["fine/code"]["kind"]["random20"]), max(NS["fine/code"]["kind"]["random20"])))
mm_ = NS["fine/multilingual"]["max_drop"]; print(str(mm_)[:400])
cdt = Counter(NB.get(i) for i in REM["code"]); sft = Counter(NB.get(i) for i in REM["safety"]); mlt = Counter(NB.get(i) for i in REM["multilingual"])
chk("removed code tagged computer_programming", "B9: 383 of 397", 383, cdt["computer_programming"])
for c, n in [("specific_info", 20), ("how_to_advice", 17), ("other", 17), ("write_fiction", 12)]: chk(f"removed safety in {c}", "B9", n, sft[c])
aya = [i for i in REM["multilingual"] if SRC[i] == "Aya"]; chk("removed Aya tagged specific_info", "B9: 55 of 98", (55, 98), (sum(NB.get(i) == "specific_info" for i in aya), sum(i in NB for i in aya)))
chk("NBER tagged", "B9: 3,980 of 4,000", 3980, len(NB))
def nb_rank(cat, dol):
    tr = [i for i in A["traffic"] if i in NB]; do = [i for i in dol if i in NB]; ct = Counter(NB[i] for i in tr); cd = Counter(NB[i] for i in do)
    return {c: r for r, c in enumerate(sorted(set(NB.values()), key=lambda c: -(ct[c] / len(tr) - cd[c] / len(do))), 1)}[cat]
chk("NBER specific_info rank, full to minus multilingual", "slide 9: 21 to 10", (21, 10), (nb_rank("specific_info", ARMS["full"]), nb_rank("specific_info", ARMS["minus_multilingual"])))
chk("NBER how_to_advice rank, full to minus safety", "B9: 6 to 5", (6, 5), (nb_rank("how_to_advice", ARMS["full"]), nb_rank("how_to_advice", ARMS["minus_safety"])))
chk("removed safety tagged", "B9: 104", 104, sum(i in NB for i in REM["safety"]))
nb_tr = [i for i in A["traffic"] if i in NB]
def nb_share(cat, ids):
    ids = [i for i in ids if i in NB]; return sum(NB[i] == cat for i in ids), len(ids)
w("Sonnet applied the paper's prompt. `tags.json` holds one fine category per prompt. Slide 9: code found at the fine level; safety not found; non-English shows up as `specific_info`.", "", f"All removed safety prompts by landing category: {dict(sft.most_common())}.", "", f"All removed non-English prompts by landing category: {dict(mlt.most_common())}.", "", f"All removed code prompts by landing category: {dict(cdt.most_common())}.")
for cat, lab_ in [("computer_programming", "moved by the code removal"), ("specific_info", "moved by the non-English removal"), ("how_to_advice", "largest drop under the safety removal"), ("write_fiction", "largest natural difference"), ("mathematical_calculation", "most Dolci-heavy")]:
    a_, b_ = nb_share(cat, nb_tr); c_, d_ = nb_share(cat, ARMS["full"])
    setline(f"Category {cat} ({lab_})", f"traffic {pct(a_, b_)}; Dolci {pct(c_, d_)}.", "`runs/gap_taxonomy/nber_topic/tags.json`", "5 uniform traffic, 5 uniform Dolci, seed 0")
    for i in pick(rng, [x for x in nb_tr if NB[x] == cat], 5): ex(i, f"traffic, {tsrc(i)}, lang {lang(i)}")
    for i in pick(rng, [x for x in ARMS["full"] if NB.get(x) == cat], 5): ex(i, f"Dolci, {SRC[i]}, family {FAM[i]}, lang {lang(i)}")
for cat in ["specific_info", "how_to_advice", "other", "write_fiction"]:
    pool = [i for i in REM["safety"] if NB.get(i) == cat]
    setline(f"Removed safety prompts that landed in {cat}", f"{len(pool)} of 104 tagged; the taxonomy has no harm category.", "`nber_topic/tags.json`, `removal_arms.json`", "5 uniform, seed 0")
    for i in pick(rng, pool, 5): ex(i, rm_label(i))
pool = [i for i in REM["multilingual"] if NB.get(i) == "specific_info"]
setline("Non-English prompts tagged specific\\_info (removed)", f"{len(pool)} of {len(REM['multilingual'])}.", "same files", "8 uniform, seed 0")
for i in pick(rng, pool, 8): ex(i, f"{SRC[i]}, lang {lang(i)}")
pool = [i for i in A["traffic"] if lang(i) != "en" and NB.get(i) == "specific_info"]
setline("Non-English traffic prompts tagged specific\\_info", f"{len(pool)} of {sum(lang(i) != 'en' and i in NB for i in A['traffic'])} tagged non-English traffic prompts.", "`nber_topic/tags.json`, `blindspots/prompts.parquet`", "5 uniform, seed 0")
for i in pick(rng, pool, 5): ex(i, f"{tsrc(i)}, lang {lang(i)}")
pool = [i for i in REM["multilingual"] if i in NB and NB[i] != "specific_info"]
setline("Against the claim: removed non-English prompts in other categories", f"{len(pool)} of {len(REM['multilingual'])}.", "same files", "5 uniform, seed 0")
for i in pick(rng, pool, 5): ex(i, f"{NB[i]}, lang {lang(i)}")
pool = [i for i in REM["code"] if NB.get(i) != "computer_programming"]
setline("Against the claim: removed code prompts not tagged computer\\_programming", f"all {len(pool)} of {len(REM['code'])}.", "same files", "all")
for i in pool: ex(i, f"{NB.get(i, 'untagged')}, {SRC[i]}")
un = sorted(i for i in full | set(A["traffic"]) if i not in NB)
setline("The prompts with no NBER label", f"{len(un)} prompts: one batch of 20 refused by the labeller's safeguards.", "`nber_topic/tags.json`", "5 uniform, seed 0")
for i in pick(rng, un, 5): ex(i, "untagged")

# ------- 11. part 23
print("section 11", flush=True)
h(2, "11. Moderation taxonomy (part 23)")
rng = fresh()
LS = jr(G / "lang_safety_tax/summary.json"); kt = Counter(KO[i] for i in A["traffic"]); kd = Counter(KO[i] for i in ARMS["full"])
w("KoalaAI/Text-Moderation, argmax class, first 2,000 characters; the classes are OpenAI's 2023 moderation categories. Summary: `runs/gap_taxonomy/lang_safety_tax/summary.json`.", "",
  f"- Traffic labels: {dict(kt.most_common())}. Dolci labels: {dict(kd.most_common())}.",
  f"- Validation on 5,000 traffic prompts against real OpenAI moderation labels: flagged kappa {LS['validation']['kappa_flagged']} (gate 0.4); per-category {({k: v['kappa'] for k, v in LS['validation']['per_category'].items()})}.",
  f"- Ceiling: {LS['ceiling']['flagged']} of {LS['ceiling']['removed_safety']} removed safety prompts are flagged by this classifier.",
  f"- Test: safety moved {LS['safety']['max_drop']['moved']}, code moved {LS['code']['max_drop']['moved']}, multilingual moved {LS['multilingual']['max_drop']['moved']}; best harm rank 1 in every arm.", "")
chk("removed safety flagged by Koala, recomputed", "part 23 summary", 0, sum(KO[i] != "OK" for i in REM["safety"]))
for cls in ["S", "H", "V", "SH"]:
    tr = sorted(i for i in A["traffic"] if KO[i] == cls); do = sorted(i for i in ARMS["full"] if KO[i] == cls)
    setline(f"Class {cls}", f"traffic {len(tr)}, Dolci {len(do)}.", "`lang_safety_tax/labels.json`", "5 uniform traffic, 5 uniform Dolci, all if fewer, seed 0")
    for i in pick(rng, tr, 5): ex(i, f"traffic, {tsrc(i)}, Sonnet explicit\\_sexual {'yes' if T.get(i, {}).get('explicit_sexual') else 'no'}, harmful\\_request {'yes' if T.get(i, {}).get('harmful_request') else 'no'}")
    for i in pick(rng, do, 5): ex(i, f"Dolci, {SRC[i]}, family {FAM[i]}")
setline("Removed safety prompts, all labelled OK by this classifier, uniformly random", "the ceiling: 0 of 105 are flagged, so this test could not see the removal.", "`lang_safety_tax/labels.json`, `removal_arms.json`", "8 uniform, seed 0")
for i in pick(rng, REM["safety"], 8): ex(i, rm_label(i))
pool = [i for i in REM["safety"] if T.get(i, {}).get("harmful_request")]
setline("Chosen: removed safety prompts Sonnet tags harmful but the classifier labels OK", f"{len(pool)} of 105.", "`facets/tags.jsonl`, `lang_safety_tax/labels.json`", "5 uniform, seed 0")
for i in pick(rng, pool, 5): ex(i, rm_label(i))
pool = [i for i in A["traffic"] if T.get(i, {}).get("explicit_sexual") and KO[i] == "OK"]
setline("Against the classifier: traffic tagged explicit\\_sexual by Sonnet, labelled OK", f"{len(pool)} prompts.", "`facets/tags.jsonl`, `lang_safety_tax/labels.json`", "5 uniform, seed 0")
for i in pick(rng, pool, 5): ex(i, f"{tsrc(i)}, Koala OK")
pool = [i for i in A["traffic"] if not T.get(i, {}).get("explicit_sexual") and KO[i] == "S"]
setline("Traffic labelled S by the classifier, not tagged explicit\\_sexual by Sonnet", f"{len(pool)} prompts.", "same files", "5 uniform, seed 0")
for i in pick(rng, pool, 5): ex(i, f"{tsrc(i)}, Koala S")

# ------- assemble
print("assemble", flush=True)
body = L; L = []
w("# Real prompts behind each way of labelling gaps (slides 7-9, B7-B9)", "", "Scope: traffic is LMSYS and WildChat first user turns; Dolci is Olmo-3-7B's post-training data. No API, LLM or GPU call was made for this file. Every prompt is verbatim from a saved file, cut to 300 characters; newlines appear as ⏎, and markdown symbols are backslash-escaped.")
h(2, "Problems found")
w(PROBLEMS.read_text() if PROBLEMS.exists() else "(not written yet)")
bad = [c for c in CHECK if not c[4]]
w("", f"Automatic number checks: {len(CHECK)} slide numbers compared with the saved files, {len(bad)} differ." + (" Differences:" if bad else ""), "")
for c in bad: w(f"- {esc(c[0])}: slide or note says {c[2]} ({c[1]}); files give {c[3]}.")
h(2, "How to read this file")
w(f"- Script: `{SCRIPT}`. Random sets: uniform draw without replacement from a sorted pool, `np.random.default_rng({SEED})` created fresh at the start of each section. Rerunning gives the same draws.",
  "- Each set states what it shows, its source file, and how it was picked. \"Chosen\" and \"Against the claim\" are labelled.",
  f"- **[EXPLICIT]** marks prompts flagged as sexual or hateful, violent or self-harm content by the Sonnet tag, the Koala label, OpenAI moderation, or a keyword list. The keyword list over-marks. {len(FLAGGED)} distinct prompts carry it.",
  f"- {len(set(WITHHELD))} prompts have their text withheld here: flagged sexual and containing a word that suggests a minor. They are listed with their id so they can be read at the source. Ids: {sorted(set(WITHHELD))}.",
  "- The screen for withheld text is a keyword match and over-withholds: some of these may be false positives.",
  "- Ids: `t123` = row 123 of `runs/negatives/index/traffic_rows.jsonl`; `d123` = idx 123 of `runs/acts_dolci/shard*.jsonl`; `L...` prompt k = item and position in `runs/loop_labels/page_items.json`.")
h(2, "Contents")
for s_ in ["1. The loop's confirmed gaps", "2. Loop precision", "3. Property tags and pairs", "4. Contrastive properties", "5. Emergent clusters and the grouping test", "6. InsTag", "7. HypotheSAEs", "8. LLooM", "9. Planted removals", "10. NBER categories", "11. Moderation taxonomy (part 23)", "12. Number checks"]: w(f"- {s_}")
L.extend(body)
h(2, "12. Number checks")
w("Every number compared by the script. Automatic comparison only: a match means the slide number equals what the saved file gives.", "", "| claim | where | stated | files | match |", "|---|---|---|---|---|")
for c in CHECK: w(f"| {esc(c[0])} | {esc(c[1])} | {esc(str(c[2]))} | {esc(str(c[3]))} | {'yes' if c[4] else '**no**'} |")
OUT.parent.mkdir(exist_ok=True); OUT.write_text("\n".join(L) + "\n")
print("wrote", OUT, OUT.stat().st_size, "withheld", len(set(WITHHELD)), "flagged", len(FLAGGED))
print("MISMATCH"); [print(c) for c in bad]
