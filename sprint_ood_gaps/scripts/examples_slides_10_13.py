"""Real examples behind main slides 10-13, from saved files only (no API, no GPU). Each section draws uniformly at random (numpy default_rng,
the seed is printed per section) before any content check, so a withheld row stays listed. Explicit sexual text and slurs are not
reproduced: the row keeps its id and a marker. Writes notes/examples/name1_limits_slides_10_13.md (body); "Problems found" is added by hand."""
import ast, glob, json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "notes/examples"; OUT.mkdir(parents=True, exist_ok=True)
CUT = 300
SEX = re.compile(r"\b(sex\w*|erotic\w*|porn\w*|nude\w*|naked|cum\w*|horny|fetish\w*|bdsm|kink\w*|seduc\w*|lust\w*|breasts?|boobs?|tits?|dick|cock|pussy|orgasm\w*|nsfw|lewd|slut\w*|whore\w*|undress\w*|arous\w*|moan\w*|smut|vibrator|fellat\w*|masturbat\w*|genital\w*|penis|vagina\w*|anal|rape\w*|incest|bimbo\w*|diaper\w*|dominatrix|foreplay|intimate|bedroom|tesão|fode\w*|putaria|buttocks|nipple\w*|hardcore|blowjob\w*|deepthroat\w*|threesome|bbc|jav|thrust\w*|erection|panties|lingerie|pee\w*|poop\w*)\b", re.I)
MINOR = re.compile(r"\b(child\w*|kid|kids|minor|minors|underage|teen\w*|schoolgirl\w*|schoolboy\w*|loli\w*|little girl|little boy|\d{1,2}[- ]year[- ]old|preteen)\b", re.I)
SLUR = re.compile(r"\b(nigg\w*|fag\w*|retard\w*|kike\w*|chink\w*|tranny)\b", re.I)
withheld = {"n": 0}
MANUAL = ["nippies", "hardened length", "trample", "firehose", "asian package", "half yokai", "break her spirit", "strip h", "exotic experience", "bosoms", "sadistic"]  # added after reading the first output
def show(t):
    t = " ".join(str(t or "").split()); sex = bool(SEX.search(t))
    if not t: return "(empty in the saved file)"
    if MINOR.search(t): withheld["n"] += 1; return "[text withheld: mentions a minor; not reproduced whatever the rest says]"
    if sex or any(m in t.lower() for m in MANUAL): withheld["n"] += 1; return "[text withheld: explicit sexual or fetish content]"
    if SLUR.search(t): withheld["n"] += 1; return "[text withheld: contains a slur]"
    return t[:CUT] + ("…" if len(t) > CUT else "")
md = []
def H(s): md.append("\n" + s + "\n")
def L(s): md.append(s)
def pick(seq, n, seed):
    seq = list(seq); idx = np.random.default_rng(seed).permutation(len(seq))[:n]; return [seq[i] for i in idx]
from refusal import is_refusal

# ---------- slide 10: probe pairs ----------
H("# Slide 10: NAME_1")
E = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/fix_loop/data/eval.jsonl")}
def rolls(p):
    out = {}
    for l in open(p):
        r = json.loads(l); out.setdefault(r["prompt_id"], []).append(r["reply"])
    return out
R = {"base": rolls(ROOT / "runs/fix_loop/out/eval_base/rollouts.jsonl"), "fix": rolls(ROOT / "runs/fix_loop/out/eval_lora/fix/rollouts.jsonl")}
PP = pd.read_csv(ROOT / "runs/fix_loop/per_prompt.csv"); share = {(a, p): v for a, p, v in zip(PP.arm, PP.prompt_id, PP.refused)}
pairs = sorted({i.split("__")[0] for i in E if i.endswith("__orig") and E[i]["set"] == "probe_placeholder"})
def pair_block(b):
    o, n = b + "__orig", b + "__names"
    L(f"- **{b}** (`runs/fix_loop/data/eval.jsonl`, `out/eval_base`, `out/eval_lora/fix`; refused share of 4 replies: token base {share[('base', o)]:.2f}, real names base {share[('base', n)]:.2f}, token fix {share[('fix', o)]:.2f})")
    L(f"  - Token prompt: {show(E[o]['prompt'])}")
    L(f"  - Real-name prompt: {show(E[n]['prompt'])}")
    L(f"  - Base reply to token: {show(R['base'][o][0])}")
    L(f"  - Base reply to real names: {show(R['base'][n][0])}")
    L(f"  - Fix reply to token: {show(R['fix'][o][0])}")
H(f"## 10a. Probe pairs: same prompt with NAME_k vs real names ({len(pairs)} pairs)")
L("Random 6 pairs, seed 0 (`scripts/examples_slides_10_13.py`). Shows: what the token does to Olmo's reply on one prompt, and what the fix changes.")
for b in pick(pairs, 6, 0): pair_block(b)
gap = sorted(pairs, key=lambda b: share[("base", b + "__names")] - share[("base", b + "__orig")])
H("Chosen (labelled chosen): the 3 pairs with the largest token-minus-real-name gap in the base model. Shows: the clearest cases, not typical ones.")
for b in gap[:3]: pair_block(b)
ag = [b for b in pairs if share[("base", b + "__names")] >= share[("base", b + "__orig")]]
st = [b for b in pairs if share[("fix", b + "__orig")] >= 0.5]
H(f"### Against the claim: {len(ag)} of {len(pairs)} pairs show no token effect in the base model (real names refused at least as often); {len(st)} of {len(pairs)} still refused at least half the time after the fix")
L("Random 5 of the no-effect pairs, seed 1. Shows: the 0.49 vs 0.13 is an average; many single prompts do not move.")
for b in pick(ag, 5, 1): pair_block(b)
L("\nRandom 3 of the still-refused pairs, seed 2.")
for b in pick(st, 3, 2): pair_block(b)

# ---------- 10b: Dolci NAME_k with training answers ----------
D = pd.concat([pd.read_parquet(p, columns=["hash", "part", "source", "refused", "prompt", "reply"]) for p in sorted(glob.glob(str(ROOT / "runs/triggers/dolci_labelled/*.parquet")))])
NK = D[D.prompt.str.contains(r"NAME_\d", regex=True)].reset_index(drop=True); NK["hash"] = NK["hash"].astype(str)
def dolci_block(r):
    L(f"- **{r['hash']}** ({r['source']}, {r['part']}; `runs/triggers/dolci_labelled`)")
    L(f"  - Prompt: {show(r['prompt'])}")
    L(f"  - Training answer: {show(r['reply'])}")
H(f"## 10b. Dolci's NAME_k prompts with their training answers ({len(NK)} prompts, {int(NK.refused.sum())} refused by the keyword rule)")
L("Refused, random 6, seed 3. Shows: what the refused NAME_k rows in the training data look like.")
for _, r in pd.DataFrame(pick(NK[NK.refused].to_dict("records"), 6, 3)).iterrows(): dolci_block(r)
H("### Against the claim: answered NAME_k prompts, random 6, seed 4")
L("Shows: Dolci also answers NAME_k prompts, so the training signal is a rate, not a rule.")
for _, r in pd.DataFrame(pick(NK[~NK.refused].to_dict("records"), 6, 4)).iterrows(): dolci_block(r)

# ---------- 10c: harmful prompts with token ----------
G = pd.read_json(ROOT / "runs/fix2/graded/graded.jsonl", lines=True); G = G[G.graded == True]
maj = G.groupby("prompt_id").request_harmful.agg(lambda s: (s == "yes").mean())
harm = G[G.prompt_id.isin(maj[maj > 0.5].index) & (G.role == "member") & (G.has_name == True)]
tab = harm.groupby(["prompt_id", "set", "model"]).refused.mean().unstack("model").reset_index()
pool = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/fix2/pool.jsonl")}
drop = tab[tab["fix"] < tab["base"]]
H(f"## 10c. Harmful prompts with the token: refusal after the fix ({len(tab)} prompts by majority grader label over replies; {len(drop)} with lower refusal under the fix than base; `runs/fix2/graded/graded.jsonl`, `runs/fix2/pool.jsonl`)")
L(f"Reply-level refusal: {harm.groupby('model').refused.mean().round(3).to_dict()}. Random 6 of the dropped, seed 5. Shows: the safety cost on prompt text, where the text can be shown.")
def harm_block(r):
    L(f"- **{r['prompt_id']}** ({r['set']}; refused share base {r['base']:.2f}, control {r['control']:.2f}, fix {r['fix']:.2f})")
    L(f"  - Prompt: {show(pool.get(r['prompt_id'], {}).get('text', '(not in pool.jsonl)'))}")
for _, r in pd.DataFrame(pick(drop.to_dict("records"), 6, 5)).iterrows(): harm_block(r)
keep = tab[tab["fix"] >= 1.0]
H(f"### Against the claim: {len(keep)} of {len(tab)} of these prompts are still fully refused after the fix, random 5, seed 6")
for _, r in pd.DataFrame(pick(keep.to_dict("records"), 5, 6)).iterrows(): harm_block(r)
L("\nAlso against: the fix2 review found that of 10 distinct compliances, 3 were benign prompts the grader mislabelled harmful (`runs/review/fix2_check_2026-10-24.md`).")

# ---------- 10d: finder words beyond the top 30 ----------
S = json.load(open(ROOT / "runs/gap_taxonomy/answers_14d/sheet.json")); A = json.load(open(ROOT / "runs/gap_taxonomy/answers_14d/reader_cb.json")); B = json.load(open(ROOT / "runs/gap_taxonomy/answers_14d/reader_sole.json"))
def word_block(it):
    w = it["word"]; v = A[w] if A[w] == B[w] else f"readers disagree ({A[w]} / {B[w]})"
    L(f"- **{w}** (rank {it['rank']}, {v}; test {it['k_test']}, traffic {it['traffic_count']}, main source {it['sources'][0][0]}; `runs/gap_taxonomy/answers_14d/sheet.json`)")
    L(f"  - Example: {show(it['examples'][0])}")
rest = [it for it in S if it["rank"] > 30]
H(f"## 10d. Answer-side finder: words beyond the top 30 (the top 30 are in `runs/gap_taxonomy/answers_14d/top30_examples.md`)")
L("Random 8 of ranks 31-100, seed 7. Shows: the middle of the list is mostly intended refusals.")
for it in pick(rest, 8, 7): word_block(it)
H("Chosen (labelled chosen): every word both readers call content-free in ranks 31-100. Shows: the candidates a behaviour test would start from.")
for it in rest:
    if A[it["word"]] == B[it["word"]] == "content-free cue": word_block(it)
H("### Against the claim: harmless requests the finder flags from Dolci's Tool Use source (chosen), and the words the readers disagree on")
L("Shows: \"carbonara\" and \"scorers\" are harmless requests refused in training, because Tool Use answers decline requests outside a listed API, whose system message the first-turn view hides. These are not adult or harmful content, so \"mostly intended refusals\" does not cover them.")
for it in S:
    if it["word"] in ("carbonara", "scorers"): word_block(it)
L("\nDisagreements: " + ", ".join(f"{it['word']} ({A[it['word']]} / {B[it['word']]})" for it in S if A[it["word"]] != B[it["word"]]) + ".")

# ---------- 10e: within WildGuardMix ----------
W = D[D.source == "WildGuardMix"].reset_index(drop=True); W["nk"] = W.prompt.str.contains(r"NAME_\d", regex=True); W["hash"] = W["hash"].astype(str)
H(f"## 10e. Within WildGuardMix: NAME_k {int(W[W.nk].refused.sum())}/{int(W.nk.sum())} refused vs {int(W[~W.nk].refused.sum())}/{int((~W.nk).sum())} without (`runs/gap_taxonomy/answers_14d/source_check.json`)")
L("Without NAME_k, refused, random 5, seed 8. Shows: what the source's other refusals look like.")
for _, r in pd.DataFrame(pick(W[~W.nk & W.refused].to_dict("records"), 5, 8)).iterrows(): dolci_block(r)
L("\nWithout NAME_k, answered, random 5, seed 9.")
for _, r in pd.DataFrame(pick(W[~W.nk & ~W.refused].to_dict("records"), 5, 9)).iterrows(): dolci_block(r)
H("### Against the claim: NAME_k rows may differ from the rest of the source in kind")
adv = lambda x: x.str.len().median()
L(f"Median prompt length with NAME_k {int(adv(W[W.nk].prompt.astype(str)))} characters, without {int(adv(W[~W.nk].prompt.astype(str)))}. Shows: the two groups are not matched; NAME_k rows come from one anonymised chat-log subset of the source, so 63% vs 41% can reflect that subset's request kinds, not the token alone. The comparison does not separate the two; only the minimal pairs on Olmo do.")

# ---------- slide 11 ----------
H("# Slide 11: what the loop cannot see")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
def parse_msgs(x):
    try: return ast.literal_eval(x) if isinstance(x, str) else list(x)
    except Exception: return json.loads(x)
rng = np.random.default_rng(10); res, seen_n = [], 0
for repo in ["lmsys--lmsys-chat-1m", "allenai--WildChat-1M"]:
    for f in sorted(glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/*.parquet")):
        for conv in pq.read_table(f, columns=["conversation"]).column("conversation").to_pylist():
            u = [m["content"] for m in conv if m["role"] == "user"]
            if len(u) < 2: continue
            seen_n += 1
            if len(res) < 6: res.append((repo, u))
            else:
                j = rng.integers(0, seen_n)
                if j < 6: res[j] = (repo, u)
H(f"## 11a. Real conversations with follow-up turns ({seen_n:,} conversations with 2+ user turns in the raw LMSYS and WildChat files)")
L("Reservoir sample of 6, seed 10. Shows: what the loop never scores, the turn after the first.")
for repo, u in res:
    L(f"- **{repo.split('--')[1]}**, {len(u)} user turns (HF cache, raw parquet)"); L(f"  - Turn 1: {show(u[0])}"); L(f"  - Turn 2: {show(u[1])}")
T, O_, Dp = [], [], []
for f in sorted(glob.glob(str(ROOT / "external/hf/dolci_answers/Dolci-Instruct-SFT/data/*.parquet"))):
    t = pd.read_parquet(f, columns=["id", "messages", "source_dataset"]); t = t[t.source_dataset.isin(["Dolci Instruct Tool Use", "OpenAssistant"])]
    for i, m, s in zip(t["id"], t["messages"], t["source_dataset"]):
        u = [x["content"] for x in parse_msgs(m) if x["role"] == "user"]
        if len(u) >= 2: (T if s == "Dolci Instruct Tool Use" else O_).append((i, u))
for f in sorted(glob.glob(str(ROOT / "external/hf/dolci_answers/Dolci-Instruct-DPO/data/*.parquet"))):
    t = pd.read_parquet(f, columns=["prompt_id", "chosen"])
    for i, m in zip(t["prompt_id"], t["chosen"]):
        u = [x["content"] for x in parse_msgs(m) if x["role"] == "user"]
        if len(u) >= 2: Dp.append((i, u))
H(f"## 11b. Dolci's multi-turn data: Tool Use {len(T):,}, OpenAssistant {len(O_):,}, DPO {len(Dp):,} conversations with 2+ user turns")
for name, rows, seed in (("Tool Use", T, 11), ("OpenAssistant", O_, 12), ("DPO", Dp, 13)):
    L(f"\n{name}, random 5, seed {seed}. Shows: how Dolci's follow-up turns compare with the real ones above.")
    for i, u in pick(rows, 5, seed):
        L(f"- **{i}** ({len(u)} user turns; `external/hf/dolci_answers`)"); L(f"  - Turn 1: {show(u[0])}"); L(f"  - Turn 2: {show(u[1])}")
H("### Against the claim")
L("Dolci is not without multi-turn data, and the slide's 4.0% vs 37.7% are different units: Dolci counts SFT conversations, traffic counts unique first-turn texts that occur in at least one multi-turn conversation (`scripts/gap_taxonomy_blindspots.py`). A first turn that appears once in a long conversation and many times alone counts as multi-turn.")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
P = pd.read_parquet(ROOT / "runs/gap_taxonomy/blindspots/prompts.parquet", columns=["topic", "member", "over_128", "minilm_tokens"])
texts = [json.loads(l)["text"] for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]
long_ids = np.flatnonzero(P.over_128.values & ~P.member.values)
def cut_block(i):
    t = texts[i][:1000]; off = tok(t, add_special_tokens=False, truncation=True, max_length=126, return_offsets_mapping=True)["offset_mapping"]; c = off[-1][1]
    L(f"- **traffic row {i}** ({P.minilm_tokens.values[i]} tokens; `runs/negatives/index/traffic_rows.jsonl`)")
    L(f"  - Last of what MiniLM reads: {show(t[max(0, c - 120):c])}"); L(f"  - First of what it never reads: {show(texts[i][c:c + 180])}")
H(f"## 11c. Prompts cut by MiniLM's 128-token window ({len(long_ids):,} non-member traffic prompts over the window)")
L("Random 6, seed 14. Shows: where the cut falls in real long prompts.")
for i in pick(long_ids, 6, 14): cut_block(int(i))
tail = [int(i) for i in pick(long_ids, 400, 15) if re.search(r"(summar|translate|rewrite|question|answer|explain)[^.]{0,80}[.?:]?\s*$", texts[i][-200:], re.I)]
H("Chosen (labelled chosen): 3 of a random 400 whose last 200 characters read as the request. Shows: cases where the embedding never sees the task.")
L(f"{len(tail)} of the 400 matched the pattern.")
for i in tail[:3]: cut_block(i)
H("### Against the claim")
L("Dolci has more prompts over the window than traffic (46.6% vs 28.9%), so the window does not favour one side, and in the random draws above the request often comes first, inside the window. The cut hides detail more often than it hides the task; this was not counted over all long prompts.")

RB = pd.read_csv(ROOT / "runs/robust/topics_both.csv"); mt, qt = set(RB.nsmallest(30, "rare_rank").lab), set(RB.nsmallest(30, "qwen_rank").lab)
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy"); mem = P.member.values
def topic_block(t, seed):
    r = RB[RB.lab == t].iloc[0]; ids = np.flatnonzero((lab == t) & ~mem)
    L(f"- **topic {t}** (MiniLM rank {int(r.rare_rank)}, Qwen rank {int(r.qwen_rank)}; `runs/robust/topics_both.csv`)")
    for i in pick(ids, 2, seed): L(f"  - {show(texts[int(i)])}")
H(f"## 11d. Topics that differ between embedders: top 30 share {len(mt & qt)} topics")
L("In MiniLM's top 30 only, random 5, seed 16. Shows: what one embedder calls far and the other does not.")
for k, t in enumerate(pick(sorted(mt - qt), 5, 16)): topic_block(t, 100 + k)
L("\nIn Qwen's top 30 only, random 5, seed 17.")
for k, t in enumerate(pick(sorted(qt - mt), 5, 17)): topic_block(t, 200 + k)
H("### Against the claim: the shared topics, and the unseparated causes")
for k, t in enumerate(sorted(mt & qt)): topic_block(t, 300 + k)
L("\nThe comparison changes both the model and the window (128 vs 512 tokens), so it does not show the model alone moves the list. Top-293 overlap is 89 against 43 by chance, Spearman 0.51: the lists are related.")

import gap_facets as F
TG = F._load(ROOT / "runs/gap_taxonomy/facets/tags.jsonl"); SO = F._load(ROOT / "runs/gap_taxonomy/facets/sole.jsonl")
SM = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/gap_taxonomy/facets/sample.jsonl")}; FS = json.load(open(ROOT / "runs/gap_taxonomy/facets/summary.json"))["facets"]
both = [i for i in SO if i in TG]
H(f"## 11e. Low-agreement properties ({len(both)} prompts tagged by both labellers; `runs/gap_taxonomy/facets/tags.jsonl`, `sole.jsonl`)")
for k, seed in (("hidden_difficulty", 18), ("buried_request", 19), ("difficult_advice", 20)):
    dis = [i for i in both if TG[i].get(k) != SO[i].get(k)]; yes = [i for i in both if TG[i].get(k) and SO[i].get(k)]
    L(f"\n**{k}** (kappa {FS[k]['kappa_sole']}; {len(dis)} disagreements, {len(yes)} both yes). Random disagreements, up to 5, seed {seed}. Shows: where the two labellers split.")
    for i in pick(dis, 5, seed): L(f"- **{i}** (Sonnet {'yes' if TG[i].get(k) else 'no'}, Sole {'yes' if SO[i].get(k) else 'no'}): {show(SM[i]['text'])}")
    if yes:
        L("Both yes, up to 3 (chosen by the agreement):")
        for i in pick(yes, 3, seed + 100): L(f"- **{i}**: {show(SM[i]['text'])}")
H("### Against the claim")
L("The check covers 200 prompts, so kappa on rare properties rests on few positives: low agreement here does not show the kinds cannot be tagged, only that these two labellers disagreed under this rubric with truncated input.")

# ---------- slide 12 ----------
H("# Slide 12: how we checked")
s = open(ROOT / "notes/audit_problems_2026-10-28.md").read(); sec = re.split(r"\n## ", s)[1]
items = re.findall(r"\n\d+\. \*\*(\S+)\*\* \(([^)]*)\)[^\n]*\n   - Problem: ([^\n]*)\n(?:   - Where: [^\n]*\n)?   - Verifier \(([^)]*)\): ([^\n]*)", sec)
def audit_block(x): L(f"- **{x[0]}** ({x[1][:70]}). Problem: {show(x[2])} Verifier ({x[3]}): {show(x[4])}")
H(f"## 12a. Audit findings that change a headline ({len(items)} parsed of 33; `notes/audit_problems_2026-10-28.md`)")
L("Random 5, seed 21. Shows: the kind of problem the audit found.")
for x in pick(items, 5, 21): audit_block(x)
H("Chosen (labelled chosen), with what each changed:")
ch = {"O09-1": "STATUS's safety-removal reading was corrected: the removal does register (change AUROC 0.71 vs 0.43 random).", "O07-1": "The NAME_1 fix's safety cost went onto the slide (refusal of harmful prompts with the token falls to 0.72-0.76).",
      "O02-1": "The old negatives eval (18 of 23) was dropped from the slides as unusable under its own rule.", "S13-9": "Sohan's \"15 new gaps vs 1\" is not used as a headline; the slides use the training-only re-run with 10 random removals."}
allit = re.findall(r"\n\d+\. \*\*(\S+)\*\* \(([^)]*)\)[^\n]*\n   - Problem: ([^\n]*)\n(?:   - Where: [^\n]*\n)?   - Verifier \(([^)]*)\): ([^\n]*)", s)
for x in allit:
    if x[0] in ch: audit_block(x); L(f"  - What it changed: {ch[x[0]]}")
over = [x for x in items if re.search(r"overstated|does not change|changes? no ", x[4], re.I)]
H(f"### Against the claim: {len(over)} of these {len(items)} verifier notes say the severity is overstated or no number changes; 8 findings were refuted; the audit was run by LLM agents")
for x in pick(over, 5, 22): audit_block(x)
stats = {}
stats["namek_rows"] = len(NK); stats["namek_refused"] = int(NK.refused.sum())
rl = NK.reply.fillna("").str.strip().str.len(); stats["namek_answer_empty"] = int((rl == 0).sum()); stats["namek_answer_under_5_chars"] = int(((rl > 0) & (rl < 5)).sum())
stats["namek_not_refused_empty_or_short"] = int(((rl < 5) & ~NK.refused).sum())
lt = pd.Series([texts[int(i)][:4000] for i in long_ids]); stats["over128_nonmember"] = len(long_ids); stats["over128_midjourney"] = int(lt.str.contains("prompt generator for a generative AI called", regex=False).sum())
tq = RB.nsmallest(30, "qwen_rank").lab.tolist(); stats["qwen_top30_chemical_intro_topics"] = sum(np.mean([("chemical company" in texts[int(i)]) for i in pick(np.flatnonzero((lab == t) & ~mem), 20, 400)]) >= 0.5 for t in tq)
tm = RB.nsmallest(30, "rare_rank").lab.tolist(); stats["minilm_top30_chemical_intro_topics"] = sum(np.mean([("chemical company" in texts[int(i)]) for i in pick(np.flatnonzero((lab == t) & ~mem), 20, 401)]) >= 0.5 for t in tm)
stats["raw_conversations_2plus_user_turns"] = seen_n
stats["raw_conversations_total"] = sum(pq.read_metadata(f).num_rows for repo in ["lmsys--lmsys-chat-1m", "allenai--WildChat-1M"] for f in glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/*.parquet"))
ap = D[D.prompt.str.contains("april ludgate", case=False, regex=False)]
stats["april_ludgate_rows"] = len(ap); stats["april_ludgate_refused"] = int(ap.refused.sum()); stats["april_ludgate_sources"] = ap.source.value_counts().to_dict(); stats["april_ludgate_answer_examples"] = [show(x) for x in ap[ap.refused].reply.head(2)]
stats["keyword_vs_capability_top3"] = {b: show(R["base"][b + "__orig"][0])[:120] for b in gap[:3]}
stats["facets_both_yes_disagree"] = {k: {"disagree": sum(TG[i].get(k) != SO[i].get(k) for i in both), "both_yes": sum(bool(TG[i].get(k) and SO[i].get(k)) for i in both), "sonnet_yes": sum(bool(TG[i].get(k)) for i in both), "sole_yes": sum(bool(SO[i].get(k)) for i in both)} for k in ("hidden_difficulty", "buried_request", "difficult_advice")}
json.dump(stats, open(OUT / "name1_limits_stats.json", "w"), indent=1, ensure_ascii=False, default=str); print(json.dumps(stats, indent=1, ensure_ascii=False, default=str))
md.append(f"\n---\nRows withheld by the content filter, all sections: {withheld['n']}.\n")
open(OUT / "name1_limits_slides_10_13.md", "w").write("# Real examples behind slides 10-13\n\nWritten for the interview, from saved files only (no API or GPU). Script: `scripts/examples_slides_10_13.py`. Verbatim text is cut to 300 characters. Explicit sexual text and slurs are replaced by a marker with the row id kept, never dropped.\n\n## Problems found\n\n{PROBLEMS}\n" + "\n".join(md))
print("written; withheld", withheld["n"], "audit parsed", len(items))
