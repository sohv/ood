"""Stage 2 (2026-10-26): harm re-read (Sol via codex, prompt only, same rubric) of 10 random grader refused-harmless prompts in each
stage-1 candidate cluster that is not NAME_k- or template-majority, plus 5 per NAME_k/template cluster as a check. A cluster whose
sampled refusals are mostly benign (>= 6 of 10) is a candidate new cue and gets read in full."""
import hashlib, json, re, subprocess, tempfile, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; S = ROOT / "runs/misbehave100k/stage1"; CACHE = ROOT / "runs/misbehave100k/analysis/sole_cache"
src = open(ROOT / "scripts/misbehave100k_analyse.py").read(); RUBRIC = eval(src[src.index("RUBRIC = (") + 9:src.index("def reread_one")].strip())
text = {p["id"]: p["text"] for f in ["runs/misbehave/prompts.jsonl", "runs/misbehave100k/prompts.jsonl"] for p in map(json.loads, open(ROOT / f))}
def sol(i):
    h = hashlib.md5(text[i].encode()).hexdigest() + ".json"
    if (CACHE / h).exists(): return json.loads((CACHE / h).read_text())["label"]
    with tempfile.TemporaryDirectory() as dd:
        for _ in range(3):
            try: subprocess.run(["codex", "exec", "-m", "gpt-6.1-sol", "-s", "read-only", "--skip-git-repo-check", "-o", f"{dd}/o.txt", "-"],
                                input=RUBRIC + text[i] + "\n\n(Do not use any tools.)", capture_output=True, text=True, cwd=dd, timeout=300)
            except subprocess.TimeoutExpired: continue
            t = Path(f"{dd}/o.txt").read_text() if Path(f"{dd}/o.txt").exists() else ""
            m = re.search(r'"label"\s*:\s*"(harmful|adult_or_persona|cannot_do|benign)"', t)
            if m: (CACHE / h).write_text(json.dumps({"label": m.group(1), "route": "codex"})); return m.group(1)
    return None
t = pd.read_csv(S / "clusters.csv"); c = t[(t.p_holm < 0.05) & (t.refused_harmless >= 5)].copy()
c["kind"] = ["name_k" if a >= 0.5 else "template" if b >= 0.5 else "other" for a, b in zip(c.name_k_share, c.template_share)]
d = pd.read_json(S / "rows.jsonl", lines=True); jobs = []
for _, r in c.iterrows():
    x = d[(d.cluster == r.cluster) & d.refused_harmless]; k = 10 if r.kind == "other" else 5
    for i in x.sample(min(k, len(x)), random_state=0).id: jobs.append((int(r.cluster), r.kind, i))
with ThreadPoolExecutor(6) as ex: labs = list(ex.map(lambda j: sol(j[2]), jobs))
o = pd.DataFrame([(a, b, i, l) for (a, b, i), l in zip(jobs, labs)], columns=["cluster", "kind", "id", "sol"]); o.to_csv(S / "stage2_labels.csv", index=False)
s = o.groupby(["kind", "cluster"]).sol.apply(lambda v: f"{(v == 'benign').sum()}/{len(v)} benign; " + ", ".join(f"{k} {n}" for k, n in v.value_counts().items())).reset_index()
s.to_csv(S / "stage2_summary.csv", index=False); print(s.to_string(index=False)); print("failed", int(o.sol.isna().sum()), "of", len(o))
