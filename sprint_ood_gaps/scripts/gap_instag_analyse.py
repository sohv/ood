"""Part 12, InsTag arm: tag frequencies in traffic vs each Dolci arm (runs/gap_taxonomy/removal_arms.json).
Tags from OFA-Sys/InsTagger (pod output on HF detectood/ood results/raw/261029_instag_v3/tags.jsonl (v2 in instag/v2/, see Part 12 amendment), downloaded to runs/gap_taxonomy/instag/).
Tags parsed from the JSON, or (when the JSON is malformed) every "tag": "..." string in the raw output. Normalisation: lowercase, non-alphanumerics to spaces, collapse spaces. Rule fixed before reading any tag: rank tags with at least 10 traffic
prompts by (share of traffic prompts with the tag) minus (share of the arm's Dolci prompts with it); top 10 per arm. Writes top10.json, summary.json."""
import json, re
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"; O = D / "instag"
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()
tags = {}
for l in open(O / "tags.jsonl"):
    r = json.loads(l)
    ts = r["tags"] or re.findall(r'"tag"\s*:\s*"([^"]+)"', r["raw"] or "")  # lenient parse: malformed JSON (repeated keys, truncation) still names its tags
    if ts: tags[r["id"]] = {norm(t) for t in ts if norm(t)}
A = json.load(open(D / "removal_arms.json")); tra = [i for i in A["traffic"] if i in tags]
ct = Counter(t for i in tra for t in tags[i]); keep = {t for t, c in ct.items() if c >= 10}
top, summ = {}, {"n_tagged": len(tags), "n_traffic": len(tra), "n_unique_tags": len({t for v in tags.values() for t in v}), "arms": {}}
for arm, ids in A["dolci_arms"].items():
    ids = [i for i in ids if i in tags]; cd = Counter(t for i in ids for t in tags[i])
    diff = sorted(((ct[t] / len(tra) - cd[t] / len(ids), t) for t in keep), reverse=True)[:10]
    top[arm] = [t for _, t in diff]
    summ["arms"][arm] = {"n_dolci": len(ids), "top10": [{"tag": t, "traffic": round(ct[t] / len(tra), 4), "dolci": round(cd[t] / len(ids), 4)} for _, t in diff]}
json.dump(top, open(O / "top10.json", "w"), indent=1); json.dump(summ, open(O / "summary.json", "w"), indent=1)
for arm, v in summ["arms"].items(): print(arm, v["n_dolci"], [(x["tag"], x["traffic"], x["dolci"]) for x in v["top10"][:6]])
