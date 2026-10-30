"""Safety-removal amendment 1: OpenAI moderation labels for the traffic pool. The index (runs/negatives/index/traffic_rows.jsonl) has
text and source but no conversation id, so labels are joined by normalised first-user-turn text (text_hash of normalise, as the index)
from LMSYS-Chat-1M (all 6 shards, `openai_moderation` of the first user turn) and WildChat-1M (all 14 shards, `openai_moderation` of the
first user turn). Duplicate texts: first occurrence kept, disagreements on `flagged` counted. Output runs/safety_removal/moderation/
labels.parquet (traffic row, matched, flagged, categories) and join.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import HfApi, hf_hub_download
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); OUT = ROOT / "runs/safety_removal/moderation"
from neg_index import normalise, text_hash
th = np.load(ROOT / "runs/negatives/index/traffic_hash.npy"); want = set(th.tolist())
lab, dis, seen_n = {}, 0, {"lmsys": 0, "wildchat": 0}
def first_user_mod(conv, mods):
    for k, m in enumerate(conv):
        if m["role"] == "user":
            mm = mods[k] if mods is not None and k < len(mods) else None
            return m["content"], mm
    return None, None
for repo, name, cols in (("lmsys/lmsys-chat-1m", "lmsys", ["conversation", "openai_moderation"]), ("allenai/WildChat-1M", "wildchat", ["conversation", "openai_moderation"])):
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    for f in files:
        d = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"), columns=cols)
        for conv, mods in zip(d.conversation, d.openai_moderation):
            t, m = first_user_mod(conv, mods)
            if not t or m is None: continue
            h = text_hash(normalise(t.strip()))
            if h not in want: continue
            seen_n[name] += 1
            fl = bool(m["flagged"]); cats = [c for c, v in m["categories"].items() if v]
            if h in lab:
                dis += lab[h][0] != fl; continue
            lab[h] = (fl, ",".join(sorted(cats)), name)
        print(name, f, len(lab), flush=True); del d
rows = [{"row": i, "matched": int(h) in lab, "flagged": lab[int(h)][0] if int(h) in lab else None,
         "categories": lab[int(h)][1] if int(h) in lab else None, "mod_source": lab[int(h)][2] if int(h) in lab else None} for i, h in enumerate(th)]
L = pd.DataFrame(rows); L.to_parquet(OUT / "labels.parquet")
src = [json.loads(l)["src"] for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]; L["src"] = src
J = {"traffic_rows": len(L), "matched": int(L.matched.sum()), "matched_by_src": L.groupby("src").matched.mean().round(4).to_dict(),
     "flagged": int((L.flagged == True).sum()), "flag_disagreements_among_duplicate_texts": int(dis), "conversations_hitting_pool": seen_n}
(OUT / "join.json").write_text(json.dumps(J, indent=1)); print(json.dumps(J, indent=1))
